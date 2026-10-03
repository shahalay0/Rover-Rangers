"""Autonomous BPSK receiver agent.

No manual tuning: the agent finds the signal, estimates Doppler and drift,
recovers timing and carrier, detects lock loss, and re-acquires on its own.

  raw IQ ──► 1. activity detection (blind SNR per block)  ──► segments
         ──► 2. coarse Doppler per block (squaring + FFT), hop detection,
                drift fit, de-rotation
         ──► (optional) modulation classification, RadioML-trained
         ──► 3. matched filter (RRC) + AGC
         ──► 4. Gardner timing recovery
         ──► 5. Costas carrier loop + lock detector
         ──► 6. ASM frame sync (resolves 180° ambiguity) ──► CRC ──► telemetry
"""
import time
import numpy as np
from .transmitter import rrc_taps
from .telemetry import ASM_BITS, FRAME_BITS, decode_body


def loop_gains(bn, zeta=0.707, kd=1.0):
    theta = bn / (zeta + 1 / (4 * zeta))
    d = 1 + 2 * zeta * theta + theta ** 2
    return 4 * zeta * theta / (d * kd), 4 * theta ** 2 / (d * kd)


class SignalAgent:
    def __init__(self, sps=8, beta=0.35, block=1024, snr_gate_db=-12.0,
                 fft_block=4096, hop_thresh=2e-3, classifier=None, verbose=True):
        self.sps, self.beta = sps, beta
        self.block, self.snr_gate = block, 10 ** (snr_gate_db / 10)
        self.fft_block, self.hop_thresh = fft_block, hop_thresh
        self.events, self.verbose = [], verbose
        self.classifier = classifier          # optional ModulationClassifier trained on RadioML
        self.rrc = rrc_taps(sps, beta)

    def log(self, sample, msg):
        self.events.append((int(sample), msg))
        if self.verbose:
            print(f"[agent @ {int(sample):>8d}] {msg}")

    def say(self, msg):
        """Progress message only (printed when verbose, not stored as an event)."""
        if self.verbose:
            print(f"      ... {msg}")

    # ---------- 1. where is there a signal? ----------
    def blind_snr(self, x):
        """Per-block SNR without knowing the carrier: the signal occupies
        < 50% of the band, so the median FFT bin is a noise-floor estimate."""
        nb = len(x) // self.block
        blocks = x[:nb * self.block].reshape(nb, self.block) * np.hanning(self.block)
        p = np.abs(np.fft.fft(blocks, axis=1)) ** 2
        noise_bin = np.median(p, axis=1) / np.log(2)      # median of exp. dist = ln2*mean
        snr = (p.sum(1) - noise_bin * self.block) / (noise_bin * self.block)
        return np.maximum(snr, 1e-6)

    def find_segments(self, x, min_blocks=8):
        snr = self.blind_snr(x)
        active = snr > self.snr_gate
        segs, start = [], None
        for i, a in enumerate(np.append(active, False)):
            if a and start is None:
                start = i
            elif not a and start is not None:
                if i - start >= min_blocks:
                    segs.append((start * self.block, i * self.block))
                start = None
        return segs, snr

    # ---------- 2. Doppler: offset, drift, hops ----------
    def doppler_track(self, x):
        """BPSK squared has a pure tone at 2*f_c -> FFT peak gives f_c."""
        n, step, nfft = self.fft_block, self.fft_block // 2, 4 * self.fft_block
        centers, f = [], []
        for s in range(0, len(x) - n + 1, step):
            sq = (x[s:s + n] * np.hanning(n)) ** 2
            spec = np.abs(np.fft.fft(sq, nfft))
            k = np.argmax(spec)
            # parabolic interpolation for sub-bin accuracy
            a, b, c = spec[k - 1], spec[k], spec[(k + 1) % nfft]
            k_frac = k + 0.5 * (a - c) / (a - 2 * b + c)
            f2 = np.fft.fftfreq(nfft)[k] + (k_frac - k) / nfft
            centers.append(s + n / 2)
            f.append(f2 / 2)
        return np.array(centers), np.array(f)

    def split_on_hops(self, centers, f, seg_start):
        cuts = [0]
        for i in range(1, len(f) - 1):
            # a real hop: big jump that persists (not a single noisy outlier)
            if i - cuts[-1] < 2:
                continue
            if abs(f[i] - f[i - 1]) > self.hop_thresh and abs(f[i + 1] - f[i - 1]) > self.hop_thresh:
                cuts.append(i)
                self.log(seg_start + centers[i], f"frequency hop detected: "
                         f"{f[i-1]:+.4f} -> {f[i+1]:+.4f} cyc/sample, re-acquiring")
        cuts.append(len(f))
        return [(cuts[j], cuts[j + 1]) for j in range(len(cuts) - 1)]

    def fit_doppler(self, c, f):
        """Robust linear fit f(t) = f0 + rate*t (outlier rejection by MAD)."""
        if len(f) < 3:
            return np.median(f), 0.0
        keep = np.abs(f - np.median(f)) < 5 * (np.median(np.abs(f - np.median(f))) + 1e-6)
        rate, f0 = np.polyfit(c[keep], f[keep], 1)
        return f0, rate

    # ---------- 3-5. demodulation ----------
    def gardner(self, y, bn=0.005):
        kp, ki = loop_gains(bn, kd=2.0)
        sps, out = self.sps, []
        t, integ = float(sps), 0.0
        yl = y.tolist()                          # python list: fast scalar access in the loop

        def interp(tt):                          # linear interpolation between samples
            i = int(tt)
            fr = tt - i
            return yl[i] + (yl[i + 1] - yl[i]) * fr
        prev = interp(t - sps)
        while t + sps < len(y) - 1:
            cur, mid = interp(t), interp(t - sps / 2)
            e = np.real((prev - cur) * np.conj(mid))
            integ += ki * e
            t += sps + np.clip(kp * e + integ, -sps / 4, sps / 4)
            out.append(cur)
            prev = cur
        return np.array(out)

    def costas(self, s, bn=0.02):
        kp, ki = loop_gains(bn)
        phase, integ = 0.0, 0.0
        out = np.empty_like(s)
        for i, v in enumerate(s):
            z = v * np.exp(-1j * phase)
            out[i] = z
            e = np.sign(z.real) * z.imag
            integ += ki * e
            phase += kp * e + integ
        return out

    @staticmethod
    def lock_metric(sym, win=64):
        r2, i2 = sym.real ** 2, sym.imag ** 2
        k = np.ones(win) / win
        return np.convolve(r2 - i2, k, "same") / (np.convolve(r2 + i2, k, "same") + 1e-12)

    # ---------- 6. frames ----------
    def frame_sync(self, sym, thresh=0.6):
        soft = sym.real / (np.mean(np.abs(sym.real)) + 1e-12)
        asm = 1.0 - 2.0 * ASM_BITS
        corr = np.correlate(soft, asm, "valid") / len(asm)
        frames, i = [], 0
        while i < len(corr) - FRAME_BITS + 32:
            if abs(corr[i]) > thresh:
                j = i + np.argmax(np.abs(corr[i:i + 8]))          # local peak
                pol = np.sign(corr[j])                            # 180° ambiguity
                body = soft[j + 32:j + FRAME_BITS] * pol
                if len(body) == FRAME_BITS - 32:
                    frames.append((j, (body < 0).astype(np.uint8), abs(corr[j])))
                i = j + FRAME_BITS - 8
            else:
                i += 1
        return frames

    # ---------- the agent ----------
    def run(self, x):
        self.say(f"scanning {len(x)} samples for signal activity (blind SNR per {self.block}-sample block)")
        segs, snr = self.find_segments(x)
        self.say(f"found {len(segs)} active segment(s)")
        self.diag = dict(block_snr=snr, doppler=[], symbols=[], lock=[])
        if not segs:
            self.log(0, "no signal found")
        records = []
        for seg_id, (s0, s1) in enumerate(segs):
            seg_snr = 10 * np.log10(np.median(snr[s0 // self.block:s1 // self.block]))
            self.log(s0, f"signal acquired (segment {seg_id}, blind SNR ≈ {seg_snr:.1f} dB full-band)")
            xs = x[s0:s1]
            self.say(f"estimating Doppler on {len(xs)} samples (squaring + FFT per {self.fft_block}-sample block)")
            c, f = self.doppler_track(xs)
            for a, b in self.split_on_hops(c, f, s0):
                lo = 0 if a == 0 else int(c[a] - self.fft_block / 2)
                hi = len(xs) if b == len(f) else int(c[b - 1] + self.fft_block / 2)
                f0, rate = self.fit_doppler(c[a:b], f[a:b])
                self.log(s0 + lo, f"Doppler {f0 + rate*lo:+.5f} cyc/sample, drift {rate:+.2e} /sample")
                k = np.arange(lo, hi)
                phase = 2 * np.pi * (f0 * k + 0.5 * rate * k ** 2)
                z = xs[lo:hi] * np.exp(-1j * phase)
                self.diag["doppler"].append((s0 + c[a:b], f[a:b], s0 + k, f0 + rate * k))

                if self.classifier is not None:
                    label, conf, top = self.classifier.classify_signal(z)
                    self.diag.setdefault("modulation", []).append((s0 + lo, label, conf, top))
                    self.log(s0 + lo, f"modulation identified: {label} ({100*conf:.0f}% of windows)")
                    if label != "BPSK":
                        self.log(s0 + lo, f"no {label} demodulator yet, trying BPSK anyway")

                self.say(f"demodulating {hi - lo} samples (~{(hi - lo) // self.sps} symbols)")
                t0 = time.time()
                y = np.convolve(z, self.rrc, "same")
                y /= np.sqrt(np.mean(np.abs(y) ** 2))
                self.diag.setdefault("filtered", []).append(y)
                self.say("matched filter + AGC done, running Gardner timing recovery...")
                sym = self.gardner(y)
                sym /= np.mean(np.abs(sym))
                self.say(f"timing recovered {len(sym)} symbols ({time.time()-t0:.1f} s), running Costas carrier loop...")
                sym = self.costas(sym)
                lock = self.lock_metric(sym)
                self.diag["symbols"].append(sym)
                self.diag["lock"].append(lock)
                locked = np.mean(lock[len(lock) // 4:] > 0.3)
                self.log(s0 + hi, f"sub-segment done, carrier locked {100*locked:.0f}% of time")

                found = self.frame_sync(sym)
                self.say(f"frame sync found {len(found)} sync markers, checking CRC...")
                n_ok = 0
                for idx, body, q in found:
                    rec, ok = decode_body(body)
                    n_ok += ok
                    rec.update(crc_ok=ok, segment=seg_id, rx_sample=s0 + lo + idx * self.sps,
                               sync_quality=round(q, 3))
                    records.append(rec)
                self.say(f"{n_ok}/{len(found)} frames passed CRC ({time.time()-t0:.1f} s for this sub-segment)")
            self.log(s1, f"signal lost (end of segment {seg_id}) — returning to search")
        return records
