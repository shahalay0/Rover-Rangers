"""
Track 1: Autonomous deep-space telemetry receiver.

generate telemetry -> impaired channel (noise, Doppler drift, hop, dropout)
-> acquire/track carrier -> clean (mixdown + matched filter)
-> classify modulation (CNN trained on RadioML) -> demodulate
-> frame sync (CCSDS ASM) -> CRC check -> telemetry log

Run:  python pipeline.py                  BPSK test signal, plots + log in results/
      python pipeline.py --mod QPSK       QPSK test signal
      python pipeline.py --snr -2         choose SNR (dB per sample)
      python pipeline.py --sweep          also plot frame success vs SNR
The receiver is never told the modulation. It uses the teammate's notebook model
(outputs/iqnet_doppler_aug.pt) or models/modclass.pt, and falls back to a
simple rule if neither exists.
"""
import argparse, binascii, csv, os, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 48_000                  # sample rate (Hz)
SPS = 8                      # samples per symbol (same as RadioML) -> 6000 baud
BETA = 0.35                  # root-raised-cosine roll-off
ASM = 0x1ACFFC1D             # CCSDS attached sync marker
FRAME_FMT = ">HfffB"         # counter, time (s), battery (V), temp (C), status
PAYLOAD_BYTES = struct.calcsize(FRAME_FMT)
FRAME_BITS = 32 + 8 * (PAYLOAD_BYTES + 2)   # ASM + payload + CRC16
N_FRAMES = 120
BLOCK = 2048                 # samples per acquisition block
LOCK_THRESHOLD = 20          # tone strength needed to declare lock
HOLD_THRESHOLD = 13          # once locked, stay locked until tone drops below this (hysteresis)
OUT = "results"
BITS_PER_SYMBOL = {"BPSK": 1, "QPSK": 2}


def to_bits(b):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


ASM_BITS = to_bits(ASM.to_bytes(4, "big"))


def rrc_taps(span=8):
    """Root-raised-cosine pulse, the standard pulse shape (also used by RadioML)."""
    t = np.arange(-span * SPS // 2, span * SPS // 2 + 1) / SPS
    h = np.zeros_like(t)
    for i, x in enumerate(t):
        if np.isclose(x, 0):
            h[i] = 1 - BETA + 4 * BETA / np.pi
        elif np.isclose(abs(x), 1 / (4 * BETA)):
            h[i] = BETA / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * BETA))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * BETA)))
        else:
            h[i] = (np.sin(np.pi * x * (1 - BETA)) + 4 * BETA * x * np.cos(np.pi * x * (1 + BETA))) \
                   / (np.pi * x * (1 - (4 * BETA * x) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


RRC = rrc_taps()


# ---------------- Transmitter + channel (test harness) ----------------
def make_telemetry(n_frames, mod, rng):
    frames, truth = [], []
    for k in range(n_frames):
        t = k * FRAME_BITS / BITS_PER_SYMBOL[mod] * SPS / FS
        batt = 28.0 - 0.02 * k + rng.normal(0, 0.02)
        temp = 20 + 5 * np.sin(2 * np.pi * k / 60) + rng.normal(0, 0.1)
        status = 1 if batt > 26.5 else 3          # 3 = low battery flag
        payload = struct.pack(FRAME_FMT, k, t, batt, temp, status)
        crc = binascii.crc_hqx(payload, 0xFFFF)
        frames.append(ASM.to_bytes(4, "big") + payload + crc.to_bytes(2, "big"))
        truth.append((k, t, batt, temp, status))
    return to_bits(b"".join(frames)), truth


QPSK_STEP = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}   # Gray code: bits -> quarter turns


def modulate(bits, mod):
    """Differential encoding: data is carried by the CHANGE in phase."""
    if mod == "BPSK":
        steps = bits.astype(int) * 2                         # 0 -> no turn, 1 -> half turn
    else:
        steps = np.array([QPSK_STEP[tuple(p)] for p in bits.reshape(-1, 2)])
    phase = np.cumsum(np.concatenate([[0], steps])) * np.pi / 2
    up = np.zeros(len(phase) * SPS, complex)
    up[::SPS] = np.exp(1j * phase)
    x = np.convolve(up, RRC)
    return x / np.sqrt(np.mean(np.abs(x) ** 2))              # unit power


def channel(x, snr_db, rng):
    n = len(x)
    t = np.arange(n) / FS
    f = 1500 + 2500 * np.sin(2 * np.pi * t / 40)   # Doppler drift (up to ~390 Hz/s), stays bounded
    f[n // 2:] -= 4000                    # frequency hop halfway through
    y = x * np.exp(1j * (2 * np.pi * np.cumsum(f) / FS + rng.uniform(0, 2 * np.pi)))
    d0 = int(0.3 * n); d1 = d0 + int(0.15 * FS)
    y[d0:d1] = 0                          # 0.15 s dropout
    sigma = np.sqrt(10 ** (-snr_db / 10) / 2)
    y += sigma * (rng.normal(size=n) + 1j * rng.normal(size=n))
    return y, f, (d0 / FS, d1 / FS)


# ---------------- Receiver (the autonomous part) ----------------
def lowpass_taps(cutoff=11_000, n=101):
    """Windowed-sinc low-pass filter: removes noise far outside where the signal can be."""
    t = np.arange(n) - (n - 1) / 2
    h = np.sinc(2 * cutoff / FS * t) * np.hamming(n)
    return h / h.sum()


LOWPASS = lowpass_taps()


def find_tone(x, order, nfft):
    """Raise x to a power and look for a single sharp spectral line.
    Each bin is compared with its neighbours (+/- 600 Hz), so broad humps of
    noise or modulation do not count, only a real tone does."""
    spec = np.abs(np.fft.fft(x ** order, nfft)) ** 2
    w = int(600 * order / (FS / nfft))
    local = np.convolve(np.concatenate([spec[-w:], spec, spec[:w]]),
                        np.ones(2 * w + 1) / (2 * w + 1), mode="valid")
    ratio = spec / local
    p = np.argmax(ratio)
    return p, ratio[p]


def track_carrier(y, block=BLOCK):
    """Raising the signal to a power removes the modulation and leaves a tone:
    squaring works for BPSK (tone at 2f), 4th power for QPSK (tone at 4f).
    Try squaring first, then 4th power. No tone in either -> no signal."""
    nfft = 4 * block
    freqs = np.fft.fftfreq(nfft, 1 / FS)
    yf = np.convolve(y, LOWPASS, mode="same")
    f_est = np.zeros(len(y)); locked = np.zeros(len(y), bool)
    events, was_locked, f_now, orders = [], False, 0.0, []
    for s in range(0, len(y), block):
        lock = False
        need = HOLD_THRESHOLD if was_locked else LOCK_THRESHOLD
        for order in (2, 4):
            s0 = max(0, min(s, len(y) - block))   # last block: use a full-length window
            p, ratio = find_tone(yf[s0:s0 + block], order, nfft)
            if ratio > need:
                lock = True
                break
        f_prev = f_now
        if lock:
            f_now = freqs[p] / order
            orders.append(order)
        t = s / FS
        if lock and not was_locked:
            events.append((t, f"ACQUIRED carrier at {f_now:+.0f} Hz"))
        elif was_locked and not lock:
            events.append((t, "SIGNAL LOST, searching"))
        elif lock and abs(f_now - f_prev) > 500:
            events.append((t, f"HOP detected, re-acquired at {f_now:+.0f} Hz"))
        f_est[s:s + block] = f_now
        locked[s:s + block] = lock
        was_locked = lock
    order = max(set(orders), key=orders.count) if orders else 2
    return f_est, locked, events, order


def mixdown(y, f_est):
    return y * np.exp(-2j * np.pi * np.cumsum(f_est) / FS)   # remove carrier


def classify(base, locked, order):
    """Ask the RadioML-trained CNN what modulation this is. It looks at many
    128-sample windows of the cleaned signal and averages its votes."""
    starts = [s for s in range(0, len(base) - 128, 128) if locked[s:s + 128].all()]
    try:
        from classifier import load_model, predict
        model, mods = load_model()
    except (ImportError, FileNotFoundError) as e:
        guess = "BPSK" if order == 2 else "QPSK"
        return guess, None, f"no classifier ({type(e).__name__}), rule-based guess"
    if not starts:
        return None, None, "no locked signal to classify"
    starts = starts[:: max(1, len(starts) // 200)]
    w = np.stack([np.stack([base[s:s + 128].real, base[s:s + 128].imag]) for s in starts])
    prob = predict(model, w).mean(axis=0)
    best = int(np.argmax(prob))
    return mods[best], float(prob[best]), "CNN"


def demod(base, mod):
    mf = np.convolve(base, RRC, mode="same")                  # matched filter
    k0 = np.argmax([np.mean(np.abs(mf[k::SPS]) ** 2) for k in range(SPS)])  # timing
    sym = mf[k0::SPS]
    diff = sym[1:] * np.conj(sym[:-1])        # phase change from one symbol to the next
    if mod == "BPSK":
        return (diff.real < 0).astype(np.uint8)  # more than a quarter turn -> flipped -> 1
    q = np.round(np.angle(diff) / (np.pi / 2)).astype(int) % 4   # nearest quarter turn
    inv = {v: k for k, v in QPSK_STEP.items()}
    return np.array([b for v in q for b in inv[v]], dtype=np.uint8)


def decode(bits):
    corr = np.correlate(1 - 2.0 * bits, 1 - 2.0 * ASM_BITS, mode="valid")
    frames, last = [], -FRAME_BITS
    for h in np.where(corr >= 28)[0]:        # tolerate up to 2 bit errors in ASM
        if h - last < FRAME_BITS - 8 or h + FRAME_BITS > len(bits):
            continue
        body = np.packbits(bits[h + 32:h + FRAME_BITS]).tobytes()
        payload, crc = body[:PAYLOAD_BYTES], int.from_bytes(body[PAYLOAD_BYTES:], "big")
        ok = binascii.crc_hqx(payload, 0xFFFF) == crc
        frames.append((ok, struct.unpack(FRAME_FMT, payload)))
        last = h
    return frames


def receive(y):
    f_est, locked, events, order = track_carrier(y)
    base = mixdown(y, f_est)
    mod, conf, how = classify(base, locked, order)
    conf_txt = f" ({100 * conf:.0f}% confident)" if conf is not None else ""
    events.append((len(y) / FS, f"CLASSIFIED as {mod}{conf_txt} via {how}"))
    if mod in BITS_PER_SYMBOL:
        frames = decode(demod(base, mod))
    else:
        frames = []
        events.append((len(y) / FS, f"No demodulator for {mod}, cannot decode"))
    mf = np.convolve(base, RRC, mode="same")
    return frames, f_est, locked, events, mf, mod


def run(snr_db, mod="BPSK", n_frames=N_FRAMES, seed=0):
    rng = np.random.default_rng(seed)
    bits, truth = make_telemetry(n_frames, mod, rng)
    y, f_true, drop = channel(modulate(bits, mod), snr_db, rng)
    frames, f_est, locked, events, mf, mod_rx = receive(y)
    good = {v[0]: v for ok, v in frames if ok and v[0] < n_frames}
    return dict(y=y, mf=mf, f_true=f_true, f_est=f_est, locked=locked, events=events,
                frames=frames, good=good, truth=truth, drop=drop, n=n_frames, mod_rx=mod_rx)


# ---------------- Outputs ----------------
def save_outputs(r, snr, mod):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/telemetry_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "battery_V", "temp_C", "status"])
        for k in sorted(r["good"]):
            c, t, b, tc, st = r["good"][k]
            w.writerow([c, f"{t:.3f}", f"{b:.3f}", f"{tc:.2f}", st])
    with open(f"{OUT}/events.log", "w") as f:
        for t, msg in r["events"]:
            f.write(f"t={t:6.3f}s  {msg}\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for a, sig, title in [(ax[0], r["y"], "Before: raw received signal"),
                          (ax[1], r["mf"], "After: carrier removed + matched filter")]:
        a.specgram(sig, NFFT=1024, Fs=FS, noverlap=512, cmap="viridis")
        a.set_title(title); a.set_xlabel("Time (s)")
    ax[0].set_ylabel("Frequency (Hz)")
    fig.suptitle(f"{mod}, SNR = {snr} dB per sample"); fig.tight_layout()
    fig.savefig(f"{OUT}/spectrogram_before_after.png", dpi=150); plt.close(fig)

    t = np.arange(len(r["y"])) / FS
    fig, a = plt.subplots(figsize=(10, 4))
    a.plot(t, r["f_true"], "k", lw=2, label="True carrier")
    a.plot(t, np.where(r["locked"], r["f_est"], np.nan), "r--", lw=1.5, label="Estimated (locked)")
    a.axvspan(*r["drop"], color="gray", alpha=0.3, label="Dropout")
    a.set_xlabel("Time (s)"); a.set_ylabel("Carrier offset (Hz)")
    a.set_title("Autonomous carrier acquisition and tracking"); a.legend(); fig.tight_layout()
    fig.savefig(f"{OUT}/carrier_tracking.png", dpi=150); plt.close(fig)

    tr = np.array(r["truth"]); g = np.array([r["good"][k] for k in sorted(r["good"])])
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for a, col, lab in [(ax[0], 2, "Battery (V)"), (ax[1], 3, "Temperature (C)")]:
        a.plot(tr[:, 1], tr[:, col], "k-", alpha=0.4, label="Transmitted")
        if len(g):
            a.plot(g[:, 1], g[:, col], "o", ms=3, label="Decoded")
        a.set_ylabel(lab); a.legend()
    ax[1].set_xlabel("Time (s)"); ax[0].set_title("Decoded telemetry")
    fig.tight_layout(); fig.savefig(f"{OUT}/decoded_telemetry.png", dpi=150); plt.close(fig)


def sweep():
    snrs = np.arange(-12, 5, 2)
    fig, a = plt.subplots(figsize=(7, 4))
    for mod in ("BPSK", "QPSK"):
        rate = [np.mean([len(run(s, mod, seed=k)["good"]) / N_FRAMES for k in range(3)])
                for s in snrs]
        a.plot(snrs, 100 * np.array(rate), "o-", label=mod)
        print(f"  {mod}: " + "  ".join(f"{s:+d}dB={100 * p:.0f}%" for s, p in zip(snrs, rate)))
    a.set_xlabel("SNR (dB per sample)"); a.set_ylabel("Frames decoded (%)")
    a.set_title("Frame success rate vs SNR"); a.grid(alpha=0.3); a.legend(); fig.tight_layout()
    fig.savefig(f"{OUT}/success_vs_snr.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=0)
    ap.add_argument("--mod", choices=list(BITS_PER_SYMBOL), default="BPSK")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    r = run(a.snr, a.mod)
    save_outputs(r, a.snr, a.mod)
    bad = sum(not ok for ok, _ in r["frames"])
    print(f"Transmitted {a.mod} at {a.snr} dB (the receiver is not told this)\nEvents:")
    for t, msg in r["events"]:
        print(f"  t={t:6.3f}s  {msg}")
    print(f"\nFrames sent: {r['n']}  decoded OK: {len(r['good'])}  "
          f"CRC failures: {bad}  missing: {r['n'] - len(r['good'])}")
    if a.sweep:
        sweep()
    print(f"Outputs written to {OUT}/")
