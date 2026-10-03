"""AFSK 1200 baud (Bell 202) + AX.25 decoder for real SatNOGS audio recordings.

Many amateur satellites (and the ISS APRS digipeater) send AX.25 packets as AFSK:
1200 Hz = mark, 2200 Hz = space, 1200 bit/s, NRZI coding, HDLC framing, CRC-16 (X.25).

  audio ─► auto-tune tones ─► band-pass ─► tone correlators ─► DPLL bit clock
        ─► NRZI decode ─► HDLC flags + bit unstuffing ─► CRC check ─► AX.25 parse

Several "slicers" (different mark/space balances) run in parallel, like the Dire Wolf
TNC, because FM pre-emphasis changes the tone levels from station to station.
"""
from fractions import Fraction
import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve, resample_poly, welch


# ---------------- CRC + AX.25 ----------------
def crc_x25(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc ^ 0xFFFF


def parse_ax25(frame: bytes):
    """frame without FCS -> dict(src, dest, path, control, pid, info, text)."""
    addrs, i = [], 0
    while i + 7 <= len(frame):
        call = "".join(chr(c >> 1) for c in frame[i:i + 6]).strip()
        ssid = (frame[i + 6] >> 1) & 0x0F
        addrs.append(call + (f"-{ssid}" if ssid else ""))
        last = frame[i + 6] & 1
        i += 7
        if last:
            break
    if len(addrs) < 2 or i >= len(frame):
        return None
    control = frame[i]
    pid = frame[i + 1] if i + 1 < len(frame) else None
    info = frame[i + 2:]
    printable = sum(32 <= c < 127 for c in info) >= 0.8 * max(1, len(info))
    text = info.decode("ascii", "replace") if printable else info.hex(" ")
    return dict(dest=addrs[0], src=addrs[1], path=",".join(addrs[2:]), control=control,
                pid=pid, info_len=len(info), text=text,
                summary=f"{addrs[1]}>{addrs[0]}{(',' + ','.join(addrs[2:])) if addrs[2:] else ''}: {text}")


def build_ax25(src, dest, info: bytes, path=()):
    """Build a UI frame (used to make test recordings)."""
    def addr(call, last):
        c, _, s = call.partition("-")
        b = bytes((ord(ch) << 1) for ch in c.ljust(6)[:6])
        return b + bytes([0x60 | ((int(s or 0) & 0xF) << 1) | last])
    calls = [dest, src] + list(path)
    body = b"".join(addr(c, i == len(calls) - 1) for i, c in enumerate(calls)) + b"\x03\xf0" + info
    crc = crc_x25(body)
    return body + bytes([crc & 0xFF, crc >> 8])


# ---------------- modulator (test recordings only) ----------------
def afsk_modulate(frames, fs=48000, baud=1200, mark=1200, space=2200, preamble_flags=30):
    bits = []
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    for _ in range(preamble_flags):
        bits += flag
    for fr in frames:
        ones = 0
        for byte in fr:
            for k in range(8):                    # LSB first
                b = (byte >> k) & 1
                bits.append(b)
                ones = ones + 1 if b else 0
                if ones == 5:                     # bit stuffing
                    bits.append(0)
                    ones = 0
        bits += flag * 3
    level, tones = 1, []
    for b in bits:                                # NRZI: 0 -> change tone, 1 -> keep
        if b == 0:
            level ^= 1
        tones.append(level)
    spb = fs / baud
    n = int(len(tones) * spb)
    f = np.array(tones)[(np.arange(n) / spb).astype(int)]
    freq = np.where(f == 1, mark, space)
    return np.sin(2 * np.pi * np.cumsum(freq) / fs)


# ---------------- demodulator ----------------
def estimate_tones(x, fs, mark=1200, space=2200, search=300):
    """Find the actual tone pair (handles mistuning / SSB offset): slide a 2-tone
    template over the spectrum of the loudest audio blocks.
    Returns (mark, space, found). found=False -> no clear AFSK tones, nominal values returned."""
    blk = int(0.05 * fs)
    nb = len(x) // blk
    e = np.array([np.sum(x[k * blk:(k + 1) * blk] ** 2) for k in range(nb)])
    loud = np.argsort(e)[-max(5, nb // 10):]
    seg = np.concatenate([x[k * blk:(k + 1) * blk] for k in sorted(loud)])
    f, p = welch(seg, fs, nperseg=min(len(seg), int(fs / 5)))
    band = p[(f > 300) & (f < 3500)]
    floor = np.median(band) if len(band) else np.median(p)
    best, best_off, best_prom = -np.inf, 0.0, 0.0
    for off in np.arange(-search, search + 1, 5.0):
        # power captured by two 200 Hz-wide windows on the tones, both must be present
        pm = p[(f > mark + off - 100) & (f < mark + off + 100)].mean()
        ps = p[(f > space + off - 100) & (f < space + off + 100)].mean()
        score = np.log(pm + 1e-20) + np.log(ps + 1e-20)
        if score > best:
            best, best_off, best_prom = score, off, min(pm, ps) / (floor + 1e-20)
    at_edge = abs(best_off) >= search - 1e-9
    if at_edge or best_prom < 1.3:            # tones not clearly above the noise
        return mark, space, False
    return mark + best_off, space + best_off, True


def tone_discriminator(x, fs, baud, mark, space):
    lo, hi = min(mark, space) - 500, max(mark, space) + 500
    sos = butter(4, [max(lo, 100), min(hi, fs / 2 - 100)], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    n = int(round(fs / baud))
    t = np.arange(len(y)) / fs
    box = np.ones(n) / n
    m = np.abs(fftconvolve(y * np.exp(-2j * np.pi * mark * t), box, "same"))
    s = np.abs(fftconvolve(y * np.exp(-2j * np.pi * space * t), box, "same"))
    return y, m, s


def slice_bits(d, sps, gain=0.3):
    """DPLL bit clock on the discriminator output (sps samples per bit).
    Returns tone levels at bit centres and the sample index of each."""
    dl = (np.asarray(d) > 0).tolist()
    step, phase, prev = 1.0 / sps, 0.0, dl[0]
    levels, where = [], []
    for i, cur in enumerate(dl):
        if cur != prev:                           # transition: pull phase toward bit boundary
            err = phase if phase < 0.5 else phase - 1.0
            phase -= gain * err
            prev = cur
        old = phase
        phase += step
        if old < 0.5 <= phase:
            levels.append(cur)
            where.append(i)
        if phase >= 1.0:
            phase -= 1.0
    return levels, where


def hdlc_frames(levels, where):
    """NRZI decode + HDLC deframing. Returns list of (frame_bytes_without_fcs, start_index)."""
    out, cur, ones, hist, collecting, start = [], [], 0, 0, False, 0
    prev = levels[0] if levels else 0
    for k, lev in enumerate(levels):
        b = 1 if lev == prev else 0               # NRZI
        prev = lev
        hist = ((hist << 1) | b) & 0xFF
        if hist == 0x7E:                          # flag 01111110
            bits = cur[:-7]
            if collecting and len(bits) >= 18 * 8 and len(bits) % 8 == 0:
                by = bytes(int("".join(map(str, bits[j:j + 8][::-1])), 2) for j in range(0, len(bits), 8))
                body, fcs = by[:-2], by[-2] | (by[-1] << 8)
                if crc_x25(body) == fcs:
                    out.append((body, where[start]))
            cur, ones, collecting, start = [], 0, True, k
            continue
        if (hist & 0x7F) == 0x7F:                 # 7 ones in a row: abort
            collecting, cur, ones = False, [], 0
            continue
        if collecting:
            if b:
                ones += 1
                cur.append(1)
            else:
                if ones == 5:                     # stuffed zero
                    ones = 0
                    continue
                ones = 0
                cur.append(0)
    return out


def decode_afsk(x, fs, baud=1200, mark=1200, space=2200, autotune=True,
                balances=(0.5, 0.7, 1.0, 1.4, 2.0), log=print):
    """Full decoder. Returns (frames list of dicts, diagnostics dict)."""
    x = np.asarray(x, dtype=np.float64)
    x = x / (np.max(np.abs(x)) + 1e-12)
    tones_found = None
    if autotune:
        mark, space, tones_found = estimate_tones(x, fs, mark, space)
        if tones_found:
            log(f"tones measured at mark {mark:.0f} Hz / space {space:.0f} Hz")
        else:
            log(f"WARNING: no clear AFSK tone pair above the noise, using nominal {mark:.0f}/{space:.0f} Hz. "
                "The recording may contain no packets.")
    y, m, s = tone_discriminator(x, fs, baud, mark, space)

    # work at 8 samples per bit
    ratio = Fraction(8 * baud, int(fs)).limit_denominator(1000)
    m8 = resample_poly(m, ratio.numerator, ratio.denominator)
    s8 = resample_poly(s, ratio.numerator, ratio.denominator)
    fs8 = fs * ratio.numerator / ratio.denominator
    sps = fs8 / baud
    env = m8 + s8
    active = env > 3 * np.median(env)                   # well above the noise floor: packets
    if active.sum() < 100:
        active = env > np.percentile(env, 95)
    auto = np.mean(m8[active]) / (np.mean(s8[active]) + 1e-12)   # tone "twist" from pre-emphasis
    log(f"signal present in {100*active.mean():.1f}% of the recording")
    log(f"measured tone balance (twist) {20*np.log10(auto):+.1f} dB, running {len(balances)} slicers")

    found = {}
    for g in balances:
        d = m8 - g * auto * s8
        levels, where = slice_bits(d, sps)
        fr = hdlc_frames(levels, where)
        new = 0
        for body, idx in fr:
            if body not in found:
                found[body] = idx / fs8
                new += 1
        log(f"  slicer balance x{g:<4}: {len(fr):3d} valid frames ({new} new)")

    frames = []
    for body, t in sorted(found.items(), key=lambda kv: kv[1]):
        p = parse_ax25(body) or dict(summary=body.hex(" "))
        frames.append(dict(time_s=round(t, 3), length=len(body), hex=body.hex().upper(), **p))
    diag = dict(bandpassed=y, mark=mark, space=space, fs8=fs8, tones_found=tones_found,
                disc=m8 - auto * s8, balance=auto)
    return frames, diag
