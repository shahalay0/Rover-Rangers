# ===== End-to-end link demo: send spacecraft commands/telemetry, add Doppler + noise, decode blindly =====
# Self-contained (numpy + matplotlib). If the Part 2 models `base`/`aug` are loaded, the CNN also classifies the signal.
import numpy as np, matplotlib.pyplot as plt, os
OUT_DIR = globals().get("OUT_DIR", "outputs"); os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- What to send (edit freely) ----------------
MESSAGES = ["CMD SET_MODE SAFE",
            "CMD DEPLOY_ANTENNA",
            "TLM BATT=7.4V TEMP=21C MODE=NOMINAL"]
CASES = [("BPSK", 0), ("QPSK", 5), ("QPSK", 10)]   # (modulation, SNR in dB per sample)
MAX_DOPPLER = 0.02                                # random Doppler per transmission, cycles/sample (unknown to receiver)
SPS = 8
rng_link = np.random.default_rng(2026)

# ---------------- Frame format (simplified CCSDS-style) ----------------
# [ 32-bit sync word 1ACFFC1D ][ 1 byte length ][ payload bytes ][ CRC-16-CCITT ]
SYNC = np.unpackbits(np.frombuffer(bytes.fromhex("1ACFFC1D"), np.uint8))

def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc

def build_frame(msg: str) -> np.ndarray:
    payload = msg.encode("ascii")
    body = bytes([len(payload)]) + payload
    body += crc16_ccitt(body).to_bytes(2, "big")
    return np.concatenate([SYNC, np.unpackbits(np.frombuffer(body, np.uint8))]).astype(np.uint8)

# ---------------- Modem ----------------
def rrc_taps(sps=8, span=6, beta=0.35):
    t = np.arange(-span * sps, span * sps + 1) / sps
    den = 1 - (2 * beta * t) ** 2; den[np.abs(den) < 1e-8] = 1e-8
    h = np.sinc(t) * np.cos(np.pi * beta * t) / den
    return h / np.linalg.norm(h)
H_RRC = rrc_taps(SPS)

def modulate(bits, mod):
    b = bits.astype(int)
    if mod == "BPSK": return (1 - 2 * b).astype(complex)
    if len(b) % 2: b = np.append(b, 0)
    b = b.reshape(-1, 2)
    return ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])) / np.sqrt(2)          # Gray-coded QPSK

def demodulate(sym, mod):
    if mod == "BPSK": return (sym.real < 0).astype(np.uint8)
    return np.stack([sym.real < 0, sym.imag < 0], 1).astype(np.uint8).ravel()

def transmit(bits, mod):
    sym = np.concatenate([np.zeros(10), modulate(bits, mod), np.zeros(10)])
    u = np.zeros(len(sym) * SPS, complex); u[::SPS] = sym
    tx = np.convolve(u, H_RRC)
    return tx / np.sqrt(np.mean(np.abs(tx[np.abs(tx) > 1e-3]) ** 2))

def channel(tx, snr_db, cfo):
    n = np.arange(len(tx))
    rx = tx * np.exp(1j * (rng_link.uniform(0, 2 * np.pi) + 2 * np.pi * cfo * n))
    return rx + np.sqrt(10 ** (-snr_db / 10) / 2) * (rng_link.normal(size=len(rx)) + 1j * rng_link.normal(size=len(rx)))

def est_doppler(z, M, fmax=0.03, nfft=1 << 16):
    fr = np.fft.fftfreq(nfft); keep = np.abs(fr) <= M * fmax
    S = np.abs(np.fft.fft(z ** M, nfft))[keep]
    return fr[keep][S.argmax()] / M

def receive_hypothesis(rx, mod):
    M = 2 if mod == "BPSK" else 4
    f_hat = est_doppler(rx, M)                                               # 1. blind Doppler estimate
    y = np.convolve(rx * np.exp(-2j * np.pi * f_hat * np.arange(len(rx))), H_RRC)   # 2. remove it + matched filter
    k = int(np.argmax([np.mean(np.abs(y[i::SPS]) ** 2) for i in range(SPS)]))      # 3. symbol timing
    s = y[k::SPS]; idx = np.arange(len(s))
    on = np.abs(s) > 0.5 * np.percentile(np.abs(s), 90)
    sm = np.where(on, s, 0) ** M * (-1.0 if mod == "QPSK" else 1.0)      # 4. fine frequency + phase (no unwrapping,
    nf = 1 << 16; fr = np.fft.fftfreq(nf)                                   #    so noisy symbols can't cause slips)
    keep = np.abs(fr) <= 0.05
    f_fine = fr[keep][np.abs(np.fft.fft(sm, nf))[keep].argmax()]
    ph = np.angle(np.sum(sm * np.exp(-2j * np.pi * f_fine * idx)))
    s = s * np.exp(-1j * (2 * np.pi * f_fine * idx + ph) / M)
    bps = 1 if mod == "BPSK" else 2
    best = (-1, 0, 0, None)
    for rot in range(M):                                                     # 5. phase ambiguity + frame sync
        b = demodulate(s * np.exp(2j * np.pi * rot / M), mod)
        for off in range(0, len(b) - len(SYNC) - 24 + 1, bps):
            score = int((b[off:off + 32] == SYNC).sum())
            if score > best[0]: best = (score, rot, off, b)
    score, rot, off, b = best
    return {"mod": mod, "f_hat": f_hat, "sync": score, "bits": b[off + 32:],
            "sym": (s * np.exp(2j * np.pi * rot / M))[(off + 32) // bps:]}

def parse_frame(bits):
    if len(bits) < 8: return None, False
    n = int(np.packbits(bits[:8])[0])
    need = 8 * (1 + n + 2)
    if len(bits) < need: return None, False
    body = np.packbits(bits[:need]).tobytes()
    ok = crc16_ccitt(body[:1 + n]) == int.from_bytes(body[1 + n:], "big")
    return body[1:1 + n].decode("ascii", errors="replace"), ok

# Optional: the CNN's opinion on the modulation (needs Part 2 models + helpers in memory)
def cnn_vote(rx):
    if "predict_proba" not in globals() or "base" not in globals(): return "n/a"
    clf = aug if ("aug" in globals() and aug is not base) else base
    W = 128; nw = len(rx) // W
    w = rx[: nw * W].reshape(nw, W)
    x = np.stack([w.real, w.imag], 1).astype(np.float32)
    v = np.bincount(predict_proba(clf, x).argmax(1), minlength=len(MODS))
    return f"{MODS[v.argmax()]} ({v.max()}/{nw})"

# ---------------- Run every message through every case ----------------
rows, first = [], None
for mod, snr_db in CASES:
    for msg in MESSAGES:
        frame = build_frame(msg)
        cfo = rng_link.uniform(-MAX_DOPPLER, MAX_DOPPLER)
        rx = channel(transmit(frame, mod), snr_db, cfo)
        cands = [receive_hypothesis(rx, "BPSK"), receive_hypothesis(rx, "QPSK")]
        res = max(cands, key=lambda c: c["sync"])                          # sync word decides the hypothesis
        text, crc_ok = parse_frame(res["bits"])
        payload_tx = frame[32:]
        nb = min(len(payload_tx), len(res["bits"]))
        bit_err = int((res["bits"][:nb] != payload_tx[:nb]).sum()) + (len(payload_tx) - nb)
        rows.append(dict(mod=mod, snr=snr_db, cfo=cfo, f_hat=res["f_hat"], chosen=res["mod"], sync=res["sync"],
                         bit_err=bit_err, nbits=len(payload_tx), crc=crc_ok, sent=msg, got=text, cnn=cnn_vote(rx)))
        if first is None and crc_ok: first = (mod, snr_db, msg, rx, res, text, crc_ok)   # plot first good frame

print(f"{'mod':5s}{'SNR':>5s}  {'Doppler true/est':>18s}  {'detected':>8s} {'sync':>6s} {'bit err':>9s} {'CRC':>5s}  recovered")
for r in rows:
    print(f"{r['mod']:5s}{r['snr']:4d}dB  {r['cfo']:+.4f}/{r['f_hat']:+.4f}  {r['chosen']:>8s} {r['sync']:3d}/32 "
          f"{r['bit_err']:4d}/{r['nbits']:<4d} {'OK' if r['crc'] else 'FAIL':>5s}  {r['got']!r}"
          + ("" if r["cnn"] == "n/a" else f"   CNN: {r['cnn']}"))
n_ok = sum(r["crc"] and r["got"] == r["sent"] for r in rows)
print(f"\n{n_ok}/{len(rows)} frames recovered exactly with a valid CRC")

# ---------------- Plot the receiver stages for the first correctly received frame ----------------
mod, snr_db, msg, rx, res, text, crc_ok = first
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
seg = rx[10 * SPS + 48: -(10 * SPS + 48)]
axes[0].scatter(seg.real, seg.imag, c=np.arange(len(seg)), cmap="viridis", s=3)
axes[0].set_title(f"1. Received: Doppler + noise ({snr_db} dB)")
zc = seg * np.exp(-2j * np.pi * res["f_hat"] * np.arange(len(seg)))
axes[1].scatter(zc.real, zc.imag, c=np.arange(len(zc)), cmap="viridis", s=3)
axes[1].set_title(f"2. Doppler removed (est {res['f_hat']:+.4f})")
sym = res["sym"][: 8 * (3 + len(msg)) // (1 if mod == "BPSK" else 2)]
axes[2].scatter(sym.real, sym.imag, s=8, c="C3")
axes[2].set_title("3. Matched filter + timing + phase: symbols")
for ax in axes: ax.set_aspect("equal"); ax.grid(alpha=0.3)
fig.suptitle(f"{mod} @ {snr_db} dB: sent {msg!r} -> recovered {text!r}  (CRC {'OK' if crc_ok else 'FAIL'})")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/link_demo.png", dpi=150); plt.show()