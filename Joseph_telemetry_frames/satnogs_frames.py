"""
Send REAL satellite telemetry frames (from db.satnogs.org) through our
simulated deep-space channel and check that the autonomous receiver gets
every byte back.

Real frames are the payload; our link adds sync word, sequence number,
length and CRC around each one, exactly like a spacecraft radio would.

Usage:  python satnogs_frames.py frames.csv
        python satnogs_frames.py frames.csv --mod QPSK --snr 4 --max-frames 200
Outputs in results/frames/: recovered_frames.csv, events.log, plots.
"""
import argparse, binascii, csv, os, re, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pipeline as P

OUT = "results/frames"
HEX = re.compile(r"^[0-9A-Fa-f]{16,}$")


def load_frames(path):
    """Read a SatNOGS DB export. Each line holds a timestamp and a frame in hex;
    we look for the long hex field so the exact column layout doesn't matter."""
    out = []
    with open(path) as f:
        for line in f:
            fields = [x.strip().strip('"') for x in re.split(r"[|,;\t]", line)]
            hexes = [x for x in fields if HEX.match(x) and len(x) % 2 == 0]
            if hexes:
                out.append((fields[0], bytes.fromhex(max(hexes, key=len))))
    return out


def build_stream(frames):
    """Short preamble, then wrap each real frame: ASM | sequence (2 B) | length (2 B) | frame | CRC16."""
    chunks = [b"\x55" * 8]          # short preamble so the receiver settles before frame 0
    for seq, (_, data) in enumerate(frames):
        header = struct.pack(">HH", seq, len(data))
        crc = binascii.crc_hqx(header + data, 0xFFFF)
        chunks.append(P.ASM.to_bytes(4, "big") + header + data + crc.to_bytes(2, "big"))
    return P.to_bits(b"".join(chunks))


def decode_stream(bits, max_len=2048):
    corr = np.correlate(1 - 2.0 * bits, 1 - 2.0 * P.ASM_BITS, mode="valid")
    found, bad = {}, 0
    for h in np.where(corr >= 28)[0]:
        hdr_end = h + 32 + 32
        if hdr_end > len(bits):
            break
        seq, n = struct.unpack(">HH", np.packbits(bits[h + 32:hdr_end]).tobytes())
        end = hdr_end + 8 * (n + 2)
        if n > max_len or end > len(bits):
            bad += 1
            continue
        body = np.packbits(bits[h + 32:end]).tobytes()      # header + data + crc
        if binascii.crc_hqx(body[:-2], 0xFFFF) == int.from_bytes(body[-2:], "big"):
            found[seq] = body[4:-2]
        else:
            bad += 1
    return found, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--mod", choices=list(P.BITS_PER_SYMBOL), default="BPSK")
    ap.add_argument("--snr", type=float, default=0)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    frames = load_frames(a.csv)[:a.max_frames]
    if not frames:
        raise SystemExit("No hex frames found in the file. Open it and check its format.")
    sizes = [len(d) for _, d in frames]
    print(f"Loaded {len(frames)} real frames ({min(sizes)} to {max(sizes)} bytes)")

    def trial(snr, seed=0):
        rng = np.random.default_rng(seed)
        y, f_true, drop = P.channel(P.modulate(build_stream(frames), a.mod), snr, rng)
        f_est, locked, events, order = P.track_carrier(y)
        base = P.mixdown(y, f_est)
        mod, conf, how = P.classify(base, locked, order)
        conf_txt = f" ({100 * conf:.0f}% confident)" if conf is not None else ""
        events.append((len(y) / P.FS, f"CLASSIFIED as {mod}{conf_txt} via {how}"))
        found, bad = decode_stream(P.demod(base, mod)) if mod in P.BITS_PER_SYMBOL else ({}, 0)
        exact = sum(found.get(i) == d for i, (_, d) in enumerate(frames))
        return dict(y=y, base=base, events=events, found=found, bad=bad, exact=exact,
                    f_true=f_true, f_est=f_est, locked=locked, drop=drop)

    r = trial(a.snr)
    print(f"Transmitted {a.mod} at {a.snr} dB (the receiver is not told this)\nEvents:")
    for t, msg in r["events"]:
        print(f"  t={t:7.3f}s  {msg}")
    print(f"\nFrames sent: {len(frames)}  recovered byte-exact: {r['exact']}  "
          f"CRC failures: {r['bad']}  missing: {len(frames) - r['exact']}")

    with open(f"{OUT}/recovered_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "original_timestamp", "status", "hex", "text"])
        for i, (ts, d) in enumerate(frames):
            got = r["found"].get(i)
            status = "OK" if got == d else ("CORRUPT" if got else "LOST")
            text = got.decode("latin1") if got and all(32 <= b < 127 for b in got) else ""
            w.writerow([i, ts, status, got.hex().upper() if got else "", text])
    with open(f"{OUT}/events.log", "w") as f:
        for t, msg in r["events"]:
            f.write(f"t={t:7.3f}s  {msg}\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    ax[0].specgram(r["y"], NFFT=1024, Fs=P.FS, noverlap=512, cmap="viridis")
    ax[0].set_title("Before: received signal carrying real frames")
    ax[1].specgram(np.convolve(r["base"], P.RRC, mode="same"), NFFT=1024, Fs=P.FS,
                   noverlap=512, cmap="viridis")
    ax[1].set_title("After: carrier removed + matched filter")
    for x in ax:
        x.set_xlabel("Time (s)")
    ax[0].set_ylabel("Frequency (Hz)"); fig.tight_layout()
    fig.savefig(f"{OUT}/spectrogram_before_after.png", dpi=150); plt.close(fig)

    ok = [r["found"].get(i) == d for i, (_, d) in enumerate(frames)]
    fig, x = plt.subplots(figsize=(10, 1.8))
    x.imshow([ok], aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    x.set_yticks([]); x.set_xlabel("Frame number")
    x.set_title("Real frames recovered byte-exact (green) or lost (red)")
    fig.tight_layout(); fig.savefig(f"{OUT}/frame_recovery.png", dpi=150); plt.close(fig)

    if a.sweep:
        snrs = np.arange(-6, 9, 2)
        rate = [np.mean([trial(s, k)["exact"] for k in range(2)]) / len(frames) for s in snrs]
        fig, x = plt.subplots(figsize=(7, 4))
        x.plot(snrs, 100 * np.array(rate), "o-")
        x.set_xlabel("SNR (dB per sample)"); x.set_ylabel("Real frames recovered (%)")
        x.set_title(f"Real SatNOGS frames through simulated channel ({a.mod})")
        x.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(f"{OUT}/recovery_vs_snr.png", dpi=150); plt.close(fig)
        print("  " + "  ".join(f"{s:+d}dB={100 * p:.0f}%" for s, p in zip(snrs, rate)))
    print(f"Outputs written to {OUT}/")


if __name__ == "__main__":
    main()
