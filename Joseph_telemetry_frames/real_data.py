"""
Run our receiver front end on a REAL satellite recording from SatNOGS.

SatNOGS does not publish raw IQ, but for BPSK observations the audio file
(.ogg) still contains the BPSK signal, shifted to an audio frequency.
This script turns that audio back into a complex (IQ) signal, then uses the
same carrier tracking and cleanup as pipeline.py, and shows the result.

Usage:  python real_data.py satnogs_12345678.ogg --baud 1200
Needs:  pip install soundfile scipy   (soundfile reads .ogg)
Outputs in results/real/: before/after spectrogram, carrier tracking,
constellation, and the classifier's verdict (if models/modclass.pt exists).
"""
import argparse, os
from math import gcd
import numpy as np
from scipy.signal import hilbert, resample_poly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pipeline as P

OUT = "results/real"


def load_audio(path):
    if path.lower().endswith(".wav"):
        from scipy.io import wavfile
        fs, x = wavfile.read(path)
        x = x.astype(float)
    else:
        import soundfile as sf
        x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)                       # stereo -> mono
    if fs != P.FS:                               # pipeline works at 48 kHz
        g = gcd(P.FS, fs)
        x = resample_poly(x, P.FS // g, fs // g)
    return x / (np.std(x) + 1e-12)


def coarse_center(z):
    """Find roughly where the signal sits: the strongest ~3 kHz wide band."""
    nfft = 8192
    segs = [z[s:s + nfft] for s in range(0, len(z) - nfft, nfft)]
    psd = np.mean([np.abs(np.fft.fft(s * np.hanning(nfft))) ** 2 for s in segs], axis=0)
    freqs = np.fft.fftfreq(nfft, 1 / P.FS)
    smooth = np.convolve(np.fft.fftshift(psd), np.ones(512), mode="same")
    return np.fft.fftshift(freqs)[np.argmax(smooth)]


def vote_chart(base8, lock8):
    """Per-window predictions as a bar chart (same idea as the team notebook)."""
    try:
        from classifier import load_model, predict
        model, mods = load_model()
    except (ImportError, FileNotFoundError):
        return
    starts = [s for s in range(0, len(base8) - 128, 128) if lock8[s:s + 128].all()]
    if not starts:
        return
    w = np.stack([np.stack([base8[s:s + 128].real, base8[s:s + 128].imag]) for s in starts])
    counts = np.bincount(predict(model, w).argmax(1), minlength=len(mods)) / len(w)
    fig, x = plt.subplots(figsize=(8, 3.5))
    x.bar(mods, counts, color=["C2" if m == "BPSK" else "C0" for m in mods])
    x.set_ylabel("fraction of windows"); x.tick_params(axis="x", rotation=45)
    x.set_title(f"IQNet vote over {len(w)} windows of the real recording (green = BPSK)")
    fig.tight_layout(); fig.savefig(f"{OUT}/class_votes.png", dpi=150); plt.close(fig)
    print("  vote: " + "  ".join(f"{m}={100 * c:.0f}%" for m, c in zip(mods, counts) if c >= 0.02))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="SatNOGS .ogg (or .wav) from a BPSK observation")
    ap.add_argument("--baud", type=int, required=True,
                    help="symbol rate, from the satellite's transmitter info (e.g. 1200)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # 1. audio (real numbers) -> complex IQ signal, then move it near 0 Hz
    z = hilbert(load_audio(a.audio))
    fc = coarse_center(z)
    z = z * np.exp(-2j * np.pi * fc * np.arange(len(z)) / P.FS)
    print(f"Loaded {len(z) / P.FS:.1f} s of audio, signal found near {fc:.0f} Hz")

    # 2. same autonomous tracking as pipeline.py
    # longer blocks than the simulation: SatNOGS already removes most Doppler,
    # and slow satellite baud rates need more time per block to show a clear tone
    f_est, locked, events, order = P.track_carrier(z, block=8192)
    for t, msg in events:
        print(f"  t={t:7.2f}s  {msg}")
    print(f"Locked {100 * locked.mean():.0f}% of the time")
    base = P.mixdown(z, f_est)

    # 3. resample so there are 8 samples per symbol (what the classifier and RRC expect)
    target = 8 * a.baud
    g = gcd(target, P.FS)
    base8 = resample_poly(base, target // g, P.FS // g)
    lock8 = resample_poly(locked.astype(float), target // g, P.FS // g) > 0.5

    # 4. classifier verdict (P.classify only needs windows of the cleaned signal)
    mod, conf, how = P.classify(base8, lock8, order)
    conf_txt = f" ({100 * conf:.0f}% confident)" if conf is not None else ""
    print(f"CLASSIFIED as {mod}{conf_txt} via {how}")
    vote_chart(base8, lock8)

    # 5. symbols: matched filter, timing, then remove leftover phase block by block
    mf = np.convolve(base8, P.RRC, mode="same")
    k0 = np.argmax([np.mean(np.abs(mf[k::8]) ** 2) for k in range(8)])
    sym = mf[k0::8][lock8[k0::8]]
    clean = []
    for s in range(0, len(sym), 256):
        c = sym[s:s + 256]
        phi = np.angle(np.sum(c ** order)) / order    # same power trick, for phase
        clean.append(c * np.exp(-1j * phi))
    clean = np.concatenate(clean) if clean else sym

    # 6. plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].specgram(z, NFFT=1024, Fs=P.FS, noverlap=512, cmap="viridis")
    ax[0].set_title("Before: SatNOGS recording (as IQ)")
    ax[1].specgram(mf, NFFT=256, Fs=target, noverlap=128, cmap="viridis")
    ax[1].set_title("After: carrier tracked + matched filter")
    for x in ax:
        x.set_xlabel("Time (s)"); x.set_ylabel("Frequency (Hz)")
    fig.tight_layout(); fig.savefig(f"{OUT}/spectrogram_before_after.png", dpi=150); plt.close(fig)

    t = np.arange(len(z)) / P.FS
    fig, x = plt.subplots(figsize=(10, 4))
    x.plot(t, np.where(locked, f_est + fc, np.nan), "r.", ms=2)
    x.set_xlabel("Time (s)"); x.set_ylabel("Carrier (Hz, audio)")
    x.set_title("Carrier tracked by our receiver (gaps = no lock)")
    fig.tight_layout(); fig.savefig(f"{OUT}/carrier_tracking.png", dpi=150); plt.close(fig)

    pts = clean[np.random.default_rng(0).permutation(len(clean))[:5000]]
    fig, x = plt.subplots(figsize=(5, 5))
    x.plot(pts.real, pts.imag, ".", ms=2, alpha=0.4)
    x.set_aspect("equal"); x.set_xlabel("I"); x.set_ylabel("Q"); x.grid(alpha=0.3)
    x.set_title("Recovered symbols (BPSK = two clusters)")
    fig.tight_layout(); fig.savefig(f"{OUT}/constellation.png", dpi=150); plt.close(fig)
    print(f"Plots written to {OUT}/")


if __name__ == "__main__":
    main()
