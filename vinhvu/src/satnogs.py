"""Helpers for SatNOGS observation files (network.satnogs.org)."""
import re
import shutil
import subprocess
import numpy as np


def load_audio(path, mono=True):
    """Read .wav, .ogg, .flac, .mp3. Tries soundfile first, then ffmpeg, then scipy (wav only).
    Returns (samples float, sample_rate)."""
    try:
        import soundfile as sf
        x, fs = sf.read(path, dtype="float32", always_2d=True)
        return (x.mean(1) if mono else x), fs
    except ImportError:
        pass
    if shutil.which("ffmpeg"):
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                "stream=sample_rate", "-of", "csv=p=0", path],
                               capture_output=True, text=True, check=True)
        fs = int(probe.stdout.strip().split(",")[0])
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.float32).copy(), fs
    if path.lower().endswith(".wav"):
        from scipy.io import wavfile
        fs, x = wavfile.read(path)
        x = x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(1)
        return x / (np.abs(x).max() + 1e-12), fs
    raise RuntimeError("Can't read this audio file: pip install soundfile  (or install ffmpeg)")


def load_reference_frames(path):
    """Frames SatNOGS already decoded, for comparison. Accepts the observation's data
    files or a DB CSV export: every long hex string in the file is taken as one frame."""
    text = open(path, encoding="utf-8", errors="ignore").read()
    hexes = re.findall(r"\b[0-9A-Fa-f]{36,}\b", text)
    return [h.upper() for h in hexes]


def compare_with_reference(frames, reference_hex):
    """Match our frames to SatNOGS' frames. SatNOGS frames may include the 2-byte FCS,
    so match on prefixes either way."""
    ours = [f["hex"] for f in frames]
    matched = sum(any(r.startswith(o) or o.startswith(r) for r in reference_hex) for o in ours)
    ref_found = sum(any(r.startswith(o) or o.startswith(r) for o in ours) for r in reference_hex)
    return dict(ours=len(ours), reference=len(reference_hex), ours_matched=matched,
                reference_found=ref_found, only_ours=len(ours) - matched,
                missed=len(reference_hex) - ref_found)


def make_test_pass(wav_path, ref_path=None, fs=48000, duration=60.0, n_packets=10,
                   tone_offset=30.0, seed=3):
    """Synthetic stand-in for a SatNOGS AFSK pass (used only when no real file is given):
    packets with fading levels, FM noise, pre-emphasis tilt and a small tuning offset."""
    from scipy.io import wavfile
    from scipy.signal import lfilter
    from .afsk import build_ax25, afsk_modulate
    rng = np.random.default_rng(seed)
    x = np.zeros(int(duration * fs))
    frames = []
    times = np.sort(rng.uniform(2, duration - 3, n_packets))
    for k, t0 in enumerate(times):
        info = f"T#{k:03d},{rng.integers(100,200)},{rng.integers(0,255)},{rng.integers(0,255)},000,00000000".encode()
        fr = build_ax25("DEMO-1", "CQ", info, ["SAT"])
        frames.append(fr)
        sig = afsk_modulate([fr], fs, mark=1200 + tone_offset, space=2200 + tone_offset)
        sig = lfilter([1, -0.7], [1], sig)                      # pre-emphasis tilt
        level = 0.6 * np.sin(np.pi * t0 / duration) + 0.1     # weak at horizon, strong overhead
        i = int(t0 * fs)
        x[i:i + len(sig)] += level * sig[:len(x) - i]
    noise = lfilter([1, -0.3], [1], rng.normal(0, 0.12, len(x)))  # FM discriminator noise
    x = x + noise
    x = (x / np.abs(x).max() * 0.9 * 32767).astype(np.int16)
    wavfile.write(wav_path, fs, x)
    if ref_path:   # what SatNOGS would list as decoded (here: all but one, with FCS)
        with open(ref_path, "w") as f:
            for fr in frames[1:]:
                f.write(fr.hex().upper() + "\n")
    return len(frames)


def burst_activity(x, fs, mark=1200, space=2200, block_s=0.05):
    """Tone-band energy vs noise-band energy per block, in dB. With FM audio an AFSK
    packet raises the tone band and quiets the noise band, so bursts stand out."""
    from scipy.signal import stft
    f, t, Z = stft(x, fs, nperseg=int(block_s * fs))
    P = np.abs(Z) ** 2
    tone = P[(f > min(mark, space) - 300) & (f < max(mark, space) + 300)].mean(0)
    noise = P[(f > 3000) & (f < min(6000, fs / 2 - 100))].mean(0)
    r = 10 * np.log10((tone + 1e-20) / (noise + 1e-20))
    return t, r - np.median(r)


def plot_results(x, fs, frames, diag, out_prefix="results/satnogs"):
    """Spectrogram (dB), burst activity, spectrum before/after, discriminator at a packet."""
    import matplotlib.pyplot as plt
    from scipy.signal import welch, spectrogram
    fig, ax = plt.subplots(4, 1, figsize=(13, 14))

    f, t, S = spectrogram(x, fs, nperseg=1024, noverlap=512)
    keep = f <= 5000
    Sdb = 10 * np.log10(S[keep] + 1e-20)
    ax[0].pcolormesh(t, f[keep], Sdb, shading="auto", cmap="viridis",
                     vmin=np.percentile(Sdb, 20), vmax=np.percentile(Sdb, 99.9))
    for fr in frames:
        ax[0].axvline(fr["time_s"], color="r", alpha=.7, lw=1)
    for tone in (diag["mark"], diag["space"]):
        ax[0].axhline(tone, color="w", ls=":", lw=.8)
    ax[0].set(title=f"Audio spectrogram: {len(frames)} decoded packets (red), tones used (dotted)",
              xlabel="time [s]", ylabel="Hz")

    tt, r = burst_activity(x, fs, diag["mark"], diag["space"])
    ax[1].plot(tt, r, lw=.8)
    ax[1].axhline(6, color="r", ls="--", lw=.8, label="+6 dB: likely packet burst")
    for fr in frames:
        ax[1].axvline(fr["time_s"], color="r", alpha=.4, lw=1)
    n_bursts = int(np.sum(np.diff((r > 6).astype(int)) == 1))
    ax[1].set(title=f"Burst activity (tone band vs noise band): {n_bursts} candidate bursts",
              xlabel="time [s]", ylabel="dB above median")
    ax[1].legend(loc="upper right"); ax[1].grid(alpha=.3)

    for sig, lab in [(x, "before: raw audio"), (diag["bandpassed"], "after: band-pass around tones")]:
        f_, p_ = welch(sig, fs, nperseg=4096)
        ax[2].plot(f_, 10 * np.log10(p_ / p_.max() + 1e-20), label=lab, alpha=.85)
    ax[2].axvline(diag["mark"], color="k", ls=":"); ax[2].axvline(diag["space"], color="k", ls=":")
    ax[2].set(title="Spectrum before / after", xlabel="Hz", ylabel="PSD [dB, normalized]", xlim=(0, 6000))
    ax[2].set_ylim(-80, 5); ax[2].legend(); ax[2].grid(alpha=.3)

    d, fs8 = diag["disc"], diag["fs8"]
    if frames:
        t0 = frames[0]["time_s"]
        i0, i1 = int(t0 * fs8), int((t0 + 0.12) * fs8)
        ax[3].plot(np.arange(i0, min(i1, len(d))) / fs8, d[i0:i1], lw=.9)
        ax[3].axhline(0, color="r", lw=.8)
        ax[3].set(title=f"Mark/space discriminator at the first packet ({frames[0].get('src', '')})",
                  xlabel="time [s]", ylabel="mark − space")
    else:
        ax[3].axis("off")
        ax[3].text(0.5, 0.5, "No packets decoded.\nIf the burst-activity panel shows no peaks, the recording\n"
                   "most likely contains no AFSK packets (check the SatNOGS waterfall).",
                   ha="center", va="center", fontsize=12, transform=ax[3].transAxes)
    ax[3].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_overview.png", dpi=130)
    return fig, n_bursts
