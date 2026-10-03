"""End-to-end demo: synthetic telemetry -> deep-space channel -> autonomous receiver.
Writes results/telemetry_log.csv and plots to results/."""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from src.transmitter import transmit
from src.channel import deep_space_channel
from src.receiver import SignalAgent
from src.telemetry import FIELDS

p = argparse.ArgumentParser()
p.add_argument("--esn0", type=float, default=8.0, help="Es/N0 in dB")
p.add_argument("--frames", type=int, default=300)
p.add_argument("--sweep", action="store_true", help="frame success rate vs Es/N0")
p.add_argument("--classifier", default="auto",
               help="trained classifier file from train_classifier.py, 'auto' or 'none'")
args = p.parse_args()
os.makedirs("results", exist_ok=True)
SPS = 8

CLF = None
if args.classifier != "none":
    from src.classifier import ModulationClassifier
    candidates = [args.classifier] if args.classifier != "auto" else \
        ["models/classifier_cnn.pt", "models/classifier_cnn_2018.pt",
         "models/classifier_features.joblib", "models/classifier_features_2018.joblib"]
    for path in candidates:
        if os.path.exists(path):
            try:
                CLF = ModulationClassifier.load(path)
                print(f"using modulation classifier: {path}")
                break
            except ImportError as e:
                print(f"skipping {path}: {e}")
    if CLF is None:
        print("no trained classifier found (run train_classifier.py), continuing without it")


def scenario(esn0, n_frames, verbose=True):
    tx, payloads, _ = transmit(n_frames, sps=SPS)
    n = len(tx) + 20000
    rx, truth = deep_space_channel(
        tx, SPS, esn0_db=esn0, f0=0.05, doppler_rate=2e-8,
        hop_at=int(0.68 * n), hop_df=0.03,
        dropouts=[(int(0.35 * n), 25000)])
    agent = SignalAgent(sps=SPS, classifier=CLF, verbose=verbose)
    recs = agent.run(rx)
    good = {r["seq"] for r in recs if r["crc_ok"]}
    return rx, truth, agent, recs, len(good & set(range(n_frames))) / n_frames


if args.sweep:
    snrs = np.arange(0, 13, 1.0)
    rates = [scenario(e, 120, verbose=False)[-1] for e in snrs]
    for e, r in zip(snrs, rates):
        print(f"Es/N0 {e:4.1f} dB  frame success {100*r:5.1f}%")
    plt.figure(figsize=(6, 4))
    plt.plot(snrs, 100 * np.array(rates), "o-")
    plt.xlabel("Es/N0 [dB]"); plt.ylabel("frames decoded (CRC ok) [%]")
    plt.title("Autonomous receiver: frame success vs SNR\n(Doppler + drift + hop + dropout)")
    plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("results/sweep.png", dpi=130)
    raise SystemExit

rx, truth, agent, recs, success = scenario(args.esn0, args.frames)

# ----- telemetry log -----
with open("results/telemetry_log.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS + ["crc_ok", "segment", "rx_sample", "sync_quality"])
    w.writeheader(); w.writerows(recs)
ok = sum(r["crc_ok"] for r in recs)
print(f"\nframes found: {len(recs)}, CRC ok: {ok}, unique frames recovered: {100*success:.1f}% of {args.frames}")
print("\nfirst decoded frames:")
for r in [r for r in recs if r["crc_ok"]][:5]:
    print(f"  seq {r['seq']:4d}  t={r['time_ms']/1000:6.1f}s  T={r['temp_C']:6.2f}°C  "
          f"Vbus={r['bus_V']:6.3f}V  I={r['current_A']:5.3f}A  status=0b{r['status']:08b}")

# ----- plots -----
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
f_b, p_b = welch(rx, nperseg=2048, return_onesided=False)
s0 = truth["lead_in"]
f_a, p_a = welch(agent.diag["filtered"][0], nperseg=2048, return_onesided=False)
for fr, pp, lab in [(f_b, p_b, "before: raw received IQ"),
                    (f_a, p_a, "after: agent's Doppler removal + matched filter")]:
    o = np.argsort(fr)
    ax[0, 0].plot(fr[o], 10*np.log10(pp[o] / pp.max()), label=lab, alpha=.8)
ax[0, 0].set(title="Spectrum before / after", xlabel="frequency [cycles/sample]", ylabel="PSD [dB, normalized]")
ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

ax[0, 1].specgram(rx, NFFT=1024, noverlap=512, Fs=1, cmap="viridis")
for (cb, fb, k, fit) in agent.diag["doppler"]:
    ax[0, 1].plot(k[::200], fit[::200], "r-", lw=1.5)
ax[0, 1].set(title="Spectrogram + agent's Doppler track (red)", xlabel="sample", ylabel="cycles/sample")

n_naive = 4000
naive = rx[s0:s0 + n_naive * SPS:SPS]
ax[1, 0].plot(naive.real, naive.imag, ".", ms=2, alpha=.4, label="before (raw samples)")
sym = agent.diag["symbols"][0][500:4500]
ax[1, 0].plot(sym.real, sym.imag, ".", ms=2, alpha=.4, label="after (agent output)")
ax[1, 0].set(title="Constellation", aspect="equal", xlim=(-3, 3), ylim=(-3, 3)); ax[1, 0].legend()

ax[1, 1].plot(np.arange(len(agent.diag["block_snr"])) * agent.block,
              10*np.log10(agent.diag["block_snr"]), lw=.8)
ax[1, 1].axhline(10*np.log10(agent.snr_gate), color="r", ls="--", label="acquisition gate")
for smp, msg in agent.events:
    ax[1, 1].axvline(smp, color="k", alpha=.25)
ax[1, 1].set(title="Blind SNR + agent events (grey lines)", xlabel="sample", ylabel="dB"); ax[1, 1].legend()
plt.tight_layout(); plt.savefig("results/overview.png", dpi=130)
print("\nsaved results/telemetry_log.csv and results/overview.png")
