"""Train modulation classifiers on RadioML 2016.10A or 2018.01A.

  python train_classifier.py                                  # 2016, feature model (fast, no GPU)
  python train_classifier.py --model cnn                      # 2016, 1-D CNN (needs PyTorch)
  python train_classifier.py --dataset 2018 --per-pair 200    # 2018 subset (needs h5py)

Outputs:
  models/classifier_features.joblib, models/classifier_cnn.pt
  results/classifier_acc_vs_snr.png, results/classifier_confusion.png, results/classifier_metrics.csv
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.radioml import load_dataset, split
from src.classifier import ModulationClassifier, train_features_model, HAVE_TORCH

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["2016", "2018"], default="2016")
p.add_argument("--data", default=None, help="path to the dataset file (default: data/<standard name>)")
p.add_argument("--per-pair", type=int, default=200, help="2018 only: examples per (modulation, SNR)")
p.add_argument("--snr-min", type=int, default=None, help="2018 only: skip SNRs below this")
p.add_argument("--model", choices=["features", "cnn", "both"], default="features")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

kw = dict(per_pair=args.per_pair, snr_min=args.snr_min, seed=args.seed) if args.dataset == "2018" else {}
X, y, snr, MODS, SNRS = load_dataset(args.dataset, args.data, **kw)
TAG = "" if args.dataset == "2016" else "_2018"          # 2016 and 2018 outputs don't overwrite each other
L = X.shape[-1]
idx_tr, idx_te = split(y, snr, seed=args.seed)
print(f"RadioML {args.dataset}: loaded {len(y)} examples of {L} samples | {len(MODS)} modulations | SNR {SNRS[0]}..{SNRS[-1]} dB")
print(f"train {len(idx_tr)} | test {len(idx_te)}")

results = {}

if args.model in ("features", "both"):
    t0 = time.time()
    clf = ModulationClassifier("features", train_features_model(X[idx_tr], y[idx_tr], args.seed, verbose=True), MODS, L)
    clf.save(f"models/classifier_features{TAG}.joblib")
    results["features + GB"] = clf.predict(X[idx_te])
    print(f"feature model trained in {time.time()-t0:.0f} s -> models/classifier_features{TAG}.joblib")

if args.model in ("cnn", "both"):
    if not HAVE_TORCH:
        raise SystemExit("PyTorch not installed: pip install torch")
    from src.classifier import train_cnn
    idx_fit, idx_val = split(y[idx_tr], snr[idx_tr], test_size=0.1, seed=args.seed)
    idx_fit, idx_val = idx_tr[idx_fit], idx_tr[idx_val]
    net = train_cnn(X[idx_fit], y[idx_fit], X[idx_val], y[idx_val], len(MODS),
                    epochs=args.epochs, save_path=f"models/_cnn_best{TAG}.pt", seed=args.seed)
    clf = ModulationClassifier("cnn", net, MODS, L)
    clf.save(f"models/classifier_cnn{TAG}.pt")
    results["1-D CNN"] = clf.predict(X[idx_te])
    print(f"CNN saved -> models/classifier_cnn{TAG}.pt")

# ---------- metrics ----------
y_te, snr_te = y[idx_te], snr[idx_te]
rows = []
for name, pred in results.items():
    for s in SNRS:
        m = snr_te == s
        rows.append(dict(model=name, snr_db=s, accuracy=float(np.mean(pred[m] == y_te[m]))))
    hi = snr_te >= 10
    print(f"{name:15s} overall acc {np.mean(pred == y_te):.3f} | SNR >= 10 dB: {np.mean(pred[hi] == y_te[hi]):.3f}")
metrics = pd.DataFrame(rows)
metrics.to_csv(f"results/classifier_metrics{TAG}.csv", index=False)

plt.figure(figsize=(7, 4.5))
for name, g in metrics.groupby("model"):
    plt.plot(g.snr_db, 100 * g.accuracy, "o-", label=name)
plt.axhline(100 / len(MODS), color="gray", ls=":", label="random guess")
plt.xlabel("SNR [dB]"); plt.ylabel("test accuracy [%]"); plt.ylim(0, 100)
plt.title(f"RadioML {args.dataset} modulation classification"); plt.grid(alpha=.3); plt.legend()
plt.tight_layout(); plt.savefig(f"results/classifier_acc_vs_snr{TAG}.png", dpi=130)

sz = 7 if len(MODS) <= 12 else 11
fig, ax = plt.subplots(1, len(results), figsize=(sz * len(results), sz - 1), squeeze=False)
hi = snr_te >= 10
for a, (name, pred) in zip(ax[0], results.items()):
    cm = confusion_matrix(y_te[hi], pred[hi], labels=range(len(MODS)), normalize="true")
    a.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    a.set_xticks(range(len(MODS)), MODS, rotation=60); a.set_yticks(range(len(MODS)), MODS)
    for i in range(len(MODS)):
        for j in range(len(MODS)):
            if cm[i, j] > 0.05:
                a.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=7,
                       color="white" if cm[i, j] > 0.5 else "black")
    a.set(title=f"{name} (SNR ≥ 10 dB)", xlabel="predicted", ylabel="true")
plt.tight_layout(); plt.savefig(f"results/classifier_confusion{TAG}.png", dpi=130)
print(f"saved results/classifier_acc_vs_snr{TAG}.png, classifier_confusion{TAG}.png, classifier_metrics{TAG}.csv")
