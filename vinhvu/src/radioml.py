"""Load RadioML datasets into numpy arrays with the same layout:
X (N, 2, L) float32 unit power, y (N,) int labels, snr (N,), MODS list, SNRS list.

  2016.10A (README Part 1): pickle, 11 classes, L = 128, fits in memory
  2018.01A (README Part 2): HDF5,   24 classes, L = 1024, 21 GB -> we read a balanced subset
"""
import os
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

DEFAULT_PATH = "data/RML2016.10a_dict.pkl"
DEFAULT_PATH_2018 = "data/GOLD_XYZ_OSC.0001_1024.hdf5"
DOWNLOAD_URL = ("https://huggingface.co/datasets/FlowVortex/RML/resolve/main/"
                "RML2016.10a_dict.pkl?download=true")


def normalize(X):
    """Scale each (2, 128) example to unit average power (like the receiver's AGC)."""
    X = X.astype(np.float32)
    return X / (np.sqrt(np.mean(np.sum(X ** 2, axis=1), axis=1))[:, None, None] + 1e-12)


def load_radioml(path=DEFAULT_PATH):
    """Returns X (N, 2, 128) float32, y (N,) int labels, snr (N,), MODS list, SNRS list."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Download it from\n{DOWNLOAD_URL}\n"
                                "and put it in the data/ folder.")
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in data})
    snrs = sorted({k[1] for k in data})
    X, y, snr = [], [], []
    for (mod, s), arr in data.items():
        X.append(arr)
        y += [mods.index(mod)] * len(arr)
        snr += [s] * len(arr)
    return normalize(np.concatenate(X)), np.array(y), np.array(snr), mods, snrs


def split(y, snr, test_size=0.2, seed=0):
    """Train/test indices, stratified by (modulation, SNR)."""
    strat = y * 1000 + (snr - snr.min())
    return train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=strat)


# Class order used by most RadioML 2018.01A code (from the dataset's classes.txt).
# The file stores labels only as one-hot columns, not names. Some users report that this
# published order does not match the file (one of the dataset's known errata), so check the
# example plots: e.g. the "BPSK" class should look like a 2-level signal at high SNR.
MODS_2018 = ["OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK", "16APSK", "32APSK",
             "64APSK", "128APSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM", "AM-SSB-WC",
             "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]


def _read_subset_2018(f, per_pair=200, snr_min=None, seed=0, verbose=True, chunk=200_000):
    """f: an open h5py.File (or anything with f["X"], f["Y"], f["Z"] arrays)."""
    rng = np.random.default_rng(seed)
    n = f["Z"].shape[0]
    t0 = time.time()
    snr = np.asarray(f["Z"][:]).reshape(-1).astype(int)
    labels = np.empty(n, dtype=np.int64)
    for k in range(0, n, chunk):                         # Y is large: read it in chunks
        labels[k:k + chunk] = np.argmax(f["Y"][k:k + chunk], axis=1)
        if verbose:
            print(f"      labels: {min(k + chunk, n):>8d}/{n} ({100*min(k + chunk, n)/n:5.1f}%)")
    if verbose:
        print(f"      read labels + SNR for {n} examples in {time.time()-t0:.1f} s")

    keep_snrs = sorted(s for s in set(snr.tolist()) if snr_min is None or s >= snr_min)
    X, y, z = [], [], []
    groups = [(c, s) for c in sorted(set(labels.tolist())) for s in keep_snrs]
    t0 = time.time()
    for g, (c, s) in enumerate(groups):
        idx = np.where((labels == c) & (snr == s))[0]
        if len(idx) == 0:
            continue
        m = min(per_pair, len(idx))
        if idx[-1] - idx[0] + 1 == len(idx):             # contiguous block: one fast slice read
            start = idx[0] + rng.integers(0, len(idx) - m + 1)
            X.append(np.asarray(f["X"][start:start + m]))
        else:                                            # scattered: sorted fancy indexing
            X.append(np.asarray(f["X"][np.sort(rng.choice(idx, m, replace=False)).tolist()]))
        y += [c] * m
        z += [s] * m
        if verbose and ((g + 1) % 50 == 0 or g + 1 == len(groups)):
            print(f"      examples: group {g+1:>3d}/{len(groups)} ({time.time()-t0:.0f} s)")
    X = np.concatenate(X).transpose(0, 2, 1)             # (N, 1024, 2) -> (N, 2, 1024)
    return normalize(X), np.array(y), np.array(z), keep_snrs


def load_radioml2018(path=DEFAULT_PATH_2018, per_pair=200, snr_min=None, seed=0, verbose=True):
    """Balanced subset of RadioML 2018.01A: `per_pair` examples for every (modulation, SNR).
    per_pair=200 -> 24 x 26 x 200 = 124,800 examples, about 1 GB in memory."""
    import h5py
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Put GOLD_XYZ_OSC.0001_1024.hdf5 in the data/ folder.")
    with h5py.File(path, "r") as f:
        X, y, snr, snrs = _read_subset_2018(f, per_pair, snr_min, seed, verbose)
    n_classes = int(y.max()) + 1
    mods = MODS_2018 if n_classes <= len(MODS_2018) else [f"class{i}" for i in range(n_classes)]
    return X, y, snr, mods, snrs


def load_dataset(dataset, path=None, **kw):
    """dataset: "2016" or "2018". Same return layout for both."""
    if str(dataset) == "2018":
        return load_radioml2018(path or DEFAULT_PATH_2018, **kw)
    return load_radioml(path or DEFAULT_PATH)
