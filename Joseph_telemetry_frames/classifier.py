"""
Loads the team's modulation classifier (IQNet, trained in
track1_part1_radioml.ipynb / track1_part2_next_steps.ipynb) so the
pipeline scripts can use it. No training happens here.

Expects the notebook's outputs/ folder next to this file:
  outputs/iqnet_doppler_aug.pt   (Doppler-robust model, preferred)
  outputs/iqnet_best.pt          (baseline model, used if the first is missing)
"""
import os
import numpy as np
import torch
import torch.nn as nn

MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
        "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]          # same sorted order as the notebook
MODEL_CANDIDATES = ["outputs/iqnet_doppler_aug.pt", "outputs/iqnet_best.pt"]


def block(cin, cout, k, pool=True):
    layers = [nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU()]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class IQNet(nn.Module):
    """Identical to the notebook, so its saved weights load."""
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            block(2, 64, 7), block(64, 64, 5), block(64, 128, 3), block(128, 128, 3, pool=False))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


_CACHE = {}


def load_model(path=None):
    """Return (model, class names) from the first model file that exists.
    The model is loaded once and reused, so sweeps don't reload it every run."""
    for p in ([path] if path else MODEL_CANDIDATES):
        if p in _CACHE:
            return _CACHE[p], MODS
        if os.path.exists(p):
            model = IQNet(len(MODS))
            model.load_state_dict(torch.load(p, map_location="cpu"))
            model.eval()
            print(f"[classifier] loaded {p}")
            _CACHE[p] = model
            return model, MODS
    raise FileNotFoundError(f"no model found, looked for {MODEL_CANDIDATES}")


def predict(model, x, bs=4096):
    """x: numpy (N, 2, 128) IQ windows. Returns class probabilities (N, 11).
    Same normalization as the notebook: unit average power per window."""
    xn = x / (np.sqrt((x ** 2).sum(axis=1).mean(axis=1))[:, None, None] + 1e-8)
    out = []
    with torch.no_grad():
        for i in range(0, len(xn), bs):
            xb = torch.from_numpy(np.ascontiguousarray(xn[i:i + bs], dtype=np.float32))
            out.append(torch.softmax(model(xb), dim=1).numpy())
    return np.concatenate(out)
