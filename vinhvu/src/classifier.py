"""Modulation classifiers trained on RadioML, and the wrapper the receiver agent uses.

Two models:
  - "features": expert features + gradient boosting (scikit-learn, no GPU needed)
  - "cnn":      1-D CNN on raw IQ with random-phase augmentation (PyTorch)
"""
import math
import time
from collections import Counter
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from .features import extract_features
from .radioml import normalize

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

WINDOW = 128          # RadioML 2016 example length (2018 uses 1024)


# ---------------- feature model ----------------
def train_features_model(X, y, seed=0, verbose=False):
    t0 = time.time()
    if verbose:
        print(f"   extracting features from {len(X)} training examples...")
    F = extract_features(X, verbose=verbose)
    if verbose:
        print(f"   {F.shape[1]} features ready ({time.time()-t0:.1f} s). Fitting gradient boosting...")
    t1 = time.time()
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                           early_stopping=True, random_state=seed)
    model.fit(F.values, y)
    if verbose:
        print(f"   fit done in {time.time()-t1:.1f} s, {model.n_iter_} boosting rounds "
              f"(early stopping), validation log-loss {-model.validation_score_[-1]:.3f}")
    return model


# ---------------- CNN model ----------------
if HAVE_TORCH:
    def _block(i, o, k):
        return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())

    class ModCNN(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                _block(2, 64, 7), _block(64, 64, 5), nn.MaxPool1d(2),
                _block(64, 128, 5), _block(128, 128, 3), nn.MaxPool1d(2),
                _block(128, 128, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Dropout(0.3), nn.Linear(128, n_classes))

        def forward(self, x):
            return self.net(x)

    def _device():
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _random_rotate(xb):
        """Random carrier phase: a real receiver never knows the absolute phase."""
        th = torch.rand(xb.size(0), 1, device=xb.device) * 2 * math.pi
        c, s = torch.cos(th), torch.sin(th)
        i, q = xb[:, 0], xb[:, 1]
        return torch.stack([i * c - q * s, i * s + q * c], dim=1)

    @torch.no_grad()
    def cnn_predict(model, X, batch=2048):
        model.eval()
        dev = next(model.parameters()).device
        return np.concatenate([model(torch.from_numpy(X[k:k + batch]).to(dev)).argmax(1).cpu().numpy()
                               for k in range(0, len(X), batch)])

    def train_cnn(X_fit, y_fit, X_val, y_val, n_classes, epochs=15, batch=512,
                  augment=True, save_path="models/radioml_cnn.pt", seed=0):
        torch.manual_seed(seed)
        dev = _device()
        Xt, yt = torch.from_numpy(X_fit), torch.from_numpy(y_fit).long()
        model = ModCNN(n_classes).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss()
        best = -1.0
        print(f"training CNN on {dev}")
        for ep in range(epochs):
            model.train()
            t0 = time.time()
            perm = torch.randperm(len(Xt))
            for k in range(0, len(Xt), batch):
                b = perm[k:k + batch]
                xb, yb = Xt[b].to(dev), yt[b].to(dev)
                if augment:
                    xb = _random_rotate(xb)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            sched.step()
            val_acc = float(np.mean(cnn_predict(model, X_val) == y_val))
            if val_acc > best:
                best = val_acc
                torch.save(model.state_dict(), save_path)
            print(f"  epoch {ep+1:2d}/{epochs}  loss {loss.item():.3f}  val acc {val_acc:.3f}  ({time.time()-t0:.0f} s)")
        model.load_state_dict(torch.load(save_path, map_location=dev))
        return model


# ---------------- what the receiver agent uses ----------------
class ModulationClassifier:
    """Classifies a long complex baseband signal by voting over 128-sample windows."""

    def __init__(self, kind, model, mods, window=WINDOW):
        self.kind, self.model, self.mods, self.window = kind, model, mods, window

    def save(self, path):
        if self.kind == "features":
            joblib.dump({"kind": "features", "model": self.model, "mods": self.mods, "window": self.window}, path)
        else:
            torch.save({"kind": "cnn", "state": self.model.state_dict(), "mods": self.mods, "window": self.window}, path)

    @staticmethod
    def load(path):
        if path.endswith(".joblib"):
            d = joblib.load(path)
            return ModulationClassifier("features", d["model"], d["mods"], d.get("window", WINDOW))
        if not HAVE_TORCH:
            raise ImportError("PyTorch is needed to load a CNN classifier (pip install torch)")
        d = torch.load(path, map_location="cpu")
        model = ModCNN(len(d["mods"]))
        model.load_state_dict(d["state"])
        return ModulationClassifier("cnn", model.to(_device()), d["mods"], d.get("window", WINDOW))

    def predict(self, X, verbose=False):
        """X: (N, 2, 128) normalized -> label indices."""
        if self.kind == "features":
            return self.model.predict(extract_features(X, verbose=verbose).values)
        return cnn_predict(self.model, X)

    def classify_signal(self, z, n_windows=300):
        """z: complex baseband (Doppler already removed). Returns (label, confidence, top-3 votes)."""
        L = self.window
        n_windows = min(n_windows, max(1, len(z) // L))
        starts = np.linspace(0, len(z) - L, n_windows).astype(int)
        W = np.stack([np.stack([z[s:s + L].real, z[s:s + L].imag]) for s in starts])
        votes = Counter(self.mods[p] for p in self.predict(normalize(W)))
        label, count = votes.most_common(1)[0]
        return label, count / n_windows, votes.most_common(3)
