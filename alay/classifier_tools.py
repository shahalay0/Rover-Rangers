import numpy as np
import torch
from model import SignalNet

torch.backends.cudnn.enabled = False


class Identifier:
    """Asks the trained student which game a signal is playing."""

    def __init__(self, weights="best_model.pt", data="data/rml_prepared.npz"):
        d = np.load(data)
        self.mods = list(d["mods"])
        # How big are RadioML's arrows? Measure on clear cards, so we can resize ours to match.
        X, S = d["X"], d["S"]
        clear = X[S >= 10]
        self.target_power = float(np.mean(clear[:, 0] ** 2 + clear[:, 1] ** 2))
        self.net = SignalNet()
        self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.net.eval()

    def __call__(self, signal):
        # Cut the signal into flashcards of 128 photos, like RadioML
        n = len(signal) // 128
        cards = signal[:n * 128].reshape(n, 128)
        # Resize every card to RadioML's typical arrow size
        power = np.mean(np.abs(cards) ** 2, axis=1, keepdims=True)
        cards = cards * np.sqrt(self.target_power / power)
        x = np.stack([cards.real, cards.imag], axis=1).astype(np.float32)   # (n, 2, 128)
        with torch.no_grad():
            probs = torch.softmax(self.net(torch.from_numpy(x)), dim=1).numpy()
        votes = np.bincount(probs.argmax(axis=1), minlength=len(self.mods))
        return probs.mean(axis=0), votes     # average confidence per game, and votes per game