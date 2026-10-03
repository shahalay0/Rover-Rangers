import numpy as np
import torch
from model import SignalNet

# Load our prepared flashcards
d = np.load("data/rml_prepared.npz")
X, Y, mods = d["X"], d["Y"], d["mods"]

# Build a brand-new brain with random knobs
net = SignalNet()
print("Number of knobs:", sum(p.numel() for p in net.parameters()))

# Take 8 practice cards and let the brain guess
card_ids = d["train"][:8]
cards = torch.from_numpy(X[card_ids])       # shape (8, 2, 128)
net.eval()                                  # "test mode" (turns off Dropout)
with torch.no_grad():                       # just guessing, not learning
    scores = net(cards)                     # shape (8, 11): 11 scores per card
guesses = scores.argmax(dim=1)              # the game with the highest score

for k in range(8):
    true_name = mods[Y[card_ids[k]]]
    guess_name = mods[guesses[k]]
    mark = "✅" if true_name == guess_name else "❌"
    print(f"card {k}: true = {true_name:7s} guess = {guess_name:7s} {mark}")