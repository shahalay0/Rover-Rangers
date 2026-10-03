import numpy as np
import torch
import matplotlib.pyplot as plt
from model import SignalNet

torch.backends.cudnn.enabled = False
device = "cuda" if torch.cuda.is_available() else "cpu"

d = np.load("data/rml_prepared.npz")
X = torch.from_numpy(d["X"]).to(device)
Y, S, test, mods = d["Y"], d["S"], d["test"], d["mods"]
sunny = test[S[test] >= 10]                 # only clear-day final exam cards

net = SignalNet().to(device)
net.load_state_dict(torch.load("best_model.pt", map_location=device))
net.eval()

guesses = []
with torch.no_grad():
    for start in range(0, len(sunny), 2048):
        batch = torch.from_numpy(sunny[start:start + 2048]).to(device)
        guesses.append(net(X[batch]).argmax(dim=1).cpu().numpy())
guesses = np.concatenate(guesses)

# Build the mix-up table: rows = true game, columns = guess
n = len(mods)
table = np.zeros((n, n))
for true_game, guess in zip(Y[sunny], guesses):
    table[true_game, guess] += 1
table = table / table.sum(axis=1, keepdims=True) * 100    # turn counts into percentages

# Print each game's score, and its biggest mix-ups
print("On sunny days (10 dB and up):\n")
for i in range(n):
    line = f"{mods[i]:7s} correct {table[i, i]:5.1f}%"
    mixups = [(table[i, j], mods[j]) for j in range(n) if j != i and table[i, j] >= 5]
    for pct, name in sorted(mixups, reverse=True):
        line += f"  | mistaken for {name} {pct:.0f}%"
    print(line)

# Draw the table as a colored grid
plt.figure(figsize=(8, 7))
plt.imshow(table, cmap="Blues", vmin=0, vmax=100)
plt.xticks(range(n), mods, rotation=90)
plt.yticks(range(n), mods)
for i in range(n):
    for j in range(n):
        if table[i, j] >= 3:
            plt.text(j, i, f"{table[i, j]:.0f}", ha="center", va="center",
                     color="white" if table[i, j] > 50 else "black", fontsize=8)
plt.xlabel("student guessed")
plt.ylabel("true game")
plt.title("Mix-up table on sunny days (numbers are %)")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_sunny.png", dpi=120)
plt.show()