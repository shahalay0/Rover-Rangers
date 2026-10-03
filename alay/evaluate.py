import numpy as np
import torch
import matplotlib.pyplot as plt
from model import SignalNet

torch.backends.cudnn.enabled = False    # same fix as in training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load flashcards and the saved brain
d = np.load("data/rml_prepared.npz")
X = torch.from_numpy(d["X"]).to(device)
Y, S, test = d["Y"], d["S"], d["test"]

net = SignalNet().to(device)
net.load_state_dict(torch.load("best_model.pt", map_location=device))
net.eval()

# The student takes the final exam
guesses = []
with torch.no_grad():
    for start in range(0, len(test), 2048):
        batch = torch.from_numpy(test[start:start + 2048]).to(device)
        guesses.append(net(X[batch]).argmax(dim=1).cpu().numpy())
guesses = np.concatenate(guesses)
right = guesses == Y[test]               # True where the guess was correct
print(f"Final exam score (all fog levels): {right.mean() * 100:.1f}%\n")

# Score for each fog level separately
snrs = sorted(set(S[test]))
scores = []
for snr in snrs:
    on_this_day = S[test] == snr          # only cards with this fog level
    scores.append(right[on_this_day].mean() * 100)
    bar = "#" * int(scores[-1] // 4)      # little text bar chart
    print(f"fog level {snr:4.0f} dB: {scores[-1]:5.1f}%  {bar}")

# Draw it
plt.plot(snrs, scores, "o-", label="our student")
plt.axhline(100 / 11, ls="--", color="gray", label="random guessing (9%)")
plt.xlabel("SNR in dB (left = thick fog, right = sunny)")
plt.ylabel("correct answers (%)")
plt.title("How well the student identifies the game")
plt.ylim(0, 100)
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("accuracy_vs_snr.png", dpi=120)
plt.show()