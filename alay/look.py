import pickle
import matplotlib.pyplot as plt

# 1. Load the dataset. It's a Python dictionary that was saved to disk.
with open("data/RML2016.10a_dict.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

# 2. See what's inside. Each key is a (modulation, SNR) pair.
mods = sorted({key[0] for key in data})
snrs = sorted({key[1] for key in data})
print("Modulations:", mods)
print("SNR levels:", snrs)
print("Shape of one group:", data[("QPSK", 18)].shape)

# 3. Take one QPSK recording at three noise levels and plot it two ways.
fig, axes = plt.subplots(3, 2, figsize=(10, 9))
for row, snr in enumerate([18, 0, -10]):
    x = data[("QPSK", snr)][0]      # first recording in this group
    i, q = x[0], x[1]               # row 0 is I, row 1 is Q

    # Left: I and Q as waves over time
    axes[row, 0].plot(i, label="I")
    axes[row, 0].plot(q, label="Q")
    axes[row, 0].set_title(f"QPSK at {snr} dB: I and Q over time")
    axes[row, 0].legend()

    # Right: each sample as one dot (x = I, y = Q)
    axes[row, 1].scatter(i, q, s=8)
    axes[row, 1].set_title(f"QPSK at {snr} dB: samples as dots")
    axes[row, 1].set_aspect("equal")

plt.tight_layout()
plt.show()