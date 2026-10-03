import pickle
import matplotlib.pyplot as plt

# Open the box of recordings (same as before)
with open("data/RML2016.10a_dict.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

# All 11 games
mods = ["BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM64",
        "CPFSK", "GFSK", "AM-DSB", "AM-SSB", "WBFM"]

# Make a grid of 12 small pictures (3 rows, 4 columns)
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()          # turn the grid into one simple list

for k, mod in enumerate(mods):
    x = data[(mod, 18)][0]     # first recording of this game, sunny day
    i, q = x[0], x[1]          # left-right and up-down of the arrow tip
    axes[k].scatter(i, q, s=6) # one dot per photo
    axes[k].set_title(mod)
    axes[k].set_aspect("equal")

axes[11].axis("off")           # we only have 11 games, so hide the 12th box
plt.tight_layout()
plt.show()