import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
SPS = 8                 # photos per arrow position (same as the dataset)
N_SYMBOLS = 200         # how many arrow positions we send

# ---------- SENDER ----------
# 1. Secret message: random bits
bits = rng.integers(0, 2, 2 * N_SYMBOLS)

# 2. Each pair of bits -> one corner (Gray coding)
corners = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2)
#                    00       01       10       11
pair_number = 2 * bits[0::2] + bits[1::2]     # (b0, b1) -> 0, 1, 2 or 3
symbols = corners[pair_number]

# 3. Smooth the turns (a "raised cosine" shape)
def smooth_turns(sps=SPS, beta=0.35, span=6):
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.sinc(t) * np.cos(np.pi * beta * t)
    denom = 1 - (2 * beta * t) ** 2
    special = np.abs(denom) < 1e-8
    h[~special] /= denom[~special]
    h[special] = np.pi / 4 * np.sinc(1 / (2 * beta))
    return h

pulses = np.zeros(N_SYMBOLS * SPS, dtype=complex)
pulses[::SPS] = symbols                        # one "push" every 8 photos
signal = np.convolve(pulses, smooth_turns())   # smooth swings between corners
delay = 6 * SPS                                # the smoothing shifts everything by this much

# ---------- RECEIVER ----------
# 1. Look only at the right moments: one photo per corner
right_moments = delay + SPS * np.arange(N_SYMBOLS)
samples = signal[right_moments]

# 2. Which corner? Check left/right and up/down
got_b0 = (samples.imag < 0).astype(int)        # bottom half -> first bit is 1
got_b1 = (samples.real < 0).astype(int)        # left half   -> second bit is 1
got_bits = np.empty_like(bits)
got_bits[0::2], got_bits[1::2] = got_b0, got_b1

# ---------- CHECK ----------
print("sent    :", "".join(map(str, bits[:30])))
print("received:", "".join(map(str, got_bits[:30])))
print(f"wrong bits: {np.sum(bits != got_bits)} out of {len(bits)}")

# ---------- PICTURES ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].plot(signal.real, signal.imag, ".", ms=3)
ax[0].set_title("ALL photos (like the dataset)")
ax[1].plot(samples.real, samples.imag, "o", ms=5)
ax[1].set_title("Only photos at the RIGHT moments")
for a in ax[:2]:
    a.set_aspect("equal"); a.grid(alpha=0.3)
    a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5)
ax[2].plot(signal.real[delay:delay + 20 * SPS], label="left/right (I)")
ax[2].plot(right_moments[:20] - delay, samples.real[:20], "ro", label="right moments")
ax[2].set_title("First 20 positions over time")
ax[2].legend()
plt.tight_layout()
plt.show()