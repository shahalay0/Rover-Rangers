import math
import numpy as np
import matplotlib.pyplot as plt
from radio_tools import SPS, send_qpsk, read_qpsk, add_fog

rng = np.random.default_rng(1)

def theory(fog_db):
    """Textbook prediction of the fraction of wrong bits for QPSK."""
    snr = 10 ** (fog_db / 10)
    return 0.5 * math.erfc(math.sqrt(snr) / math.sqrt(2))

# Part 1: pictures at three fog levels
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
for k, fog_db in enumerate([20, 10, 4]):
    bits = rng.integers(0, 2, 2000)
    signal, delay = send_qpsk(bits)
    foggy = add_fog(signal, fog_db, rng)
    samples = foggy[delay + SPS * np.arange(len(bits) // 2)]   # right moments
    wrong = np.sum(read_qpsk(samples) != bits)
    ax[k].plot(samples.real, samples.imag, ".", ms=3)
    ax[k].axhline(0, color="r", lw=1); ax[k].axvline(0, color="r", lw=1)   # the borders
    ax[k].set_aspect("equal"); ax[k].set_xlim(-2, 2); ax[k].set_ylim(-2, 2)
    ax[k].set_title(f"fog {fog_db} dB: {wrong} wrong bits of {len(bits)}")

# Part 2: count wrong bits over many fog levels, compare to the textbook
print("fog dB | wrong bits (ours) | textbook says")
for fog_db in range(0, 13, 2):
    bits = rng.integers(0, 2, 200_000)
    signal, delay = send_qpsk(bits)
    samples = add_fog(signal, fog_db, rng)[delay + SPS * np.arange(len(bits) // 2)]
    ours = np.mean(read_qpsk(samples) != bits)
    print(f"{fog_db:6d} | {ours:17.5f} | {theory(fog_db):.5f}")

plt.tight_layout()
plt.show()