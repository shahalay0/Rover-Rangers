import numpy as np
import matplotlib.pyplot as plt
from radio_tools import SPS, send_qpsk, add_fog, add_spin

rng = np.random.default_rng(2)
N = 1000                       # arrow positions
TRUE_SPIN = 0.0002             # turns per photo (the receiver doesn't know this!)

# Sender -> merry-go-round -> light fog
bits = rng.integers(0, 2, 2 * N)
signal, delay = send_qpsk(bits)
received = add_fog(add_spin(signal, TRUE_SPIN, start_angle=0.5), 15, rng)
samples = received[delay + SPS * np.arange(N)]          # right moments

# ---- Measure the spin with the "times 4" trick ----
p4 = samples ** 4                                       # message disappears, spin stays (x4)
smooth = np.convolve(p4, np.ones(20) / 20, mode="valid")  # average 20 neighbors to calm the fog
angle = np.unwrap(np.angle(smooth))                     # angle over time, without jumps at 360°
slope, _ = np.polyfit(np.arange(len(angle)), angle, 1)  # straight-line fit: angle change per position
measured_spin = slope / 4 / (2 * np.pi * SPS)           # undo x4, convert to turns per photo

print(f"true spin     : {TRUE_SPIN:.6f} turns per photo")
print(f"measured spin : {measured_spin:.6f} turns per photo")

# ---- Pictures ----
t = np.arange(N)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
ax[0].scatter(samples.real, samples.imag, c=t, s=4, cmap="viridis")
ax[0].set_title("Right moments, but spinning (color = time)")
ax[1].scatter(p4.real, p4.imag, c=t, s=4, cmap="viridis")
ax[1].set_title("After 'times 4': all corners become one")
for a in ax[:2]:
    a.set_aspect("equal"); a.grid(alpha=0.3)
ax[2].plot(angle, label="measured angle (x4)")
ax[2].plot(slope * np.arange(len(angle)) + angle[0], "r--", label="straight-line fit")
ax[2].set_xlabel("arrow position number"); ax[2].set_ylabel("angle (radians)")
ax[2].set_title("The spot turns at a steady speed")
ax[2].legend()
plt.tight_layout()
plt.show()