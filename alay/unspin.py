import numpy as np
import matplotlib.pyplot as plt
from radio_tools import SPS, send_qpsk, read_qpsk, add_fog, add_spin

rng = np.random.default_rng(3)
N = 1000
TRUE_SPIN = 0.0002

# ---------- SENDER: handshake first, then the secret message ----------
hello = np.array([0, 0, 0, 1, 1, 1, 1, 0] * 4)        # 32 agreed bits (uses all 4 corners)
secret = rng.integers(0, 2, 2 * N - len(hello))
bits = np.concatenate([hello, secret])
signal, delay = send_qpsk(bits)

# ---------- CHANNEL: spin with a random-looking start angle, plus fog ----------
received = add_fog(add_spin(signal, TRUE_SPIN, start_angle=2.0), 10, rng)
samples = received[delay + SPS * np.arange(N)]

# ---------- RECEIVER ----------
# Job 1: measure the spin (same as spin.py) and undo it
p4 = samples ** 4
smooth = np.convolve(p4, np.ones(20) / 20, mode="valid")
slope, _ = np.polyfit(np.arange(len(smooth)), np.unwrap(np.angle(smooth)), 1)
spin_per_position = slope / 4                          # radians per arrow position
k = np.arange(N)
unspun = samples * np.exp(-1j * spin_per_position * k)   # rotate each photo backwards

# Job 2: straighten the leftover tilt (corners at 45° become 180° after x4)
tilt = (np.angle(np.mean(unspun ** 4)) - np.pi) / 4
straight = unspun * np.exp(-1j * tilt)

# Job 3: try all 4 head turns, keep the one where the handshake matches
best = None
for turn in range(4):
    attempt = straight * np.exp(-1j * np.pi / 2 * turn)    # turn by 0, 90, 180, 270 degrees
    got = read_qpsk(attempt)
    matches = np.sum(got[:len(hello)] == hello)
    print(f"head turned {90 * turn:3d} degrees: handshake matches {matches}/{len(hello)}")
    if best is None or matches > best[0]:
        best = (matches, turn, attempt, got)
_, turn, fixed, got = best

# ---------- CHECK ----------
wrong = np.sum(got[len(hello):] != secret)
print(f"\nchose {90 * turn} degrees")
print(f"wrong secret bits: {wrong} out of {len(secret)} ({wrong / len(secret):.5f})")
print("textbook best at 10 dB fog: 0.00078")

# ---------- PICTURES ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
for a, data, title in zip(ax, [samples, straight, fixed],
                          ["1. received (spinning)", "2. un-spun + straightened",
                           f"3. turned {90 * turn} degrees using handshake"]):
    a.scatter(data.real, data.imag, c=k, s=4, cmap="viridis")
    a.axhline(0, color="r", lw=0.8); a.axvline(0, color="r", lw=0.8)
    a.set_aspect("equal"); a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6)
    a.set_title(title)
plt.tight_layout()
plt.show()