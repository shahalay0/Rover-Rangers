import numpy as np
import matplotlib.pyplot as plt
from radio_tools import SPS, send_qpsk, read_qpsk, add_fog, add_delay

rng = np.random.default_rng(4)
N = 1000
TRUE_DELAY = 5.4              # the receiver does NOT know this

bits = rng.integers(0, 2, 2 * N)
signal, delay = send_qpsk(bits)
received = add_fog(add_delay(signal, TRUE_DELAY), 10, rng)   # no spin today, one problem at a time

# Try all 8 starting photos. Measure how long the arrow is on average at each.
print("start photo | avg arrow length^2 | wrong bits")
results = []
for offset in range(SPS):
    s = received[delay + offset + SPS * np.arange(N)]
    loudness = np.mean(np.abs(s) ** 2)          # how long the arrow is, on average
    wrong = np.sum(read_qpsk(s) != bits)        # cheating check: we know the bits
    results.append((loudness, offset, s))
    bar = "#" * int(loudness * 30)
    print(f"{offset:11d} | {loudness:18.3f} | {wrong:5d}   {bar}")

loudness, best_offset, best = max(results, key=lambda r: r[0])   # pick the longest arrow
print(f"\nreceiver picks start photo {best_offset}  (true delay was {TRUE_DELAY})")
print(f"wrong bits: {np.sum(read_qpsk(best) != bits)} of {len(bits)}  "
      f"(textbook best at 10 dB: about {0.00078 * len(bits):.1f})")

# Pictures: a bad choice, the middle, and the receiver's choice
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
for a, off in zip(ax, [(best_offset + 4) % SPS, (best_offset + 2) % SPS, best_offset]):
    s = results[off][2]
    a.plot(s.real, s.imag, ".", ms=3)
    a.axhline(0, color="r", lw=0.8); a.axvline(0, color="r", lw=0.8)
    a.set_aspect("equal"); a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6)
    a.set_title(f"start photo {off}: {np.sum(read_qpsk(s) != bits)} wrong bits")
plt.tight_layout()
plt.show()