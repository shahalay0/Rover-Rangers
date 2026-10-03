import numpy as np
import matplotlib.pyplot as plt
from radio_tools import SPS, send_qpsk, read_qpsk, add_fog, add_delay

rng = np.random.default_rng(5)
N = 1000

# ---------- SENDER ----------
hello = np.array([0, 0, 0, 1, 1, 1, 1, 0] * 4)        # 32-bit handshake = 16 positions
secret = rng.integers(0, 2, 2 * N - len(hello))
bits = np.concatenate([hello, secret])
signal, _ = send_qpsk(bits)       # note: the receiver no longer gets 'delay'

# ---------- CHANNEL: hiss, then the message at an unknown moment, then hiss ----------
recording = np.concatenate([np.zeros(3000), signal, np.zeros(2000)])
recording = add_fog(add_delay(recording, 0.4), 10, rng)
# (true start of the signal: photo 3000.4. The receiver doesn't know this.)

# ---------- JOB A: find the loud part ----------
W = 64                                              # photos per piece
n_pieces = len(recording) // W
loud = np.mean(np.abs(recording[:n_pieces * W].reshape(n_pieces, W)) ** 2, axis=1)
hiss = np.median(np.sort(loud)[:n_pieces // 4])     # the quietest quarter = just hiss
is_signal = loud > 3 * hiss                         # "clearly louder than the hiss"
first = np.argmax(is_signal) * W
last = (n_pieces - np.argmax(is_signal[::-1])) * W
print(f"A. hiss level {hiss:.3f}. Loud part found from photo {first} to {last}")
region = recording[first:last]

# ---------- JOB B: find the right moments (longest arrow) ----------
lengths = [np.mean(np.abs(region[off::SPS]) ** 2) for off in range(SPS)]
best_off = int(np.argmax(lengths))
samples = region[best_off::SPS]
print(f"B. best start photo within each group of 8: {best_off}")

# ---------- JOB C: slide the handshake to find the first symbol ----------
matches = []
for shift in range(min(200, len(samples) - 16)):
    got = read_qpsk(samples[shift:shift + 16])
    matches.append(np.sum(got == hello))
start = int(np.argmax(matches))
print(f"C. handshake lines up at position {start} ({matches[start]}/32 bits match)")

# ---------- READ THE MESSAGE ----------
message = samples[start:start + N]
got = read_qpsk(message)
n = min(len(got), len(bits))
wrong = np.sum(got[len(hello):n] != bits[len(hello):n])
print(f"\nwrong secret bits: {wrong} out of {n - len(hello)}")

# ---------- PICTURES ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].plot(np.arange(n_pieces) * W, loud, label="loudness of each piece")
ax[0].axhline(3 * hiss, color="r", ls="--", label="threshold (3x hiss)")
ax[0].axvspan(first, last, color="g", alpha=0.15, label="found message")
ax[0].set_xlabel("photo number"); ax[0].set_title("A. Finding the loud part"); ax[0].legend()
ax[1].plot(matches, ".-")
ax[1].set_xlabel("handshake position tried"); ax[1].set_ylabel("bits matching (of 32)")
ax[1].set_title("C. Sliding the handshake")
plt.tight_layout()
plt.show()