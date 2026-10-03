import math
import numpy as np
import matplotlib.pyplot as plt
from radio_tools import (SPS, send_qpsk, read_qpsk_soft, add_fog, add_delay, add_spin,
                         receive_qpsk, conv_encode, viterbi_decode)

rng = np.random.default_rng(11)
FOG_DB = 7                       # fog per MESSAGE bit (same battery for both links)
hello = np.array([int(b) for b in format(0x1ACFFC1D, "032b")])

# ---------------- 1. The message: text -> bits ----------------
text = ("Hello Earth! This is the deep-space probe. All systems nominal. "
        "Sending science data now. Signal is weak, but our code keeps it clean.")
message = np.unpackbits(np.frombuffer(text.encode(), dtype=np.uint8)).astype(int)
print(f"Message: {len(text)} characters = {len(message)} bits\n")

def bits_to_text(bits):
    bits = np.concatenate([bits, np.zeros(len(message), dtype=int)])[:len(message)]
    return np.packbits(bits).tobytes().decode("ascii", errors="replace")

def through_space(bits, fog_db):
    signal, _ = send_qpsk(bits)
    recording = np.concatenate([np.zeros(3000), signal, np.zeros(4000)])
    recording = add_delay(recording, 0.37)
    recording = add_spin(recording, 0.00015, 1.1)
    return add_fog(recording, fog_db, rng)

# ---------------- 2. Link WITHOUT a code ----------------
bits = np.concatenate([hello, message])
got, _ = receive_qpsk(through_space(bits, FOG_DB), hello, len(bits) // 2)
got = np.concatenate([got, np.zeros(len(bits), dtype=int)])[:len(bits)][len(hello):]
plain_errors = int(np.sum(got != message))
plain_text = bits_to_text(got)

# ---------------- 3. Link WITH the Voyager code (same battery) ----------------
coded = conv_encode(message)
bits = np.concatenate([hello, coded])
recording = through_space(bits, FOG_DB - 10 * math.log10(2))
_, info = receive_qpsk(recording, hello, len(bits) // 2)
soft = read_qpsk_soft(info["samples"])[len(hello):]
soft = np.concatenate([soft, np.zeros(len(coded))])[:len(coded)]
decoded = viterbi_decode(soft, len(message))
coded_errors = int(np.sum(decoded != message))
coded_text = bits_to_text(decoded)

print(f"WITHOUT code ({plain_errors} wrong bits):\n  {plain_text}\n")
print(f"WITH Voyager code ({coded_errors} wrong bits):\n  {coded_text}\n")
print("Receiver figured out:", {k: v for k, v in info.items() if k != "samples"})

# ---------------- 4. Pictures: the receiver's journey ----------------
first, last = info["found"]
W = 64
n_pieces = len(recording) // W
loud = np.mean(np.abs(recording[:n_pieces * W].reshape(n_pieces, W)) ** 2, axis=1)
raw_moments = recording[first:last][info["photo"]::SPS]     # right moments, before un-spinning

fig, ax = plt.subplots(2, 2, figsize=(13, 10))
ax[0, 0].plot(np.arange(n_pieces) * W, loud)
ax[0, 0].axvspan(first, last, color="g", alpha=0.2, label="message found here")
ax[0, 0].set_title("1. FIND: loudness of the recording"); ax[0, 0].set_xlabel("photo number")
ax[0, 0].legend()

region = recording[first:last]
ax[0, 1].plot(region.real, region.imag, ".", ms=1, alpha=0.3)
ax[0, 1].set_title("2. What arrived: every photo (a foggy mess)")

ax[1, 0].scatter(raw_moments.real, raw_moments.imag, c=np.arange(len(raw_moments)), s=3, cmap="viridis")
ax[1, 0].set_title("3. Right moments only: still spinning (color = time)")

s = info["samples"]
ax[1, 1].plot(s.real, s.imag, ".", ms=2, alpha=0.5)
ax[1, 1].axhline(0, color="r", lw=0.8); ax[1, 1].axvline(0, color="r", lw=0.8)
ax[1, 1].set_title(f"4. CLEANED: un-spun & turned -> decoded with {coded_errors} errors")
for a in [ax[0, 1], ax[1, 0], ax[1, 1]]:
    a.set_aspect("equal"); a.grid(alpha=0.3)

fig.suptitle(f"Deep-space link at {FOG_DB} dB: without code {plain_errors} wrong bits, "
             f"with Voyager code {coded_errors} wrong bits", fontsize=13)
plt.tight_layout()
plt.savefig("demo.png", dpi=120)
plt.show()