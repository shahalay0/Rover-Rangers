import math
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from radio_tools import (SPS, send_qpsk, read_qpsk_soft, add_fog, add_delay, add_spin,
                         receive_qpsk, conv_encode, viterbi_decode)
from classifier_tools import Identifier

rng = np.random.default_rng(11)
FOG_DB = 7                       # fog per MESSAGE bit
hello = np.array([int(b) for b in format(0x1ACFFC1D, "032b")])

# ================= SPACECRAFT =================
text = ("Hello Earth! This is the deep-space probe. All systems nominal. "
        "Sending science data now. Signal is weak, but our code keeps it clean.")
message = np.unpackbits(np.frombuffer(text.encode(), dtype=np.uint8)).astype(int)
coded = conv_encode(message)                                  # Voyager code
signal, _ = send_qpsk(np.concatenate([hello, coded]))

# ================= SPACE =================
recording = np.concatenate([np.zeros(3000), signal, np.zeros(4000)])
recording = add_delay(recording, 0.37)
recording = add_spin(recording, 0.00015, 1.1)
recording = add_fog(recording, FOG_DB - 10 * math.log10(2), rng)

# ================= GROUND STATION (knows nothing but the handshake) =================
# 1. FIND + CLEAN (timing, spin, which way is up)
n_symbols = (len(hello) + len(coded)) // 2
_, info = receive_qpsk(recording, hello, n_symbols)
first, last = info["found"]
print(f"1. FIND: message between photos {first} and {last}")

# 2. IDENTIFY: ask the neural network which game this is
ident = Identifier()
avg, votes = ident(recording[first:last])
game = ident.mods[int(np.argmax(votes))]
print(f"2. IDENTIFY: the network says {game} "
      f"({votes.max()} of {votes.sum()} cards, {avg.max() * 100:.0f}% confident)")

# 3. DECODE (only possible for games we have a reader for)
if game == "QPSK":
    soft = read_qpsk_soft(info["samples"])[len(hello):]
    soft = np.concatenate([soft, np.zeros(len(coded))])[:len(coded)]
    decoded = viterbi_decode(soft, len(message))
    errors = int(np.sum(decoded != message))
    got_text = np.packbits(decoded).tobytes().decode("ascii", errors="replace")
    print(f"3. DECODE: {errors} wrong bits\n\n   \"{got_text}\"")
else:
    errors, got_text = None, f"(no reader for {game} yet)"
    print(f"3. DECODE: {got_text}")

# ================= PICTURES =================
W = 64
n_pieces = len(recording) // W
loud = np.mean(np.abs(recording[:n_pieces * W].reshape(n_pieces, W)) ** 2, axis=1)
raw_moments = recording[first:last][info["photo"]::SPS]
s = info["samples"]

fig, ax = plt.subplots(2, 3, figsize=(18, 10))
ax[0, 0].plot(np.arange(n_pieces) * W, loud)
ax[0, 0].axvspan(first, last, color="g", alpha=0.2, label="message found here")
ax[0, 0].set_title("1. FIND: loudness of the recording"); ax[0, 0].set_xlabel("photo number")
ax[0, 0].legend()

order = np.argsort(-votes)[:5]
ax[0, 1].barh([ident.mods[i] for i in order][::-1], votes[order][::-1])
ax[0, 1].set_title(f"2. IDENTIFY: neural network votes -> {game}")
ax[0, 1].set_xlabel("cards (128 photos each)")

region = recording[first:last]
ax[0, 2].plot(region.real, region.imag, ".", ms=1, alpha=0.3)
ax[0, 2].set_title("What arrived: every photo (a foggy mess)")

ax[1, 0].scatter(raw_moments.real, raw_moments.imag, c=np.arange(len(raw_moments)), s=3, cmap="viridis")
ax[1, 0].set_title("3a. CLEAN: right moments, still spinning")

ax[1, 1].plot(s.real, s.imag, ".", ms=2, alpha=0.5)
ax[1, 1].axhline(0, color="r", lw=0.8); ax[1, 1].axvline(0, color="r", lw=0.8)
ax[1, 1].set_title("3b. CLEAN: un-spun and turned the right way up")
for a in [ax[0, 2], ax[1, 0], ax[1, 1]]:
    a.set_aspect("equal"); a.grid(alpha=0.3)

ax[1, 2].axis("off")
ax[1, 2].set_title(f"4. DECODE (Voyager code): {errors} wrong bits")
ax[1, 2].text(0.02, 0.95, "\n".join(textwrap.wrap(got_text, 42)), va="top", family="monospace", fontsize=11)

fig.suptitle(f"Blind deep-space receiver at {FOG_DB} dB: find -> identify -> clean -> decode", fontsize=15)
plt.tight_layout()
plt.savefig("demo_full.png", dpi=120)
plt.show()