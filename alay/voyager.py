import math
import numpy as np
from radio_tools import SPS, send_qpsk, read_qpsk_soft, add_fog, conv_encode, viterbi_decode

rng = np.random.default_rng(8)

# --- Mini demo: wreck 40 of the sent bits on purpose. Can the detective fix them? ---
m = rng.integers(0, 2, 1000)
sent = conv_encode(m)
soft = 1 - 2 * sent.astype(float)                  # perfectly clear: bit 0 -> +1, bit 1 -> -1
wrecked = rng.choice(len(sent), 40, replace=False)
soft[wrecked] *= -1                                # flip 40 sent bits
print(f"Mini demo: flipped 40 sent bits -> {np.sum(viterbi_decode(soft, 1000) != m)} wrong message bits\n")

# --- The fair fight: same energy per message bit ---
FAIR = 10 * math.log10(2)            # 2 sent bits per message bit -> each 2x quieter = 3 dB more fog

def textbook(fog_db):
    return 0.5 * math.erfc(math.sqrt(10 ** (fog_db / 10)) / math.sqrt(2))

print("message fog dB | no code (textbook) | Voyager code, FAIR | improvement")
for fog_db in range(0, 9):
    message = rng.integers(0, 2, 40_000)
    sent = conv_encode(message)
    signal, delay = send_qpsk(sent)
    samples = add_fog(signal, fog_db - FAIR, rng)[delay + SPS * np.arange(len(sent) // 2)]
    got = viterbi_decode(read_qpsk_soft(samples), len(message))
    ber = np.mean(got != message)
    gain = textbook(fog_db) / max(ber, 1 / len(message))
    print(f"{fog_db:14d} | {textbook(fog_db):18.5f} | {ber:18.5f} | {gain:6.0f}x fewer errors")