import math
import numpy as np
from radio_tools import send_qpsk, add_fog, add_delay, add_spin, receive_qpsk

rng = np.random.default_rng(6)
hello = np.array([int(b) for b in format(0x1ACFFC1D, "032b")])   # NASA/ESA sync marker
N = 2000

def textbook(fog_db):
    return 0.5 * math.erfc(math.sqrt(10 ** (fog_db / 10)) / math.sqrt(2))

print("fog dB | our receiver | textbook | what it figured out (first try)")
for fog_db in [2, 4, 6, 8, 10, 12]:
    wrong = total = 0
    for trial in range(5):
        # Sender
        secret = rng.integers(0, 2, 2 * N - len(hello))
        bits = np.concatenate([hello, secret])
        signal, _ = send_qpsk(bits)
        # Space: random arrival time, fractional delay, random spin, fog
        before = int(rng.integers(1000, 5000))
        recording = np.concatenate([np.zeros(before), signal, np.zeros(2000)])
        recording = add_delay(recording, rng.uniform(0, 1))
        true_spin = rng.uniform(-0.0002, 0.0002)
        recording = add_spin(recording, true_spin, rng.uniform(0, 2 * np.pi))
        recording = add_fog(recording, fog_db, rng)
        # Receiver: knows only the handshake
        got, info = receive_qpsk(recording, hello, N)
        n = min(len(got), len(bits))
        wrong += np.sum(got[len(hello):n] != bits[len(hello):n])
        total += n - len(hello)
        if trial == 0:
            example = (f"arrived ~{before}, found {info['found'][0]} | "
                       f"spin {true_spin:+.6f} vs {info['spin']:+.6f} | handshake {info['handshake']}/32")
    print(f"{fog_db:6d} | {wrong / total:12.5f} | {textbook(fog_db):8.5f} | {example}")