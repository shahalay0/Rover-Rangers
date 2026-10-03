import math
import numpy as np
from radio_tools import (send_qpsk, add_fog, add_delay, add_spin, receive_qpsk,
                         read_qpsk_soft, conv_encode, viterbi_decode)

rng = np.random.default_rng(10)
hello = np.array([int(b) for b in format(0x1ACFFC1D, "032b")])   # NASA/ESA sync marker
N_MESSAGE = 3000                                                   # message bits per trial
TRIALS = 6

def through_space(bits, fog_db):
    """Sender -> unknown arrival time, delay and spin -> fog. Returns the raw recording."""
    signal, _ = send_qpsk(bits)
    recording = np.concatenate([np.zeros(int(rng.integers(1000, 5000))), signal, np.zeros(4000)])
    recording = add_delay(recording, rng.uniform(0, 1))
    recording = add_spin(recording, rng.uniform(-0.0002, 0.0002), rng.uniform(0, 2 * np.pi))
    return add_fog(recording, fog_db, rng)

def textbook(fog_db):
    return 0.5 * math.erfc(math.sqrt(10 ** (fog_db / 10)) / math.sqrt(2))

print("message fog dB | no code (textbook) | our receiver, no code | our receiver + Voyager code")
for fog_db in [3, 4, 5, 6, 7, 8]:
    plain_wrong = coded_wrong = 0
    for trial in range(TRIALS):
        message = rng.integers(0, 2, N_MESSAGE)

        # A) no code: send handshake + message as is
        bits = np.concatenate([hello, message])
        got, _ = receive_qpsk(through_space(bits, fog_db), hello, len(bits) // 2)
        got = np.concatenate([got, np.zeros(len(bits), dtype=int)])[:len(bits)]   # pad if cut short
        plain_wrong += np.sum(got[len(hello):] != message)

        # B) Voyager code: same battery per message bit, so each sent bit is 2x quieter (3 dB more fog)
        coded = conv_encode(message)
        bits = np.concatenate([hello, coded])
        _, info = receive_qpsk(through_space(bits, fog_db - 10 * math.log10(2)), hello, len(bits) // 2)
        soft = read_qpsk_soft(info["samples"])[len(hello):]
        soft = np.concatenate([soft, np.zeros(len(coded))])[:len(coded)]          # pad if cut short
        coded_wrong += np.sum(viterbi_decode(soft, N_MESSAGE) != message)

    total = TRIALS * N_MESSAGE
    print(f"{fog_db:14d} | {textbook(fog_db):18.5f} | {plain_wrong / total:21.5f} | {coded_wrong / total:.5f}")