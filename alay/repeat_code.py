import math
import numpy as np
from radio_tools import SPS, send_qpsk, read_qpsk, read_qpsk_soft, add_fog

rng = np.random.default_rng(7)
R = 3                                     # say every bit 3 times

def run(message, fog_db, use_code):
    sent = np.repeat(message, R) if use_code else message          # 1 0 -> 1 1 1 0 0 0
    signal, delay = send_qpsk(sent)
    samples = add_fog(signal, fog_db, rng)[delay + SPS * np.arange(len(sent) // 2)]
    if not use_code:
        return np.mean(read_qpsk(samples) != message), None
    hard = read_qpsk(samples).reshape(-1, R).sum(axis=1) >= 2       # majority vote
    soft = read_qpsk_soft(samples).reshape(-1, R).sum(axis=1) < 0   # add up how sure
    return np.mean(hard != message), np.mean(soft != message)

FAIR = 10 * math.log10(R)                 # 3x more bits -> each one 3x quieter = 4.77 dB more fog
print("              |  no code | 3x, UNFAIR (3x the energy) | 3x, FAIR (same energy)")
print("message fog dB|          |    vote   |  add sureness  |   vote   | add sureness")
for fog_db in range(0, 11, 2):
    message = rng.integers(0, 2, 120_000)
    base, _ = run(message, fog_db, False)
    uh, us = run(message, fog_db, True)
    fh, fs = run(message, fog_db - FAIR, True)
    print(f"{fog_db:14d}| {base:8.5f} | {uh:9.5f} | {us:14.5f} | {fh:8.5f} | {fs:8.5f}")