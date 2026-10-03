import numpy as np
from radio_tools import send_qpsk, add_fog, add_spin, add_delay
from classifier_tools import Identifier

rng = np.random.default_rng(12)
ident = Identifier()
print(f"RadioML typical arrow power: {ident.target_power:.6f}\n")

bits = rng.integers(0, 2, 4000)
signal, delay = send_qpsk(bits)

for fog_db in [20, 10, 6, 4]:
    received = add_fog(add_spin(add_delay(signal, 0.37), 0.00015, 1.1), fog_db, rng)
    avg, votes = ident(received[delay:-delay])          # skip the quiet edges
    top = np.argsort(-avg)[:3]                          # the 3 games with the highest confidence
    ranking = "  |  ".join(f"{ident.mods[i]} {avg[i] * 100:.0f}%" for i in top)
    qpsk_votes = votes[ident.mods.index("QPSK")]
    print(f"fog {fog_db:2d} dB: {ranking}   (cards voting QPSK: {qpsk_votes}/{votes.sum()})")