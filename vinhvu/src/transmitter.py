"""BPSK transmitter with root-raised-cosine pulse shaping."""
import numpy as np
from .telemetry import make_packet, build_frame


def rrc_taps(sps, beta=0.35, span=10):
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0):
            h[i] = 1 - beta + 4 * beta / np.pi
        elif np.isclose(abs(ti), 1 / (4 * beta)):
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) \
                   / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def transmit(n_frames, sps=8, beta=0.35, seed=0):
    """Returns (baseband IQ, list of payloads, bit stream)."""
    rng = np.random.default_rng(seed)
    payloads = [make_packet(k, rng) for k in range(n_frames)]
    bits = np.concatenate([build_frame(p) for p in payloads])
    symbols = 1.0 - 2.0 * bits                      # bit 0 -> +1, bit 1 -> -1
    up = np.zeros(len(symbols) * sps)
    up[::sps] = symbols
    x = np.convolve(up, rrc_taps(sps, beta), mode="same") * np.sqrt(sps)
    return x.astype(np.complex128), payloads, bits
