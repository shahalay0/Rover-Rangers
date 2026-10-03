"""BPSK / QPSK / 8PSK version of our transmitter and blind receiver.
Same ideas as radio_tools.py (energy timing, M-th power Doppler search, sync-word
ambiguity), generalised from 4 corners to M points."""
import numpy as np
from radio_tools import SPS, smooth_turns


def _gray(n):
    return n ^ (n >> 1)


def _psk_table(M):
    """table[label] = symbol. Neighbouring points on the circle differ by one bit (Gray code)."""
    table = np.zeros(M, dtype=complex)
    for p in range(M):
        table[_gray(p)] = np.exp(2j * np.pi * p / M)
    return table


TABLES = {
    "BPSK": np.array([1 + 0j, -1 + 0j]),                                # 0 -> right, 1 -> left
    "QPSK": np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2),  # same corners as radio_tools
    "8PSK": _psk_table(8),
}
ORDER = {"BPSK": 2, "QPSK": 4, "8PSK": 8}        # points on the circle = power that removes the data
BITS = {"BPSK": 1, "QPSK": 2, "8PSK": 3}         # bits per symbol


def send_psk(bits, mod, sps=SPS):
    k = BITS[mod]
    bits = np.concatenate([bits, np.zeros((-len(bits)) % k, dtype=int)])   # pad to whole symbols
    labels = bits.reshape(-1, k) @ (1 << np.arange(k)[::-1])
    pulses = np.zeros(len(labels) * sps, dtype=complex)
    pulses[::sps] = TABLES[mod][labels]
    return np.convolve(pulses, smooth_turns(sps)), len(labels)


def soft_psk(samples, mod):
    """How sure we are about each bit: positive -> 0, negative -> 1 (same convention as before).
    For each bit: (distance to the nearest point where the bit is 1) - (... where it is 0)."""
    table, k = TABLES[mod], BITS[mod]
    d2 = np.abs(samples[:, None] - table[None, :]) ** 2          # distance to every point
    labels = np.arange(len(table))
    soft = np.empty((len(samples), k))
    for b in range(k):
        has_one = ((labels >> (k - 1 - b)) & 1).astype(bool)
        soft[:, b] = d2[:, has_one].min(axis=1) - d2[:, ~has_one].min(axis=1)
    return soft.ravel()


def hard_psk(samples, mod):
    return (soft_psk(samples, mod) < 0).astype(int)


def find_burst(recording, W=64):
    """Loud part above the noise floor measured from the quietest 10% of blocks (+ safety margin)."""
    n_pieces = len(recording) // W
    loud = np.mean(np.abs(recording[:n_pieces * W].reshape(n_pieces, W)) ** 2, axis=1)
    hiss = np.percentile(loud, 10)
    is_signal = loud > 2 * hiss
    first = np.argmax(is_signal) * W
    last = (n_pieces - np.argmax(is_signal[::-1])) * W
    return max(0, first - 2 * W), min(len(recording), last + 2 * W)


def receive_psk(recording, hello, n_symbols, mod):
    """Raw recording -> aligned, cleaned message symbols. Knows only the sync word and the modulation."""
    table, M, k = TABLES[mod], ORDER[mod], BITS[mod]

    first, last = find_burst(recording)
    region = recording[first:last]

    # Timing + Doppler together: for each sampling instant, raise to the M-th power (data disappears,
    # phase only so noise spikes can't dominate) and search many Doppler speeds. The right instant gives
    # the sharpest constellation, so the strongest line picks both the timing and the Doppler.
    best = (-1.0, 0, 0.0)
    for off in range(SPS):
        r_off = region[off::SPS]
        k_off = np.arange(len(r_off))
        pM = (r_off / np.maximum(np.abs(r_off), 1e-12)) ** M
        step = 1 / (64 * SPS * len(r_off))
        speeds = np.arange(-0.0005, 0.0005 + step, step)
        rot = np.exp(-1j * M * 2 * np.pi * SPS * np.outer(speeds, k_off))       # all candidates at once
        scores = np.abs(rot @ pM) / len(r_off)
        i = int(np.argmax(scores))
        if scores[i] > best[0]:
            best = (scores[i], off, speeds[i])
    _, best_off, spin = best
    spin_coarse = spin
    raw = region[best_off::SPS]
    kk = np.arange(len(raw))
    unspun = raw * np.exp(-1j * 2 * np.pi * spin * SPS * kk)
    ref = np.angle(np.mean(table ** M))
    tilt = (np.angle(np.mean((unspun / np.maximum(np.abs(unspun), 1e-12)) ** M)) - ref) / M
    straight = unspun * np.exp(-1j * tilt)

    # Refine: compare each sample with its nearest constellation point. The leftover phase error,
    # averaged over blocks of 16 symbols, should be flat; fit a line to it and remove it (twice).
    for _ in range(2):
        nearest = table[np.argmin(np.abs(straight[:, None] - table[None, :]), axis=1)]
        err = straight * np.conj(nearest)
        nb = len(err) // 16
        if nb < 3:
            break
        blocks = err[:nb * 16].reshape(nb, 16).mean(axis=1)
        centers = 16 * np.arange(nb) + 7.5
        ph = np.unwrap(np.angle(blocks))
        slope, icpt = np.polyfit(centers, ph, 1, w=np.abs(blocks))
        straight = straight * np.exp(-1j * (slope * kk + icpt))
        spin += slope / (2 * np.pi * SPS)

    # Which way is up: try M rotations x many start positions against the sync word
    n_sync = len(hello) // k                     # whole symbols covered by the sync word
    sync_bits = hello[:n_sync * k]
    best = (-1, 0, 0)
    for turn in range(M):
        turned = straight * np.exp(-2j * np.pi * turn / M)
        for shift in range(min(400, len(turned) - n_sync)):
            m = np.sum(hard_psk(turned[shift:shift + n_sync], mod) == sync_bits)
            if m > best[0]:
                best = (m, turn, shift)
    m, turn, shift = best
    corrected = straight * np.exp(-2j * np.pi * turn / M)
    return dict(found=(int(first), int(last)), photo=best_off, raw=raw, spin=float(spin), spin_coarse=float(spin_coarse), tilt=float(tilt),
                turn=int(turn), start=int(shift), sync_match=(int(m), len(sync_bits)),
                samples=corrected[shift:shift + n_symbols], all_corrected=corrected)
