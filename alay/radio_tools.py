import numpy as np

SPS = 8   # photos per arrow position
CORNERS = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2)   # 00, 01, 10, 11


def smooth_turns(sps=SPS, beta=0.35, span=6):
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.sinc(t) * np.cos(np.pi * beta * t)
    denom = 1 - (2 * beta * t) ** 2
    special = np.abs(denom) < 1e-8
    h[~special] /= denom[~special]
    h[special] = np.pi / 4 * np.sinc(1 / (2 * beta))
    return h


def send_qpsk(bits, sps=SPS):
    """Bits -> smooth arrow signal. Also returns where the first right moment is."""
    pair_number = 2 * bits[0::2] + bits[1::2]
    pulses = np.zeros(len(pair_number) * sps, dtype=complex)
    pulses[::sps] = CORNERS[pair_number]
    signal = np.convolve(pulses, smooth_turns(sps))
    delay = 6 * sps
    return signal, delay


def read_qpsk(samples):
    """Photos taken at the right moments -> bits (which corner is each one in?)."""
    got = np.empty(2 * len(samples), dtype=int)
    got[0::2] = (samples.imag < 0).astype(int)
    got[1::2] = (samples.real < 0).astype(int)
    return got


def add_fog(signal, fog_db, rng):
    """Give every photo a random shove. Higher fog_db = clearer day."""
    shove_power = 10 ** (-fog_db / 10)      # e.g. 10 dB -> shoves 10x weaker than the arrow
    shove = np.sqrt(shove_power / 2) * (rng.standard_normal(len(signal))
                                        + 1j * rng.standard_normal(len(signal)))
    return signal + shove

def add_spin(signal, spin_speed, start_angle=0.0):
    """Merry-go-round: rotate the arrow a little more on every photo.
    spin_speed = turns per photo (e.g. 0.0002 = one full turn every 5000 photos)."""
    n = np.arange(len(signal))
    return signal * np.exp(1j * (2 * np.pi * spin_speed * n + start_angle))

def add_delay(signal, photos):
    """The signal arrives late by some number of photos (can be a fraction, like 5.4)."""
    n = np.arange(len(signal))
    re = np.interp(n - photos, n, signal.real, left=0, right=0)
    im = np.interp(n - photos, n, signal.imag, left=0, right=0)
    return re + 1j * im

def receive_qpsk(recording, hello, n_symbols):
    """Raw recording in -> message bits out. Knows nothing except the handshake."""

    # Job A: find the loud part (the message is louder than the hiss)
    W = 64
    n_pieces = len(recording) // W
    loud = np.mean(np.abs(recording[:n_pieces * W].reshape(n_pieces, W)) ** 2, axis=1)
    hiss = np.percentile(loud, 10)       # quietest 10% of pieces = hiss     
    is_signal = loud > 2 * hiss
    first = np.argmax(is_signal) * W
    last = (n_pieces - np.argmax(is_signal[::-1])) * W
    first, last = max(0, first - 2 * W), min(len(recording), last + 2 * W)   # safety margin
    region = recording[first:last]

    # Job B: find the right moments (the arrow is longest there)
    lengths = [np.mean(np.abs(region[off::SPS]) ** 2) for off in range(SPS)]
    best_off = int(np.argmax(lengths))
    samples = region[best_off::SPS]

    # Job C: find the spin by trying many speeds; keep the one where the x4 dots pile up best
    p4 = samples ** 4
    k = np.arange(len(samples))
    step = 1 / (64 * SPS * len(samples))
    speeds = np.arange(-0.0005, 0.0005 + step, step)          # turns per photo
    scores = [np.abs(np.mean(p4 * np.exp(-1j * 4 * 2 * np.pi * sp * SPS * k))) for sp in speeds]
    spin = speeds[int(np.argmax(scores))]
    samples = samples * np.exp(-1j * 2 * np.pi * spin * SPS * k)   # un-spin
    tilt = (np.angle(np.mean(samples ** 4)) - np.pi) / 4
    samples = samples * np.exp(-1j * tilt)                          # straighten

    # Job D: try 4 head turns x many start positions; keep where the handshake matches best
    h = len(hello) // 2
    best = (-1, 0, 0)
    for turn in range(4):
        turned = samples * np.exp(-1j * np.pi / 2 * turn)
        for shift in range(min(400, len(turned) - h)):
            m = np.sum(read_qpsk(turned[shift:shift + h]) == hello)
            if m > best[0]:
                best = (m, turn, shift)
    m, turn, shift = best

    # Read the message
    message = samples[shift:shift + n_symbols] * np.exp(-1j * np.pi / 2 * turn)
    info = dict(found=(int(first), int(last)), photo=best_off,
                spin=float(spin), turn=90 * turn, start=shift, handshake=int(m),
                samples=message)
    return read_qpsk(message), info

def read_qpsk_soft(samples):
    """Like read_qpsk, but instead of 0/1 it says HOW SURE it is.
    Positive = leaning towards 0, negative = leaning towards 1, near 0 = unsure."""
    soft = np.empty(2 * len(samples))
    soft[0::2] = samples.imag
    soft[1::2] = samples.real
    return soft

# ---------------- Voyager's code: convolutional encoder + Viterbi reader ----------------
K = 7                                   # the window looks at the last 7 message bits
TAPS = [np.array([1, 1, 1, 1, 0, 0, 1]),  # check bit 1: which window spots to count (octal 171)
        np.array([1, 0, 1, 1, 0, 1, 1])]  # check bit 2 (octal 133)


def conv_encode(message):
    """Each message bit -> 2 check bits, each mixing the last 7 message bits."""
    padded = np.concatenate([message, np.zeros(K - 1, dtype=int)])   # 6 zeros at the end: a clean finish
    c1 = np.convolve(padded, TAPS[0])[:len(padded)] % 2              # count 1s in chosen spots: odd->1
    c2 = np.convolve(padded, TAPS[1])[:len(padded)] % 2
    sent = np.empty(2 * len(padded), dtype=int)
    sent[0::2], sent[1::2] = c1, c2
    return sent


def _viterbi_tables():
    n_states = 2 ** (K - 1)                                 # 64 possible "memories" (last 6 bits)
    ns = np.arange(n_states)
    new_bit = ns >> (K - 2)                                 # the bit that just entered
    pred = [((ns & (n_states // 2 - 1)) << 1) | p for p in (0, 1)]   # the 2 memories that lead here
    signs = []
    for p in pred:
        window = np.stack([new_bit] + [(p >> (K - 2 - j)) & 1 for j in range(K - 1)], axis=1)
        bits = np.stack([(window * t).sum(1) % 2 for t in TAPS], axis=1)   # expected check bits
        signs.append(1 - 2 * bits)                          # bit 0 -> +1, bit 1 -> -1 (like soft values)
    return new_bit, pred, signs


def viterbi_decode(soft, n_message):
    """The detective: keeps the best story for each of the 64 memories, adds up sureness."""
    new_bit, pred, signs = _viterbi_tables()
    steps = len(soft) // 2
    score = np.full(2 ** (K - 1), -1e18)
    score[0] = 0.0                                          # the encoder starts with an empty memory
    choice = np.zeros((steps, 2 ** (K - 1)), dtype=np.uint8)
    for n in range(steps):
        r = soft[2 * n:2 * n + 2]
        s0 = score[pred[0]] + signs[0] @ r                  # story arriving from predecessor 0
        s1 = score[pred[1]] + signs[1] @ r                  # story arriving from predecessor 1
        choice[n] = s1 > s0                                 # keep the better story
        score = np.maximum(s0, s1)
    state, out = 0, np.empty(steps, dtype=int)              # the clean finish -> memory 0 at the end
    for n in range(steps - 1, -1, -1):                      # walk the best story backwards
        out[n] = new_bit[state]
        state = pred[choice[n, state]][state]
    return out[:n_message]