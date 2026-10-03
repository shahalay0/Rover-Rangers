"""Deep-space channel impairments: Doppler offset + drift, frequency hop,
phase noise, sample-clock drift, dropouts, AWGN."""
import numpy as np


def deep_space_channel(x, sps, esn0_db=8.0, f0=0.05, doppler_rate=2e-8,
                       hop_at=None, hop_df=0.0, dropouts=(), clock_ppm=50.0,
                       phase_noise_std=2e-3, lead_in=20000, tail=20000, seed=1):
    rng = np.random.default_rng(seed)
    # silence before/after the pass (receiver must find the signal itself)
    x = np.concatenate([np.zeros(lead_in), x, np.zeros(tail)]).astype(complex)
    n = len(x)

    # sample-clock drift (TX/RX oscillator mismatch)
    t = np.arange(n) * (1 + clock_ppm * 1e-6) + 0.37
    t = t[t < n - 1]
    x = np.interp(t, np.arange(n), x.real) + 1j * np.interp(t, np.arange(n), x.imag)
    n = len(x)

    # carrier: Doppler offset, linear Doppler drift, optional hop, phase noise
    k = np.arange(n)
    f = f0 + doppler_rate * k
    if hop_at is not None:
        f[hop_at:] += hop_df
    phase = 2 * np.pi * np.cumsum(f) + np.cumsum(rng.normal(0, phase_noise_std, n))
    x = x * np.exp(1j * (phase + rng.uniform(0, 2 * np.pi)))

    for start, length in dropouts:                 # signal fades / occultations
        x[start:start + length] = 0

    # AWGN at the requested Es/N0
    active = np.abs(x) > 0
    es = np.mean(np.abs(x[active]) ** 2) * sps
    n0 = es / 10 ** (esn0_db / 10)
    noise = np.sqrt(n0 / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    truth = dict(f=f, lead_in=lead_in, dropouts=list(dropouts), hop_at=hop_at)
    return x + noise, truth
