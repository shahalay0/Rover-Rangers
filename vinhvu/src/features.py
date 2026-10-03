"""Expert features for modulation recognition (cumulants, spectral lines, phase stats)."""
import numpy as np
import pandas as pd


def extract_features(X, verbose=False, chunk=None):
    """X: (N, 2, L) IQ, any length L -> DataFrame of features, one row per example.
    Works in chunks to keep memory bounded; verbose=True prints progress."""
    if chunk is None:
        chunk = max(1000, 2_560_000 // X.shape[-1])       # ~20k examples of 128 samples
    if len(X) > chunk:
        parts = []
        for k in range(0, len(X), chunk):
            parts.append(extract_features(X[k:k + chunk]))
            done = min(k + chunk, len(X))
            if verbose:
                print(f"      features: {done:>7d}/{len(X)} examples ({100*done/len(X):5.1f}%)")
        return pd.concat(parts, ignore_index=True)
    z = (X[:, 0] + 1j * X[:, 1]).astype(np.complex128)
    z /= np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True)) + 1e-12
    a = np.abs(z)
    an = a / (a.mean(1, keepdims=True) + 1e-12) - 1
    F = {}
    F["amp_std"] = an.std(1)
    F["amp_kurt"] = np.mean(an ** 4, 1) / (np.mean(an ** 2, 1) ** 2 + 1e-12)

    # higher-order cumulants: constellation shape (PSK vs QAM vs PAM)
    M20, M21 = np.mean(z ** 2, 1), np.mean(a ** 2, 1)
    M40, M41, M42 = np.mean(z ** 4, 1), np.mean(z ** 3 * np.conj(z), 1), np.mean(a ** 4, 1)
    M60, M63 = np.mean(z ** 6, 1), np.mean(a ** 6, 1)
    C40 = M40 - 3 * M20 ** 2
    C41 = M41 - 3 * M20 * M21
    C42 = M42 - np.abs(M20) ** 2 - 2 * M21 ** 2
    C60 = M60 - 15 * M20 * M40 + 30 * M20 ** 3
    C63 = M63 - 9 * C42 * M21 - 6 * M21 ** 3
    for name, c in [("C20", M20), ("C40", C40), ("C41", C41), ("C42", C42), ("C60", C60), ("C63", C63)]:
        F[f"|{name}|"] = np.abs(c)
    F["C42_real"] = np.real(C42)

    # instantaneous frequency / phase: FSK and FM
    dph = np.diff(np.unwrap(np.angle(z), axis=1), axis=1)
    F["freq_std"] = dph.std(1)
    F["freq_kurt"] = np.mean((dph - dph.mean(1, keepdims=True)) ** 4, 1) / (dph.var(1) ** 2 + 1e-12)
    F["freq_mean_abs"] = np.abs(dph.mean(1))
    F["abs_phase_std"] = np.std(np.abs(np.angle(z)), 1)

    # spectral lines of z^p: AM carrier (p=1), BPSK/PAM (p=2), QPSK (p=4)
    for p in (1, 2, 4):
        S = np.abs(np.fft.fft(z ** p, axis=1))
        F[f"line_z{p}"] = S.max(1) / (S.mean(1) + 1e-12)
    P = np.abs(np.fft.fft(z, axis=1)) ** 2
    half = P.shape[1] // 2
    pos, neg = P[:, 1:half].sum(1), P[:, half + 1:].sum(1)
    F["spec_asym"] = (pos - neg) / (pos + neg + 1e-12)          # AM-SSB
    F["spec_flat"] = np.exp(np.mean(np.log(P + 1e-12), 1)) / (P.mean(1) + 1e-12)
    F["dc"] = np.abs(z.mean(1))
    return pd.DataFrame(F)
