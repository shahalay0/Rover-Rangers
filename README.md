# Rover Rangers — Deep-Space Communication & Signal Intelligence

**MATLAB in Space Hackathon · Track 1 (Advanced)**

Deep-space links are weak, noisy, and constantly drifting. Signals arrive buried in noise, distorted by multipath, and shifted by Doppler and oscillator drift. This project builds a **blind deep-space receiver** that **finds, identifies, cleans and decodes** radio signals under those conditions with **no human tuning**, using the RadioML benchmark as a stand-in for real deep-space transmissions.

![Full pipeline: find, identify, clean, decode](figures/demo_full_1.png)

*At 7 dB the cleaned constellation still looks like a foggy cloud, yet the message is decoded with 0 bit errors.*

---

## Table of Contents

1. [Team](#team)
2. [Highlights](#highlights)
3. [Problem Statement](#problem-statement)
4. [Approach](#approach)
5. [Repository Structure](#repository-structure)
6. [Getting Started](#getting-started)
7. [Datasets](#datasets)
8. [Usage](#usage)
9. [Results](#results)
10. [Live Demo](#live-demo)
11. [Real Satellite Data: A Reality Check](#real-satellite-data-a-reality-check)
12. [Challenges & Lessons Learned](#challenges--lessons-learned)
13. [Future Work](#future-work)
14. [Citations & License](#citations--license)

---

## Team

**Rover Rangers** — a team of roboticists applying estimation, filtering, and learning techniques to space communications.

---

## Highlights

| | Result |
|---|---|
| **Identify** | 1D CNN, 11 modulations: **54.1%** overall test accuracy, **~79%** at SNR ≥ 0 dB, **90–100%** on 9 of 11 classes at high SNR |
| **Generalize** | Recognizes our own transmitter, never seen in training: **93 / 125** windows correct at 4 dB |
| **Clean** | Fully blind receiver stays within **0.3–0.6 dB** of the theoretical bit-error rate |
| **Decode** | Voyager K = 7 convolutional code + soft Viterbi: **0 errors in 18,000 bits** where the uncoded link loses **1.4%** |
| **Live demo** | Type any message, pick BPSK / QPSK / 8PSK, watch it travel and get decoded: **0 failures in 90** stress-test transmissions |

---

## Problem Statement

A ground station listening to a distant spacecraft faces four core problems:

- **Low SNR:** the signal is often at or below the noise floor.
- **Channel impairments:** carrier frequency offset, Doppler drift, unknown arrival time and sampling phase distort the waveform.
- **Unknown signal type:** the receiver may not know in advance which modulation scheme is being used.
- **Bit errors:** even after synchronization, noise flips bits.

**Our goal:** build a receiver that takes a raw recording and returns the transmitted message, knowing only the 32-bit CCSDS sync marker `1ACFFC1D` that real spacecraft frames begin with. Every other parameter (noise floor, timing, frequency offset, phase, modulation) is estimated from the received samples, and every stage is measured against theory.

---

## Approach

Our pipeline has four stages:

```
 Raw IQ samples ──► 1. Find ──► 2. Identify ──► 3. Clean ──► 4. Decode ──► Message
                    (energy       (1D CNN +       (timing,       (Voyager code,
                     detection,    sync-word       Doppler,       soft Viterbi)
                     frame sync)   check)          phase)
```

| Stage | Method | Why it needs no tuning |
|---|---|---|
| **1. Find** | Split the recording into 64-sample blocks and measure energy. The noise floor is the quietest 10% of blocks; the burst is everything clearly above it, plus a safety margin. The frame start is found by sliding the sync marker over the symbols. | The noise floor is measured from the recording itself. |
| **2. Identify** | 1D CNN trained on RadioML. Each 128-sample window is power-normalized to the training data's scale; the decision is a majority vote over windows, then **confirmed by the sync word**. | Votes over many windows; the sync word catches wrong guesses. |
| **3a. Timing** | Pick the sampling instant where the symbols are sharpest (energy peak; for BPSK/QPSK/8PSK, jointly with the Doppler search). | Chosen from the data. |
| **3b. Doppler** | M-th-power method (Viterbi & Viterbi): raising PSK samples to the power M removes the data, leaving only the rotation. Search many candidate offsets and keep the one where the M-th-power samples add up most coherently; a decision-directed pass removes residual drift. | Grid search, no loop gains to tune. |
| **3c. Phase ambiguity** | After removing the rotation, the constellation still looks the same when turned by 360°/M. Try every rotation and keep the one matching the sync marker. | Standard space-link practice. |
| **4. Decode** | Rate-1/2, K = 7 convolutional code (generators 171/133 octal, the Voyager / CCSDS code) with a **soft-decision Viterbi decoder** (64 states). | Soft decisions use the received sample values directly. |

### Classifier details
- **Model:** four 1D convolution layers (32, 64, 128, 128 filters; batch norm + ReLU; two max-pools), global average pooling, dense 64 with dropout, softmax. About 148k parameters.
- **Input:** 2 × 128 IQ matrix.
- **Output:** one of 11 modulation classes (8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM).
- **Split:** 70 / 15 / 15 train / validation / test, shuffled with a fixed seed.
- **Training:** 20 epochs, Adam (learning rate 10⁻³), batch 256, about 86 s per epoch on a GTX 1650.

### Test bench
RadioML has **no ground-truth bits**, so it can't score decoding. We built our own transmitter (Gray-coded BPSK / QPSK / 8PSK, raised-cosine pulses with roll-off 0.35, 8 samples per symbol) and a channel model (noise, fractional delay, Doppler offset, random phase, unknown arrival time). Known bits give exact bit-error rates.

---

## Repository Structure

```
Rover-Rangers/
├── README.md                     ← you are here
├── data/                         ← datasets go here (NOT committed, see Datasets)
│
├── radio_tools.py                ← QPSK transmitter, channel models, blind receiver, Voyager code + Viterbi
├── psk_tools.py                  ← BPSK / QPSK / 8PSK transmitter + receiver (joint timing + Doppler search)
├── model.py                      ← the 1D CNN
├── classifier_tools.py           ← applies the trained CNN to any signal (power normalization + voting)
│
├── prepare_data.py               ← RadioML → shuffled train / validation / test splits
├── train.py                      ← trains the CNN, saves best_model.pt
├── evaluate.py                   ← accuracy per SNR on the test set
├── confusion.py                  ← confusion matrix at high SNR
├── identify_test.py              ← tests the CNN on our own transmitter
├── test_brain.py                 ← sanity check of the untrained network
├── best_model.pt                 ← trained classifier weights
│
├── make_signal.py                ← transmitter test bench
├── fog.py                        ← bit-error rate in noise vs theory
├── timing.py                     ← symbol timing recovery
├── spin.py                       ← Doppler estimation (4th-power method)
├── unspin.py                     ← carrier recovery + phase ambiguity
├── find_start.py                 ← burst detection + frame sync
├── full_test.py                  ← full blind receiver vs theory
│
├── repeat_code.py                ← repetition code: fair vs unfair comparison
├── voyager.py                    ← Voyager convolutional code + Viterbi
├── deepspace.py                  ← full receiver, coded vs uncoded
│
├── demo.py                       ← coded vs uncoded text message
├── demo_full.py                  ← full pipeline figure: find → identify → clean → decode
├── live_demo.py                  ← interactive animated demo
├── look.py, compare.py           ← dataset exploration plots
├── save_all_figures.py           ← reruns every experiment, saves all figures + tables to figures/
├── diag.py, diag2.py             ← GPU timing diagnostics (found the cuDNN slowdown)
│
├── figures/                      ← every figure and printed result table
└── Blind_Deep_Space_Receiver.pptx ← presentation
```

---

## Getting Started

### Requirements

**Python 3.11** with numpy, matplotlib and PyTorch. A GPU is optional (training takes about 20 minutes on a GTX 1650; the receiver and decoder run on any CPU).

### Installation (Windows PowerShell)

```powershell
git clone https://github.com/shahalay0/Rover-Rangers.git; cd Rover-Rangers; python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install numpy matplotlib; python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

This clones the repo, creates a private Python environment, activates it, and installs the libraries (PyTorch built for NVIDIA GPUs with CUDA 12.6). For a CPU-only machine, use `python -m pip install torch` instead of the last command.

Then download the dataset into `data/` (see below).

---

## Datasets

> ⚠️ **The datasets are not stored in this repository.** They exceed GitHub's 100 MB per-file limit. Download them from the links below and place the files in `data/`.

| Dataset | File | Size | Used for |
|---|---|---|---|
| RadioML 2016.10A | `RML2016.10a_dict.pkl` | 641 MB | Classifier training and evaluation |
| RadioML 2018.01A *(optional)* | `GOLD_XYZ_OSC.0001_1024.hdf5` | 21.4 GB | Not used yet (future work) |
| SatNOGS frames for QB50P2 | `.csv` export | 100 KB | Reality check (see below) |

**RadioML 2016.10A:** 11 modulation types, 20 SNR levels (−20 dB to +18 dB in 2 dB steps), 1,000 examples per (modulation, SNR) pair, 220,000 examples in total. Each example is 128 complex samples stored as 2 rows (I and Q). It is synthetic, generated with GNU Radio, and includes white noise, multipath fading, frequency offset and sample-rate drift.

**RadioML 2018.01A:** 24 modulation types, 26 SNR levels (−20 dB to +30 dB), about 2.5 million examples of 1,024 complex samples each.

**Download:**

```powershell
mkdir data; curl.exe -L -o data/RML2016.10a_dict.pkl "https://huggingface.co/datasets/FlowVortex/RML/resolve/main/RML2016.10a_dict.pkl?download=true"
```

### Using the data from MATLAB (optional)

Our pipeline is written in Python, but the dataset can also be loaded in MATLAB. The 2016 file is a Python pickle, which MATLAB can't read directly; this one-time conversion writes a `.mat` file:

```python
import pickle
import numpy as np
import scipy.io as sio

with open("data/RML2016.10a_dict.pkl", "rb") as f:
    d = pickle.load(f, encoding="latin1")

mods = sorted({k[0] for k in d})
snrs = sorted({k[1] for k in d})
X, mod_idx, snr = [], [], []
for m in mods:
    for s in snrs:
        x = d[(m, s)]                                  # (1000, 2, 128)
        X.append(x)
        mod_idx += [mods.index(m) + 1] * len(x)        # 1-based for MATLAB
        snr += [s] * len(x)

sio.savemat("data/rml2016a.mat",
            {"X": np.vstack(X).astype(np.float32), "mod_idx": np.array(mod_idx),
             "snr": np.array(snr), "mods": np.array(mods, dtype=object)},
            do_compression=True)
```

```matlab
S = load("data/rml2016a.mat");
k = 1;
x = squeeze(S.X(k,1,:)) + 1i*squeeze(S.X(k,2,:));   % complex signal, 128x1
fprintf("Modulation: %s, SNR: %d dB\n", strtrim(S.mods{S.mod_idx(k)}), S.snr(k));
```

---

## Usage

| Task | Command |
|---|---|
| Prepare the dataset splits | `python prepare_data.py` |
| Train the classifier | `python train.py` |
| Evaluate on the test set | `python evaluate.py; python confusion.py` |
| Blind receiver vs theory | `python full_test.py` |
| Coded vs uncoded, full receiver | `python deepspace.py` |
| Full pipeline figure | `python demo_full.py` |
| **Interactive live demo** | `python live_demo.py` |
| Reproduce every figure and table | `python save_all_figures.py` |

All experiments use fixed random seeds, so every number in this README reproduces exactly. `save_all_figures.py` writes all figures and printed tables into `figures/`.

---

## Results

### 1. Classification accuracy (held-out test set)

| SNR range | Accuracy |
|---|---|
| High SNR (≥ 10 dB) | 79.9% |
| Medium SNR (0 to 8 dB) | 78.4% |
| Low SNR (< 0 dB) | 29.5% |
| **Overall** | **54.1%** |

Random guessing would be 9% (11 classes). Per SNR level:

| SNR (dB) | ≤ −12 | −10 | −8 | −6 | −4 | −2 | 0 | 2 to 18 |
|---|---|---|---|---|---|---|---|---|
| Accuracy | 8–15% | 22% | 38% | 50% | 59% | 71% | 76% | 78–81% |

![Accuracy vs SNR](figures/evaluate_1.png)

### 2. Confusion matrix (SNR ≥ 10 dB)

![Confusion matrix](figures/confusion_1.png)

Above +2 dB accuracy is flat at about 80%, so the remaining errors are not caused by noise. Nine of the 11 classes score **90–100%**; almost all errors come from two pairs:

- **QAM16 → QAM64 (88% confused):** a 128-sample window holds only about 16 symbols, too few to tell 16 constellation points from 64.
- **AM-DSB → WBFM (84% confused):** many analog examples were recorded during silent audio, which looks identical for both. This is a known limitation of the dataset.

### 3. Generalization to our own transmitter

The network was never trained on our transmitter, yet identified it correctly by majority vote at every SNR tested (125 windows each). The only adaptation needed was rescaling each window to RadioML's power level (our signal was about 15,000 times stronger).

| SNR | 20 dB | 10 dB | 6 dB | 4 dB |
|---|---|---|---|---|
| Windows voting QPSK | 120 | 116 | 101 | 93 |

### 4. Receiver stages

**Noise only, perfect sync: matches theory exactly.**

| Es/N0 (dB) | 0 | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| Measured BER | 0.1584 | 0.1039 | 0.0569 | 0.0227 | 0.0059 | 0.00069 | 0.00002 |
| Theory | 0.1587 | 0.1040 | 0.0565 | 0.0230 | 0.0060 | 0.00078 | 0.00003 |

![Constellations at 20, 10 and 4 dB](figures/fog_1.png)

**Timing.** True delay 5.4 samples, unknown to the receiver. The maximum-energy sampling phase gives 1 error in 2,000 bits, in line with theory (≈ 1.6).

![Wrong vs correct sampling phase](figures/timing_1.png)

**Doppler (4th-power method).** The QPSK phases 45°, 135°, 225° and 315° all become 180° at the 4th power, so the data vanishes and four times the rotation remains. Estimated offset **0.000199** vs true **0.000200** cycles/sample.

![Frequency offset estimation](figures/spin_1.png)

**Phase ambiguity.** Only the correct rotation (180°) matched all 32 sync bits (the others: 0/32, 16/32, 16/32), giving 1 error in 1,968 bits at 10 dB.

![Carrier recovery and ambiguity resolution](figures/unspin_1.png)

**Detection and frame sync.** First symbol found at sample 3049 vs the true 3048.4, with 32/32 sync bits.

![Energy detection and sync-word search](figures/find_start_1.png)

### 5. Effect of signal recovery: full blind receiver vs theory

Each trial uses a random arrival time, fractional delay, Doppler offset and carrier phase, none of which the receiver knows.

| Es/N0 (dB) | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| Blind receiver | 0.1072 | 0.0617 | 0.0237 | 0.0074 | 0.0015 | 0.0002 |
| Theory (perfect sync) | 0.1040 | 0.0565 | 0.0230 | 0.0060 | 0.0008 | 0.00003 |

Without recovery, the same signals are undecodable (a half-symbol timing error alone gave 886 wrong bits out of 2,000; uncorrected Doppler smears the constellation into a ring). With recovery, the blind receiver stays within 0.3–0.6 dB of a receiver that knows everything.

### 6. Error correction: a fair comparison

All comparisons use the **same energy per information bit**, so coded links pay for their extra bits with more noise per transmitted bit.

**Repetition code (×3) gives no real gain** (at 6 dB per information bit):

| Scheme | Bit error rate |
|---|---|
| No code | 0.0230 |
| ×3, soft combining, but **3× the energy** (unfair) | 0.0003 |
| ×3, equal energy, majority vote | 0.0415 (worse) |
| ×3, equal energy, soft combining | 0.0226 (no gain) |

**Voyager convolutional code with the full blind receiver:**

| SNR per info bit (dB) | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| Uncoded, theory | 0.079 | 0.057 | 0.038 | 0.023 | 0.013 | 0.0060 |
| Blind receiver, uncoded | 0.081 | 0.062 | 0.041 | 0.026 | 0.014 | 0.0071 |
| Blind receiver + Voyager code | 0.41 | 0.20 | 0.0039 | 0.0019 | **0** | **0** |

Above about 5 dB the coded link is error-free (0 errors in 18,000 bits) where the uncoded link loses about 1%. Below about 4 dB every code falls off a "cliff" and performs worse than no code. In a separate test, deliberately flipping 40 code bits still gave 0 message errors.

**Same energy, same channel (7 dB), a 134-character message:**

```
Uncoded (16 bit errors):
Hello Earth! This is the deep-space p2obe. �ll sxsTems�nominal.�Sending z�ience dqta now, SiGnal is weck, b}t our co`e kemps it clean.

Voyager code (0 bit errors):
Hello Earth! This is the deep-space probe. All systems nominal. Sending science data now. Signal is weak, but our code keeps it clean.
```

![Coded vs uncoded demo](figures/demo_1.png)

### Key observations
- The blind receiver performs within a fraction of a dB of the theoretical limit, so synchronization is not the bottleneck; noise is.
- QAM16 / QAM64 and AM-DSB / WBFM confusions are structural (window length and silent audio), not model failures.
- Below −10 dB, classification accuracy falls to near chance (≈ 9% for 11 classes).
- Error correction only helps above its cliff (about 4–5 dB per bit here); below it, coding makes things worse.

---

## Live Demo

```powershell
python live_demo.py
```

Type any message (up to 60 characters), choose **BPSK, QPSK or 8PSK**, and press **TRANSMIT**. The window animates every stage:

1. Message → bits → Voyager code
2. The transmitted constellation, with bit labels
3. The recording building up through noise, Doppler and an unknown delay
4. Burst detection, then the CNN's votes, confirmed by the sync word
5. The rotating cloud of samples snapping into clean clusters
6. The message typed out twice: without coding (errors in red) and with the Voyager code

The noise slider defaults to a level above each modulation's coding cliff (BPSK 4.5 dB, QPSK 4.5 dB, 8PSK 9 dB). Lowering it shows the cliff live. At the defaults, a stress test of 30 transmissions per modulation gave **0 failures**.

---

## Real Satellite Data: A Reality Check

We examined a SatNOGS frame export for the satellite **QB50P2** (1,096 frames from amateur ground stations, 2016–2026). It contains already-decoded text rather than raw IQ samples, so our receiver cannot run on it, but it is instructive:

- **4 frames (March 2016)** are genuine AX.25 telemetry packets with the callsign `QB50P2`.
- **The other 1,092 frames are almost certainly decoded noise.** 81.5% of their characters are E, I, T, S, N or A, the shortest Morse symbols, which is the signature of a Morse decoder running on pure noise (English text would be around 50%).

This is exactly the failure our **Find** stage prevents: without a detector that measures the noise floor, ground stations turn noise into junk frames.

---

## Challenges & Lessons Learned

- **RadioML has no ground-truth bits.** It can test classification but not decoding, so we built our own transmitter and channel model with known bits.
- **cuDNN was ~40× slower on our laptop GPU.** Training appeared frozen; timing each step in isolation showed cuDNN's 1D convolutions took 4.6 s per batch vs 0.1 s without it. Disabling cuDNN cut training from ~15 hours to 20 minutes.
- **Two Python installations on one machine** (MSYS2 and Python 3.11) caused missing-library errors. A per-project virtual environment fixed it.
- **Domain shift.** Our signals were ~15,000× stronger than RadioML's; rescaling each window to the training data's power was enough for the CNN to generalize.
- **A periodic sync word fooled the frame search.** A sync pattern that walks around the constellation looks like itself when rotated 90° and shifted, so it matched at the wrong place. Switching to the CCSDS marker `1ACFFC1D` fixed it.
- **End-to-end testing exposed two receiver bugs:** the noise floor was misjudged when a long coded frame filled most of the recording, and the detected burst sometimes clipped the first sync symbols. Both were fixed (quietest-10% noise estimate, safety margin).
- **8th-power noise spikes.** For 8PSK, raising samples to the 8th power let a few noisy samples dominate the Doppler search. Using phase only (unit magnitude) fixed it.
- **8PSK timing.** "Highest energy" sometimes picked the wrong sampling instant for 8PSK; searching timing and Doppler jointly took failures from 4 in 150 to 0.
- **Fair comparisons matter.** The repetition code looked 70× better until energy per information bit was held equal; then it gave nothing.
- **RadioML is synthetic and has known errata,** so results may not transfer directly to real hardware.

---

## Future Work

- Test on **raw IQ recordings** from the [SatNOGS network](https://network.satnogs.org).
- Track a **changing Doppler offset** (Doppler rate) over real satellite passes.
- **Fractional timing** (interpolating between samples) to close the remaining gap to theory at high SNR.
- Evaluate on **RadioML 2018.01A** (1,024-sample windows) to separate QAM16 from QAM64.
- **Stronger codes:** concatenate the convolutional code with Reed–Solomon (as on Voyager), or use turbo / LDPC codes, to move the coding cliff to lower SNR.
- Demodulators for non-PSK classes (QAM, FSK, analog).

---

## Citations & License

### Datasets

**RadioML 2016.10A and 2018.01A** — DeepSig Inc., licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial use, with attribution).

> O'Shea, T. J., & West, N. (2016). *Radio Machine Learning Dataset Generation with GNU Radio.* Proceedings of the GNU Radio Conference.

Dataset page: https://www.deepsig.ai/datasets/

**SatNOGS** — Libre Space Foundation (QB50P2 frame data).

### Methods

- A. J. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm," *IEEE Transactions on Information Theory*, 1967.
- A. J. Viterbi and A. M. Viterbi, "Nonlinear estimation of PSK-modulated carrier phase with application to burst digital transmission," *IEEE Transactions on Information Theory*, 1983.
- CCSDS 131.0-B, *TM Synchronization and Channel Coding* (rate-1/2, K = 7 convolutional code; attached sync marker `1ACFFC1D`).

### Code
[Choose a license for your own code, e.g. MIT. Note that the datasets keep their own CC BY-NC-SA 4.0 license regardless of the code license.]

---

*Built by Rover Rangers for the MATLAB in Space Hackathon.*
