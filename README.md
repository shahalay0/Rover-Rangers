# Rover Rangers — Deep-Space Communication & Signal Intelligence

**MATLAB in Space Hackathon · Track 1 (Advanced)**

Deep-space links are weak, noisy, and constantly drifting. Signals arrive buried in noise, distorted by multipath, and shifted by Doppler and oscillator drift. This project explores how to **detect, clean up, and identify** radio signals under those conditions, using the RadioML benchmark datasets as a stand-in for real deep-space transmissions.

> 🚧 *Replace the bracketed placeholders below with your team's specifics as the project develops.*

---

## Table of Contents

1. [Team](#team)
2. [Problem Statement](#problem-statement)
3. [Approach](#approach)
4. [Repository Structure](#repository-structure)
5. [Getting Started](#getting-started)
6. [Datasets](#datasets)
7. [Usage](#usage)
8. [Results](#results)
9. [Challenges & Lessons Learned](#challenges--lessons-learned)
10. [Future Work](#future-work)
11. [Citations & License](#citations--license)

---

## Team

**Rover Rangers** — a team of roboticists applying estimation, filtering, and learning techniques to space communications.

---

## Problem Statement

A ground station listening to a distant spacecraft faces three core problems:

- **Low SNR:** the signal is often at or below the noise floor.
- **Channel impairments:** multipath fading, carrier frequency offset, Doppler drift, and sample-rate mismatch distort the waveform.
- **Unknown signal type:** the receiver may not know in advance which modulation scheme is being used.

**Our goal:** [One or two sentences describing exactly what your system does, e.g. "Build a MATLAB pipeline that compensates for frequency drift with an adaptive filter, then classifies the modulation type of each received burst, and measure how accuracy degrades as SNR drops."]

---

## Approach

Our pipeline has [three] stages:

```
 Raw IQ samples ──► 1. Preprocessing ──► 2. Signal Recovery ──► 3. Classification ──► Modulation label + confidence
                     (normalize,          (adaptive filtering,     (features / neural
                      impairments)         drift compensation)      network)
```

### 1. Preprocessing
- Load IQ samples and combine into complex baseband signals: `x = I + jQ`.
- Normalize power per example.
- [Optional] Inject extra impairments on top of RadioML (Doppler drift, dropouts, frequency hops) to create harder test cases.

### 2. Signal Recovery
- [Describe your method, e.g. LMS/RLS adaptive equalizer, Kalman filter for carrier phase/frequency tracking, PLL, or matched filtering.]
- [Why you chose it — e.g. a Kalman filter naturally models slowly drifting frequency offset as a state.]

### 3. Modulation Classification
- [Describe your classifier, e.g. a 1D CNN trained with the Deep Learning Toolbox, or hand-crafted features (higher-order cumulants, spectral features) with an SVM.]
- **Input:** [e.g. 2 × 128 IQ matrix]
- **Output:** one of 11 modulation classes (8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM).
- **Train / val / test split:** [e.g. 60 / 20 / 20, stratified across modulation and SNR]

---

## Repository Structure

```
Rover-Rangers/
├── README.md               ← you are here
├── .gitignore              ← keeps large datasets out of the repo
├── data/                   ← datasets go here (NOT committed, see below)
│   └── README.md
├── scripts/
│   ├── download_data.sh    ← downloads RadioML datasets
│   └── convert_pkl_to_mat.py ← converts the 2016 .pkl into a MATLAB .mat file
├── src/
│   ├── preprocessing/      ← loading, normalization, impairment injection
│   ├── recovery/           ← adaptive filtering / drift compensation
│   ├── classification/     ← model definition and training
│   └── utils/              ← plotting and helper functions
├── models/                 ← trained models (small ones only)
├── results/                ← figures, confusion matrices, accuracy-vs-SNR plots
└── main.m                  ← runs the full pipeline end to end
```

---

## Getting Started

### Requirements

**MATLAB** [R20XXx] or newer, with:
- Signal Processing Toolbox
- Communications Toolbox
- Deep Learning Toolbox [if using a neural network]
- Statistics and Machine Learning Toolbox [if using classical ML]

**Python 3.8+** (only needed once, to convert the 2016 dataset):
```bash
pip install numpy scipy
```

### Installation

```bash
git clone https://github.com/shahalay0/Rover-Rangers.git
cd Rover-Rangers
bash scripts/download_data.sh
python scripts/convert_pkl_to_mat.py
```

Then open MATLAB in the `Rover-Rangers` folder and run:
```matlab
main
```

---

## Datasets

> ⚠️ **The datasets are not stored in this repository.** They exceed GitHub's 100 MB per-file limit. Use the download script or the links below, and place the files in `data/`.

| Dataset | File | Size | Used for |
|---|---|---|---|
| RadioML 2016.10A | `RML2016.10a_dict.pkl` | 641 MB | Main development and evaluation |
| RadioML 2018.01A *(optional)* | `GOLD_XYZ_OSC.0001_1024.hdf5` | 21.4 GB | Harder benchmark, 24 modulations |

### RadioML 2016.10A
- 11 modulation types, 20 SNR levels (−20 dB to +18 dB, 2 dB steps)
- 1,000 examples per (modulation, SNR) pair → 220,000 examples
- Each example: 128 complex samples stored as 2 rows (I and Q)
- Synthetic, generated with GNU Radio, including white noise, multipath fading, frequency offset, and sample-rate drift

### RadioML 2018.01A
- 24 modulation types, 26 SNR levels (−20 dB to +30 dB)
- ~2.5 million examples of 1,024 complex samples each

### Download

`scripts/download_data.sh`:
```bash
#!/bin/bash
mkdir -p data
curl -L -o data/RML2016.10a_dict.pkl \
  "https://huggingface.co/datasets/FlowVortex/RML/resolve/main/RML2016.10a_dict.pkl?download=true"

# Optional 21.4 GB dataset — uncomment if needed
# curl -L -o data/GOLD_XYZ_OSC.0001_1024.hdf5 \
#   "https://huggingface.co/datasets/FlowVortex/RML/resolve/main/GOLD_XYZ_OSC.0001_1024.hdf5?download=true"
```

### Loading the data in MATLAB

The 2016 dataset is a Python pickle, which MATLAB can't read directly. Convert it once with this script:

`scripts/convert_pkl_to_mat.py`:
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
        x = d[(m, s)]                       # (1000, 2, 128)
        X.append(x)
        mod_idx += [mods.index(m) + 1] * len(x)   # 1-based for MATLAB
        snr += [s] * len(x)

sio.savemat(
    "data/rml2016a.mat",
    {
        "X": np.vstack(X).astype(np.float32),     # (220000, 2, 128)
        "mod_idx": np.array(mod_idx),
        "snr": np.array(snr),
        "mods": np.array(mods, dtype=object),     # becomes a cell array
    },
    do_compression=True,
)
print("Saved data/rml2016a.mat")
```

Then in MATLAB:
```matlab
S = load("data/rml2016a.mat");
k = 1;                                         % example index
x = squeeze(S.X(k,1,:)) + 1i*squeeze(S.X(k,2,:));   % complex signal, 128x1
fprintf("Modulation: %s, SNR: %d dB\n", strtrim(S.mods{S.mod_idx(k)}), S.snr(k));
```

The 2018 dataset is HDF5, which MATLAB reads natively. Read slices only, since the full file won't fit in memory. Note that MATLAB reverses the dimension order compared with Python:
```matlab
f = "data/GOLD_XYZ_OSC.0001_1024.hdf5";
X = h5read(f, "/X", [1 1 1], [2 1024 1000]);   % first 1000 examples, size 2x1024x1000
Y = h5read(f, "/Y", [1 1], [24 1000]);         % one-hot labels
Z = h5read(f, "/Z", [1 1], [1 1000]);          % SNR values
```

---

## Usage

| Task | Command |
|---|---|
| Run the full pipeline | `main` |
| Train the classifier | `[e.g. run("src/classification/train_classifier.m")]` |
| Evaluate on the test set | `[e.g. run("src/classification/evaluate.m")]` |
| Generate result plots | `[e.g. run("src/utils/plot_results.m")]` |

[Add any configurable parameters here, e.g. SNR range, model type, number of training epochs.]

---

## Results

> [Fill in once you have numbers. Plots go in `results/` and can be embedded with `![caption](results/filename.png)`.]

### Classification accuracy

| SNR range | Accuracy |
|---|---|
| High SNR (≥ 10 dB) | [xx %] |
| Medium SNR (0 to 8 dB) | [xx %] |
| Low SNR (< 0 dB) | [xx %] |
| Overall | [xx %] |

### Accuracy vs. SNR
![Accuracy vs SNR](results/accuracy_vs_snr.png)

### Confusion matrix
![Confusion matrix](results/confusion_matrix.png)

### Effect of signal recovery
[Compare accuracy with and without your filtering stage, e.g. "Adding the Kalman drift compensator improved accuracy at 0 dB from xx % to yy %."]

### Key observations
- [e.g. QAM16 and QAM64 are frequently confused, since they look similar at low SNR.]
- [e.g. AM-DSB and WBFM are hard to separate, which is a known issue with this dataset.]
- [e.g. Below −10 dB, accuracy falls to near chance (≈ 9 % for 11 classes).]

---

## Challenges & Lessons Learned

- [e.g. Getting the dataset into MATLAB required a Python conversion step.]
- [e.g. Balancing model size against training time during a short hackathon.]
- [e.g. RadioML is synthetic and has known errata, so results may not transfer directly to real hardware.]

---

## Future Work

- Test on **real satellite recordings** from the [SatNOGS network](https://network.satnogs.org).
- Add more realistic deep-space impairments, such as large Doppler ramps and long dropouts.
- Evaluate on the larger **RadioML 2018.01A** dataset.
- [Your own ideas.]

---

## Citations & License

### Datasets

**RadioML 2016.10A and 2018.01A** — DeepSig Inc., licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial use, with attribution).

> O'Shea, T. J., & West, N. (2016). *Radio Machine Learning Dataset Generation with GNU Radio.* Proceedings of the GNU Radio Conference.

Dataset page: https://www.deepsig.ai/datasets/

**SatNOGS** — Libre Space Foundation (if real satellite data is used).

### Code
[Choose a license for your own code, e.g. MIT. Note that the datasets keep their own CC BY-NC-SA 4.0 license regardless of the code license.]

---

*Built by Rover Rangers for the MATLAB in Space Hackathon.*
