# Rover Rangers: An Autonomous Receiver for Weak Satellite Signals

**MATLAB in Space Hackathon (SEDS × ESS at Northeastern), October 3, 2026**
**Track 1 (Advanced): Deep-Space Communication & Signal Intelligence**

A ground station listening to a spacecraft receives a signal that is weak, buried in noise and shifted in frequency by Doppler. Rover Rangers is a receiver that handles all of it **on its own**: it finds the signal, identifies its modulation, cleans it up and decodes it back into data, **with no human tuning**.

The project has four parts, each built by a different team member, moving step by step from simulated benchmarks to a real satellite pass.

| Part | What it does | Headline result |
|---|---|---|
| **1. Recognizing signals and fixing Doppler** | CNN modulation classifier + blind Doppler correction | 88% accuracy (11 classes, SNR ≥ 0 dB); Doppler recovery 33% → 96.7% |
| **2. Blind deep-space receiver** | Find, identify, clean and decode with the Voyager code | Within 0.3–0.6 dB of theory; 0 errors in 18,000 bits at 7 dB |
| **3. Real telemetry through a hostile channel** | 861 real SatNOGS frames through drift, dropouts and hops | 99.2% recovered byte-for-byte at 2 dB |
| **4. Decoding a real satellite pass** | Real OrigamiSat-2 recording, decoded end to end | 71 CRC-verified packets; 67 of 75 main packets (89%) |

Each part's numbers come from its own models and test conditions, so they are reported separately rather than combined.

---

## Part 1: Recognizing signals and fixing Doppler

**Goal:** identify the modulation of a weak signal automatically, and remove Doppler shift without being told the modulation.

**What we did**
1. Trained a compact 1D CNN (~150k parameters) on raw I/Q samples from **RadioML 2016.10A** (220,000 examples, 11 modulations, SNR −20 to +18 dB), with a 70/15/15 split balanced by modulation and SNR.
2. Stress-tested it with simulated Doppler (random offsets up to ±0.02 cycles/sample).
3. Built a **blind hypothesis-testing receiver**: it estimates and removes the offset assuming M = 2 and M = 4 (M-th power method), reclassifies each version, and keeps the most confident answer consistent with its hypothesis. No true labels are used.
4. Repeated classification on **RadioML 2018.01A** (24 modulations, 1024-sample windows) using a balanced 10% subset.
5. Built an end-to-end **link demo**: spacecraft commands and telemetry in a simplified CCSDS frame (sync word `1ACFFC1D`, length byte, CRC-16), sent through unknown Doppler and noise, then decoded blindly.

**Results**

| Test | Result |
|---|---|
| Accuracy, SNR ≥ 0 dB (11 classes) | **88%**; crosses 50% at −6 dB |
| Doppler, BPSK/PAM4/QPSK, SNR ≥ 0 dB | 98.4% clean → 33.3% with Doppler → **96.7% blind-corrected** (oracle: 96.0%) |
| Side effect on the other 8 classes | −1.2 points |
| RadioML 2018, SNR ≥ 10 dB (24 classes) | **84%**; QAM16 and 8PSK at 100% |
| Blind offset estimation, 128 → 1024 samples | QPSK 85% → 100%; 16QAM ~30% → ~90% |
| Link demo, BPSK at 0 dB + QPSK at 5/10 dB | **8 of 9 frames** exact; the 9th had 1 flipped bit, rejected by the CRC |

Remaining errors come from the data rather than the model: QAM16 vs QAM64 (too few symbols in 128 samples) and WBFM vs AM-DSB (silent source audio in the dataset).

### Part 1 in pictures

![The 11 RadioML 2016 modulations at +18 dB: each leaves a distinct shape](images/p1_constellations.png)
*The 11 RadioML 2016 modulations at +18 dB: each leaves a distinct shape*

![Classifier accuracy vs SNR: chance below −12 dB, about 88% from 0 dB up](images/p1_accuracy_vs_snr.png)
*Classifier accuracy vs SNR: chance below −12 dB, about 88% from 0 dB up*

![Confusion matrix at SNR ≥ 0 dB: remaining errors are QAM16/QAM64 and WBFM/AM-DSB](images/p1_confusion_high_snr.png)
*Confusion matrix at SNR ≥ 0 dB: remaining errors are QAM16/QAM64 and WBFM/AM-DSB*

![Injected Doppler turns BPSK's line into a spinning loop; the blind estimate (0.0055 vs true 0.0055) restores it](images/p1_doppler_constellations.png)
*Injected Doppler turns BPSK's line into a spinning loop; the blind estimate (0.0055 vs true 0.0055) restores it*

![Blind offset estimation success vs SNR for BPSK, PAM4 and QPSK](images/p1_doppler_estimator.png)
*Blind offset estimation success vs SNR for BPSK, PAM4 and QPSK*

![Blind hypothesis-test correction (green) matches the oracle (red): 33% → 96.7% for BPSK/PAM4/QPSK](images/p1_doppler_blind_correction.png)
*Blind hypothesis-test correction (green) matches the oracle (red): 33% → 96.7% for BPSK/PAM4/QPSK*

![RadioML 2018: 24-class accuracy vs SNR, about 84% at SNR ≥ 10 dB](images/p1_rml2018_accuracy.png)
*RadioML 2018: 24-class accuracy vs SNR, about 84% at SNR ≥ 10 dB*

![RadioML 2018 confusion matrix at SNR ≥ 10 dB](images/p1_rml2018_confusion.png)
*RadioML 2018 confusion matrix at SNR ≥ 10 dB*

![Blind offset estimation with 128 (dashed) vs 1024 (solid) samples on RadioML 2018](images/p1_rml2018_window_length.png)
*Blind offset estimation with 128 (dashed) vs 1024 (solid) samples on RadioML 2018*

![Link demo: 'CMD DEPLOY_ANTENNA' recovered at 0 dB with a valid CRC](images/p1_link_demo.png)
*Link demo: 'CMD DEPLOY_ANTENNA' recovered at 0 dB with a valid CRC*

---

## Part 2: A complete blind deep-space receiver

**Goal:** turn raw noisy samples into the original message, knowing only the CCSDS sync marker `1ACFFC1D`.

**Pipeline**
1. **Find:** block energy against a noise floor measured from the recording, then a sync-marker search.
2. **Identify:** 1D CNN on 128-sample windows trained on RadioML, majority vote across windows.
3. **Clean:** energy-peak symbol timing, 4th-power Doppler search, sync-word phase resolution.
4. **Decode:** Voyager K = 7, rate-1/2 convolutional code (CCSDS standard) with soft-decision Viterbi decoding.

**Results**
- **79%** classification accuracy at SNR ≥ 0 dB; recognizes a transmitter it never trained on (93 of 125 windows correct at 4 dB).
- Measured BER matches textbook QPSK theory with perfect sync; the blind receiver stays **within 0.3–0.6 dB of theory** from 2 to 10 dB.
- Frame start found to **±0.6 sample** at Es/N0 10 dB; Doppler estimated as 0.000199 vs true 0.000200.
- **0 errors in 18,000 bits** at 7 and 8 dB with the Voyager code (uncoded: 1.4% / 0.7%).
- End-to-end demo: a 134-character probe message arrives with 0 errors (16 errors without coding, at the same energy).

### Part 2 in pictures

![Classifier accuracy climbs from chance to about 80% with SNR](images/p2_accuracy_vs_snr.jpg)
*Classifier accuracy climbs from chance to about 80% with SNR*

![Errors at high SNR come from two look-alike pairs](images/p2_confusion.jpg)
*Errors at high SNR come from two look-alike pairs*

![With perfect synchronisation, measured BER matches QPSK theory](images/p2_ber_vs_theory.jpg)
*With perfect synchronisation, measured BER matches QPSK theory*

![The blind receiver tracks the theoretical limit within 0.3–0.6 dB](images/p2_blind_receiver_vs_theory.jpg)
*The blind receiver tracks the theoretical limit within 0.3–0.6 dB*

![Full system: error-free with the Voyager code where uncoded loses 1%](images/p2_full_system_coded.jpg)
*Full system: error-free with the Voyager code where uncoded loses 1%*

![Same energy, same channel: uncoded message corrupted, coded message clean](images/p2_demo_corrupted_vs_clean.jpg)
*Same energy, same channel: uncoded message corrupted, coded message clean*

![End to end: find, identify, clean, decode](images/p2_end_to_end.jpg)
*End to end: find, identify, clean, decode*

---

## Part 3: Real satellite telemetry through a hostile channel

**Goal:** prove real satellite telemetry survives a difficult link and is delivered byte-for-byte.

**What we did**
1. Collected **861 real telemetry frames** (CW beacon frames) from **27 SatNOGS ground stations** over one week (Sept 26 – Oct 3, 2026).
2. Framed each one like a spacecraft radio: sync word `1ACFFC1D`, sequence number, length and CRC.
3. Sent them through a simulated channel with Doppler drift, a dropout, a frequency hop and noise.
4. Recovered them with the autonomous receiver (carrier tracking, re-acquisition, IQNet classification, demodulation) and compared every byte.

**Results**
- **99.2%** of frames (854 of 861) recovered byte-exact at 2 dB; ~98–99% from 0 dB up.
- Automatic re-acquisition after the dropout (0.13 s) and the frequency hop.
- IQNet identified the signal as BPSK with **95% confidence**, no human input.

### Part 3 in pictures

![Before: Doppler curve, dropout and hop. After: a steady signal at 0 Hz for the whole pass](images/p3_before_after_spectrum.jpg)
*Before: Doppler curve, dropout and hop. After: a steady signal at 0 Hz for the whole pass*

![Byte-exact frame recovery vs SNR: 99% from 0 dB up](images/p3_recovery_vs_snr.jpg)
*Byte-exact frame recovery vs SNR: 99% from 0 dB up*

![Lost frames line up with the dropout and the hop; the receiver re-acquires on its own](images/p3_channel_events.jpg)
*Lost frames line up with the dropout and the hop; the receiver re-acquires on its own*

---

## Part 4: Decoding a real satellite pass

**Goal:** decode real telemetry from a real satellite recording.

**Data:** OrigamiSat-2 (NORAD 68795), 437.505 MHz, AFSK 1200 baud, AX.25. SatNOGS observation 15110269, ground station 4869, 2026-10-03, 06:43:51–06:49:51 UTC; 355 s of audio at 48 kHz.

**Pipeline:** find tones → band-pass (700–2700 Hz) → mark/space discrimination → bit clock (DPLL) → NRZI, HDLC flags and bit unstuffing → CRC-16 and AX.25 parsing. Five decoders with different mark/space balances run in parallel; only CRC-valid frames are kept.

**Results**
- **71 AX.25 frames** decoded, every one passing its CRC (SatNOGS reported 50 for the same observation; count comparison only).
- **67 of 75** main packets recovered (89%), measured with the satellite's own packet counter.
- 355 s of audio decoded in **27 s** on a laptop, with no manual tuning.

### Part 4 in pictures

![The OrigamiSat-2 recording; red lines mark decoded packets](images/p4_recording.jpg)
*The OrigamiSat-2 recording; red lines mark decoded packets*

![Where the packets are in the 6-minute pass](images/p4_packet_locations.jpg)
*Where the packets are in the 6-minute pass*

![Before and after filtering: clean tones and readable bits](images/p4_before_after_filtering.jpg)
*Before and after filtering: clean tones and readable bits*

![67 of 75 main packets decoded, checked with the satellite's own counter](images/p4_missed_packets.jpg)
*67 of 75 main packets decoded, checked with the satellite's own counter*

![Five decoders in parallel; the CRC decides which packets are kept](images/p4_self_tuning.jpg)
*Five decoders in parallel; the CRC decides which packets are kept*

![Decoded telemetry log](images/p4_telemetry_log.jpg)
*Decoded telemetry log*

---

## Repository structure

> Check these paths against the repo and adjust names if needed. The `images/` folder must sit next to this README for the pictures to show.

```
Rover-Rangers/
├── part1_classification_doppler/
│   ├── track1_part1_radioml.ipynb      # RadioML 2016 classifier, Doppler test, blind correction
│   ├── track1_part1_radioml.py         # same, headless (HPC)
│   ├── track1_part2_next_steps.ipynb   # Doppler-augmented training, window-length study, SatNOGS pipeline
│   ├── track1_part3_radioml2018.ipynb  # RadioML 2018, 24 classes, 1024 samples
│   ├── track1_part3_radioml2018.py     # same, headless
│   ├── run_rml2018.sbatch              # Slurm job for the HPC
│   ├── recovery.py                     # end-to-end command/telemetry link demo
│   └── outputs/                        # plots, metrics.json, trained models
├── part2_blind_receiver/
│   ├── radio_tools.py                  # transmitter, channel, blind receiver, Viterbi
│   ├── model.py, train.py              # classifier
│   └── demo_full.py                    # end-to-end demo
├── part3_telemetry_frames/
│   ├── satnogs_frames.py
│   ├── pipeline.py
│   └── events.log
├── part4_satnogs_pass/
│   └── results/satnogs_frames.csv      # all 71 decoded frames
├── slides/                              # presentation decks
├── images/                              # figures used in this README
└── README.md
```

## Setup

```bash
conda create -n radioml python=3.10 -y
conda activate radioml
pip install torch scikit-learn matplotlib scipy h5py ipykernel
```

On a GPU with an older driver (CUDA 12.x), install a matching PyTorch build:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## Data (not included in the repo)

| Dataset | Size | Source |
|---|---|---|
| RadioML 2016.10A (`RML2016.10a_dict.pkl`) | 641 MB | [Hugging Face mirror](https://huggingface.co/datasets/FlowVortex/RML/resolve/main/RML2016.10a_dict.pkl?download=true) · [DeepSig](https://www.deepsig.ai/datasets/) |
| RadioML 2018.01A (`GOLD_XYZ_OSC.0001_1024.hdf5`) | 21.4 GB | [Hugging Face mirror](https://huggingface.co/datasets/FlowVortex/RML/resolve/main/GOLD_XYZ_OSC.0001_1024.hdf5?download=true) |
| SatNOGS telemetry frames and recordings | — | [db.satnogs.org](https://db.satnogs.org) · [network.satnogs.org](https://network.satnogs.org) |

On an HPC cluster, download the 21 GB file to scratch storage rather than your home directory.

## How to run

**Part 1**
```bash
# RadioML 2016: place the .pkl next to the notebook, then run all cells
jupyter notebook track1_part1_radioml.ipynb      # set QUICK = True for a fast smoke test first

# RadioML 2018 on a Slurm cluster
sbatch --export=ALL,MAX_MINUTES=8 run_rml2018.sbatch

# End-to-end link demo (numpy + matplotlib only)
MPLBACKEND=Agg python recovery.py                # writes outputs/link_demo.png
```

**Parts 2–4:** see the scripts in each folder (for example `python demo_full.py` in Part 2).

## Limitations

- Most channel impairments are simulated; full real-RF IQ testing is still to come.
- Doppler is mostly modelled as a constant offset within a window; real passes need Doppler-rate tracking.
- Blind M-th power correction covers BPSK/PAM4/QPSK; 8PSK and higher-order QAM need longer windows and timing recovery.
- Part 2's demodulator is QPSK-only; Part 4 covers AFSK 1200 / AX.25 only.
- RadioML 2018 was trained on a 10% subset for 8 minutes because of time limits.

## Next steps

- Run the full blind receiver on real SatNOGS IQ recordings.
- Compare blind Doppler tracks with orbit-predicted Doppler (TLE) and track Doppler rate.
- Add BPSK/8PSK demodulators, symbol-timing recovery and GMSK 9600 support.
- Train on the full RadioML 2018 dataset with a deeper model; finish Doppler-augmented training.
- Add Reed–Solomon or LDPC coding to lower the decoding "cliff".

## Team

| Part | Member |
|---|---|
| 1. Recognizing signals and fixing Doppler | Priyanka Lakariya |
| 2. Blind deep-space receiver | Alay |
| 3. Real telemetry through a hostile channel | _add name_ |
| 4. Decoding a real satellite pass | _add name_ |

## Credits and references

- **RadioML 2016.10A and 2018.01A:** DeepSig Inc., CC BY-NC-SA 4.0 (non-commercial, with attribution). T. J. O'Shea and N. West, "Radio Machine Learning Dataset Generation with GNU Radio," *Proc. GNU Radio Conference*, 2016. DeepSig notes known errata in these datasets; they are used here for prototyping, not as ground truth for real hardware.
- **SatNOGS:** Libre Space Foundation, CC BY-SA 4.0. OrigamiSat-2 observation 15110269, ground station 4869.
- A. J. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm," *IEEE Trans. Information Theory*, 1967.
- A. J. Viterbi and A. M. Viterbi, "Nonlinear estimation of PSK-modulated carrier phase with application to burst digital transmission," *IEEE Trans. Information Theory*, 1983.
- CCSDS 131.0-B, *TM Synchronization and Channel Coding* (rate-1/2, K = 7 convolutional code; attached sync marker 1ACFFC1D).
