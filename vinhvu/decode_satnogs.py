"""Decode a real SatNOGS pass recording (AFSK 1200 / AX.25).

  python decode_satnogs.py data/satnogs/satnogs_1234567_....ogg
  python decode_satnogs.py data/satnogs/audio.ogg --reference data/satnogs/frames.txt
  python decode_satnogs.py --demo        # synthetic stand-in recording, to test the pipeline

Outputs: results/satnogs_frames.csv, results/satnogs_overview.png
"""
import argparse
import os
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from src.afsk import decode_afsk
from src.satnogs import load_audio, load_reference_frames, compare_with_reference, make_test_pass, plot_results

p = argparse.ArgumentParser()
p.add_argument("audio", nargs="?", help="SatNOGS audio file (.ogg/.wav)")
p.add_argument("--reference", help="frames SatNOGS decoded (hex), for comparison")
p.add_argument("--baud", type=int, default=1200)
p.add_argument("--demo", action="store_true", help="generate and decode a synthetic test pass")
args = p.parse_args()
os.makedirs("results", exist_ok=True)

if args.demo or not args.audio:
    os.makedirs("data/satnogs", exist_ok=True)
    args.audio, args.reference = "data/satnogs/demo_pass.wav", "data/satnogs/demo_reference.txt"
    make_test_pass(args.audio, args.reference)
    print("DEMO MODE: synthetic test recording (not real satellite data)")

t0 = time.time()
x, fs = load_audio(args.audio)
print(f"loaded {args.audio}: {len(x)/fs:.1f} s at {fs} Hz")
frames, diag = decode_afsk(x, fs, baud=args.baud)
print(f"decoded {len(frames)} AX.25 frames with valid CRC in {time.time()-t0:.1f} s\n")
for f in frames:
    print(f"  t={f['time_s']:7.2f}s  {f['summary'][:110]}")

pd.DataFrame(frames).drop(columns=[], errors="ignore").to_csv("results/satnogs_frames.csv", index=False)
_, n_bursts = plot_results(x, fs, frames, diag)
print(f"candidate packet bursts in the audio: {n_bursts}")
print("\nsaved results/satnogs_frames.csv, results/satnogs_overview.png")

if args.reference and os.path.exists(args.reference):
    c = compare_with_reference(frames, load_reference_frames(args.reference))
    print(f"\ncomparison with SatNOGS decoder: we decoded {c['ours']}, SatNOGS listed {c['reference']}")
    print(f"  found {c['reference_found']}/{c['reference']} of SatNOGS' frames, "
          f"missed {c['missed']}, extra frames only we decoded: {c['only_ours']}")
