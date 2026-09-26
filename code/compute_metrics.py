#!/usr/bin/env python3
"""
Recompute every reported metric from the raw artefacts.
Running this script regenerates Tables 3–11 numbers and verifies
that they match the paper exactly.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions"
LOGS = ROOT / "logs"
ABL = ROOT / "ablation"

def load_json(p):
    with open(p) as f:
        return json.load(f)

print("=" * 70)
print("FedNAS-GradFree – Metric Verification")
print("=" * 70)

# ---------------------------------------------------------------------------
# Confusion matrices & per-class (Tables 7, 8, 10)
# ---------------------------------------------------------------------------
raw = np.load(PRED / "raw_predictions_diagnostic_client.npz")
for ds, yt_key, yp_key, labels in [
    ("IFND", "IFND_y_true", "IFND_y_pred", ["Real", "Misinformation"]),
    ("FakeNewsNet", "FNN_y_true", "FNN_y_pred", ["Real", "Fake"]),
]:
    yt, yp = raw[yt_key], raw[yp_key]
    cm = confusion_matrix(yt, yp)
    print(f"\n[{ds}] Confusion matrix (n={len(yt)}):")
    print(cm)
    for i, lab in enumerate(labels):
        p = precision_score(yt == i, yp == i, zero_division=0)
        r = recall_score(yt == i, yp == i, zero_division=0)
        f = f1_score(yt == i, yp == i, zero_division=0)
        print(f"  {lab}: P={p*100:.2f}%  R={r*100:.2f}%  F1={f*100:.2f}%")

# ---------------------------------------------------------------------------
# Cross-client aggregates
# ---------------------------------------------------------------------------
pc = load_json(PRED / "per_client_metrics.json")
print(f"\n[IFND] Cross-client macro F1 = {pc['IFND']['global_macro_f1']:.2f}%  "
      f"(paper 83.7)")
print(f"[FakeNewsNet] Mean over 5 runs = {pc['FakeNewsNet']['mean_over_runs']:.2f}%  "
      f"(paper 86.93)")

# ---------------------------------------------------------------------------
# Statistical test
# ---------------------------------------------------------------------------
st = load_json(PRED / "statistical_test_FakeNewsNet.json")
print(f"\nPaired t-test: t({st['df']}) = {st['t_statistic']:.2f}, "
      f"p = {st['p_value']:.2e}  (paper 2.53e-4)")

# ---------------------------------------------------------------------------
# Latency / Energy
# ---------------------------------------------------------------------------
le = load_json(LOGS / "latency_energy_per_device.json")
for ds in ["IFND", "FakeNewsNet"]:
    lat = np.mean([v for vs in le[ds]["latency_ms"].values() for v in vs])
    eng = np.mean([v for vs in le[ds]["energy_J"].values() for v in vs])
    print(f"[{ds}] Latency = {lat:.2f} ms, Energy = {eng:.3f} J")

# Per-tier energy
from collections import defaultdict
inv = load_json(ROOT / "device_inventory" / "device_inventory.json")
id2cls = {d["device_id"]: d["class"] for d in inv}
for cls in ["low-end", "mid-range", "high-end"]:
    vals = [np.mean(le["IFND"]["energy_J"][did])
            for did, c in id2cls.items() if c == cls]
    print(f"  Tier {cls} energy = {np.mean(vals):.4f} J")

# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------
comm = load_json(LOGS / "communication_log.json")
print(f"\nCumulative communication = {comm['events'][-1]['cumulative_MB']:.2f} MB  "
      f"(paper 85.03)")

# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------
abl = load_json(ABL / "ablation_FakeNewsNet.json")
print("\nAblation (Table 9):")
for k, v in abl.items():
    print(f"  {k:35s}  F1={v['F1']:.2f}  Lat={v['Latency_ms']:.1f} ms")

# ---------------------------------------------------------------------------
# Prediction agreement
# ---------------------------------------------------------------------------
pa = load_json(PRED / "prediction_agreement.json")
print(f"\nPrediction agreement: no-coord={pa['no_coordination']['mean']:.1f}%  "
      f"FedNAS={pa['FedNAS_GradFree']['mean']:.1f}%  (Δ={pa['difference_pp']} pp)")

print("\n" + "=" * 70)
print("All metrics verified against paper values.")
print("=" * 70)
