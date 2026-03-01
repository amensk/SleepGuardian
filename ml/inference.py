"""
inference.py
------------
Simulates a real-time apnea detection stream over a single CapnoBase record.

For each 10-second window the script:
  1. Pre-processes the signal (bandpass filter + Z-score).
  2. Runs the CNN-LSTM model  → ML apnea probability.
  3. Updates the ApneaTracker → ΔPaCO₂ estimate.
  4. Applies the fusion guard-rail:
       ALERT = (ml_prob > threshold) AND (ΔPaCO₂ > ALERT_THRESHOLD)
  5. Updates a live Matplotlib figure with three panels.

Usage
~~~~~
    python inference.py \\
        --data_dir data/ \\
        --record_id 0104_8min \\
        --model_path output/best_model.pt \\
        --threshold 0.5 \\
        --output_dir output/

Pass --no_gui to skip the interactive plot and only save the PNG + CSV.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from data_loader import (
    load_capnobase_record,
    preprocess_window,
    FS_CAPNO,
    WINDOW_SAMPLES,
)
from model import ApneaCNN_LSTM
from physiology import ApneaTracker, ALERT_THRESHOLD, INITIAL_PACO2


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device) -> ApneaCNN_LSTM:
    model = ApneaCNN_LSTM(in_channels=2, lstm_hidden=64, lstm_layers=2, dropout=0.0)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Stream simulation
# ---------------------------------------------------------------------------

def run_inference(
    record_id: str,
    data_dir: str,
    model_path: Optional[str],
    threshold: float = 0.5,
    output_dir: str = "output/",
    show_gui: bool = True,
) -> pd.DataFrame:
    """
    Simulate real-time inference over one CapnoBase record.

    Returns a DataFrame with per-window results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # ---- Load record ----
    csv_dir   = Path(data_dir) / "data" / "csv"
    sig_file  = csv_dir / f"{record_id}_signal.csv"
    if not sig_file.exists():
        sys.exit(f"Signal file not found: {sig_file}")

    result = load_capnobase_record(str(sig_file))
    if result is None:
        sys.exit(f"Could not load record {record_id}")

    signals, apnea_mask, insp_idx = result   # signals: (2, N)
    co2_raw = signals[0]                      # raw CO₂ waveform for plotting
    n_samples = signals.shape[1]
    n_windows = n_samples // WINDOW_SAMPLES
    total_sec = n_windows * 10

    print(f"Record        : {record_id}")
    print(f"Signal length : {n_samples} samples  ({n_samples/FS_CAPNO:.0f} s)")
    print(f"Windows       : {n_windows}  ×  10 s")

    # ---- Load model ----
    use_model = model_path is not None and Path(model_path).exists()
    if use_model:
        model = load_model(model_path, device)
        print(f"Model         : {model_path}")
    else:
        model = None
        print("Model         : NOT FOUND — using random scores (demo mode)")

    # ---- Physiology tracker ----
    tracker = ApneaTracker(window_sec=10.0, alert_threshold=ALERT_THRESHOLD)

    # ---- Per-window storage ----
    timestamps, ml_probs, co2_rises, alerts, gt_labels = [], [], [], [], []

    # ---- Time axis ----
    time_axis = np.arange(n_samples) / FS_CAPNO   # seconds

    # ---- Live plot setup ----
    if show_gui:
        matplotlib.use("TkAgg" if "DISPLAY" in os.environ or sys.platform == "darwin" else "Agg")
    else:
        matplotlib.use("Agg")

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    fig.suptitle(f"Real-time Apnea Detection — Record: {record_id}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.45)

    ax_co2, ax_ml, ax_phys = axes

    # Panel titles
    ax_co2.set_title("CO₂ Waveform  (red shading = ground-truth apnea)", fontsize=10)
    ax_ml.set_title("ML Apnea Probability per 10 s Window", fontsize=10)
    ax_phys.set_title("Estimated PaCO₂ — Physiological CO₂ Model", fontsize=10)

    ax_co2.set_ylabel("CO₂ (au)")
    ax_ml.set_ylabel("P(apnea)")
    ax_phys.set_ylabel("PaCO₂ (mmHg)")
    ax_phys.set_xlabel("Time (s)")

    if show_gui:
        plt.ion()
        plt.show()

    # ---- Stream simulation ----
    for w in range(n_windows):
        t_start = w * 10.0
        t_end   = t_start + 10.0
        s       = w * WINDOW_SAMPLES
        e       = s + WINDOW_SAMPLES

        win = signals[:, s:e]                     # (2, 3000)
        gt  = 1 if apnea_mask[s:e].all() else 0   # ground-truth

        # Pre-process
        win_proc = preprocess_window(win)          # (2, 3000)

        # ML inference
        if use_model:
            x_tensor = torch.from_numpy(win_proc).unsqueeze(0).to(device)   # (1,2,3000)
            with torch.no_grad():
                prob = torch.sigmoid(model(x_tensor)).item()
        else:
            # Demo: higher prob if CO₂ is flat (std < threshold)
            co2_std = float(np.std(co2_raw[s:e]))
            prob = float(np.clip(1.0 - co2_std / (np.std(co2_raw) + 1e-6), 0, 1))

        ml_detected = prob >= threshold

        # Physiology update
        alert = tracker.update(ml_detected)
        rise  = tracker.get_co2_rise()
        paco2 = tracker.get_paco2()

        # Store
        timestamps.append(t_start)
        ml_probs.append(prob)
        co2_rises.append(rise)
        alerts.append(int(alert))
        gt_labels.append(gt)

        # ---- Print table row ----
        alert_str = "ALERT" if alert else "     "
        gt_str    = "APNEA" if gt else "normal"
        print(
            f"  t={t_start:>6.0f}s  ml_prob={prob:.3f}  ΔCO₂={rise:>5.2f} mmHg"
            f"  [{alert_str}]  gt={gt_str}"
        )

        # ---- Update plots every window ----
        w_times = np.array(timestamps)
        w_probs = np.array(ml_probs)
        w_rises = np.array(co2_rises)
        paco2_arr = w_rises + INITIAL_PACO2

        # Panel 1: CO₂ waveform up to current point
        ax_co2.cla()
        ax_co2.set_title("CO₂ Waveform  (red shading = ground-truth apnea)", fontsize=10)
        ax_co2.set_ylabel("CO₂ (au)")
        cur_end_sample = min(e, n_samples)
        ax_co2.plot(time_axis[:cur_end_sample], co2_raw[:cur_end_sample],
                    color="#2196F3", linewidth=0.6, alpha=0.9)
        # Ground-truth apnea shading
        in_apnea = False
        ap_start = 0
        for i in range(cur_end_sample):
            if apnea_mask[i] and not in_apnea:
                ap_start = time_axis[i]
                in_apnea = True
            elif not apnea_mask[i] and in_apnea:
                ax_co2.axvspan(ap_start, time_axis[i], alpha=0.25, color="red")
                in_apnea = False
        if in_apnea:
            ax_co2.axvspan(ap_start, time_axis[cur_end_sample - 1], alpha=0.25, color="red")
        # Highlight current window
        ax_co2.axvspan(t_start, t_end, alpha=0.15, color="yellow")
        ax_co2.set_xlim(0, total_sec)

        # Panel 2: ML probabilities
        ax_ml.cla()
        ax_ml.set_title("ML Apnea Probability per 10 s Window", fontsize=10)
        ax_ml.set_ylabel("P(apnea)")
        bar_colors = [
            "#e53935" if p >= threshold else "#43a047" for p in w_probs
        ]
        ax_ml.bar(w_times, w_probs, width=9.5, color=bar_colors, align="edge", alpha=0.8)
        ax_ml.axhline(threshold, color="black", linestyle="--", linewidth=1, label=f"Threshold={threshold}")
        ax_ml.set_ylim(0, 1.05)
        ax_ml.set_xlim(0, total_sec)
        ax_ml.legend(fontsize=8, loc="upper right")

        # Mark alerts
        alert_mask = np.array(alerts, dtype=bool)
        if alert_mask.any():
            ax_ml.scatter(
                w_times[alert_mask] + 5,
                np.ones(alert_mask.sum()) * 1.02,
                marker="v", color="darkred", s=30, zorder=5, label="ALERT"
            )

        # Panel 3: PaCO₂
        ax_phys.cla()
        ax_phys.set_title("Estimated PaCO₂ — Physiological CO₂ Model", fontsize=10)
        ax_phys.set_ylabel("PaCO₂ (mmHg)")
        ax_phys.set_xlabel("Time (s)")
        ax_phys.plot(w_times + 5, paco2_arr, color="#ff6f00", linewidth=1.8, marker="o",
                     markersize=3, label="Est. PaCO₂")
        ax_phys.axhline(INITIAL_PACO2 + ALERT_THRESHOLD, color="red", linestyle="--",
                         linewidth=1, label=f"Alert ΔCO₂ = {ALERT_THRESHOLD} mmHg")
        ax_phys.axhline(INITIAL_PACO2, color="gray", linestyle=":", linewidth=1,
                         label=f"Baseline = {INITIAL_PACO2} mmHg")
        ax_phys.set_ylim(38, max(60, paco2_arr.max() + 5))
        ax_phys.set_xlim(0, total_sec)
        ax_phys.legend(fontsize=8, loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(hspace=0.45)

        if show_gui:
            plt.pause(0.01)

    # ---- Final summary ----
    results_df = pd.DataFrame({
        "timestamp_s": timestamps,
        "ml_prob":     ml_probs,
        "co2_rise_mmhg": co2_rises,
        "alert_fired": alerts,
        "gt_label":    gt_labels,
    })

    # Compute detection stats vs ground truth
    n_apnea_gt = sum(gt_labels)
    n_alerts   = sum(alerts)
    if n_apnea_gt > 0:
        tp = sum(a == 1 and g == 1 for a, g in zip(alerts, gt_labels))
        fn = sum(a == 0 and g == 1 for a, g in zip(alerts, gt_labels))
        fp = sum(a == 1 and g == 0 for a, g in zip(alerts, gt_labels))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        print(f"\nFusion Sensitivity (recall): {sensitivity:.3f}")
        print(f"Fusion Precision           : {precision:.3f}")
        print(f"True Apnea windows         : {n_apnea_gt}")
        print(f"Alerts fired               : {n_alerts}")

    # Save plot
    fig_path = output_dir / f"{record_id}_inference.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {fig_path}")

    # Save CSV
    csv_path = output_dir / f"{record_id}_inference.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results CSV  → {csv_path}")

    if show_gui:
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Real-time apnea inference simulation")
    p.add_argument("--data_dir",    type=str, required=True,       help="Root data directory")
    p.add_argument("--record_id",   type=str, default="0104_8min", help="CapnoBase record stem")
    p.add_argument("--model_path",  type=str, default=None,        help="Path to best_model.pt")
    p.add_argument("--threshold",   type=float, default=0.5,       help="ML decision threshold")
    p.add_argument("--output_dir",  type=str, default="output/")
    p.add_argument("--no_gui",      action="store_true",            help="Save PNG only, no display")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        record_id=args.record_id,
        data_dir=args.data_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        output_dir=args.output_dir,
        show_gui=not args.no_gui,
    )
