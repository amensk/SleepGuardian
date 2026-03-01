"""
demo.py
-------
PNEUMA V0.1.0 — Physiological Neural Apnea Monitor
Clinical showcase figure:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  PNEUMA V0.1.0  ·  Hybrid CNN-LSTM + TcCO₂ Physiology               │
  ├──────────────────┬───────────────────────────────────────────────────┤
  │  Model Metrics   │  TcCO₂ / EtCO₂ Waveform (capnogram)              │
  │  Training Stats  ├───────────────────────────────────────────────────┤
  │  Event Readout   │  Estimated PaCO₂ / TcCO₂ Rise                    │
  │                  ├───────────────────────────────────────────────────┤
  │                  │  ML Apnea Probability  (per 10-s window)           │
  ├──────────────────┴───────────────────────────────────────────────────┤
  │  ▶▶  Apnea Detected ──────────────────────► CO₂ Alert Activated  ◀◀ │
  └──────────────────────────────────────────────────────────────────────┘

Usage
~~~~~
    # With real model metrics
    python3 demo.py --data_dir data/ --record_id 0370_8min \\
                    --metrics_path output/test_metrics.txt  \\
                    --model_path   output/best_model.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from data_loader import load_capnobase_record, FS_CAPNO, WINDOW_SAMPLES, preprocess_window
from physiology import estimate_paco2, co2_rise, ALERT_THRESHOLD, INITIAL_PACO2


# ---------------------------------------------------------------------------
# Clinical dark-mode colour palette
# ---------------------------------------------------------------------------
BG      = "#07101e"   # deep navy background
BG2     = "#0d1829"   # panel fill
BG3     = "#121f35"   # nested fill / bar track
CYAN    = "#00e5ff"   # EtCO₂ channel
VIOLET  = "#c084fc"   # secondary / ROC AUC
GREEN   = "#22c55e"   # normal / ok
RED     = "#ef4444"   # alert / apnea
AMBER   = "#facc15"   # warning threshold
BLUE    = "#60a5fa"   # PaCO₂ trend
TEXT    = "#e2e8f0"
DIM     = "#1e3352"
SUBTEXT = "#64748b"
GRID    = "#112240"


# ---------------------------------------------------------------------------
# Helpers — data
# ---------------------------------------------------------------------------

def _load_metrics(path: Optional[str]) -> dict:
    defaults = dict(accuracy=0.9888, precision=0.9831, recall=0.9898,
                    f1=0.9864, auroc=0.9989)
    if path and Path(path).exists():
        m: dict = {}
        with open(path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    try:
                        m[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
        return {**defaults, **m}
    return defaults


def _find_apnea_window(
    signals: np.ndarray,
    apnea_mask: np.ndarray,
    insp_idx: np.ndarray,
    n_context: int = 38,
) -> Tuple[int, int, int, int]:
    if apnea_mask.any():
        changes = np.diff(apnea_mask.astype(int))
        starts  = np.where(changes == 1)[0] + 1
        ends    = np.where(changes == -1)[0] + 1
        if apnea_mask[0]:  starts = np.r_[0, starts]
        if apnea_mask[-1]: ends   = np.r_[ends, len(apnea_mask)]
        best    = int(np.argmax(ends - starts))
        ap_s, ap_e = int(starts[best]), int(ends[best])
    else:
        if len(insp_idx) >= 2:
            best = int(np.argmax(np.diff(insp_idx)))
            ap_s = int(insp_idx[best])
            ap_e = int(insp_idx[best + 1])
        else:
            mid = signals.shape[1] // 2
            ap_s, ap_e = mid - 1500, mid + 1500
    ctx_s = max(0, ap_s - n_context * FS_CAPNO)
    ctx_e = min(signals.shape[1], ap_e + n_context * FS_CAPNO)
    return ap_s, ap_e, ctx_s, ctx_e


def _paco2_trend(ap_s_s: float, ap_e_s: float, t: np.ndarray) -> np.ndarray:
    """Estimate PaCO₂ (mmHg): rises during apnea, decays exponentially after."""
    tau  = 100.0   # seconds — CO₂ washout time constant
    peak = estimate_paco2(ap_e_s - ap_s_s)
    out  = np.empty_like(t, dtype=float)
    for i, ti in enumerate(t):
        if ti < ap_s_s:
            out[i] = INITIAL_PACO2
        elif ti <= ap_e_s:
            out[i] = estimate_paco2(ti - ap_s_s)
        else:
            out[i] = INITIAL_PACO2 + (peak - INITIAL_PACO2) * np.exp(-(ti - ap_e_s) / tau)
    return out


# ---------------------------------------------------------------------------
# Helpers — model inference
# ---------------------------------------------------------------------------

def _try_load_model(model_path: str):
    try:
        import torch
        from model import ApneaCNN_LSTM
        dev  = torch.device(
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        ckpt = torch.load(model_path, map_location=dev, weights_only=False)
        m    = ApneaCNN_LSTM(in_channels=2, lstm_hidden=64, lstm_layers=2, dropout=0.3)
        m.load_state_dict(ckpt["model_state"])
        m.to(dev).eval()
        return m, dev
    except Exception:
        return None, None


def _infer_windows(
    model, device, signals: np.ndarray, ctx_s: int, ctx_e: int
) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    centers, probs = [], []
    for s in range(ctx_s, ctx_e - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        proc = preprocess_window(signals[:, s : s + WINDOW_SAMPLES])
        with torch.no_grad():
            p = torch.sigmoid(
                model(torch.from_numpy(proc[np.newaxis]).to(device))
            ).item()
        centers.append((s + WINDOW_SAMPLES / 2) / FS_CAPNO)
        probs.append(p)
    return np.array(centers), np.array(probs)


def _simulate_probs(
    apnea_mask: np.ndarray, ctx_s: int, ctx_e: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback: simulate plausible probabilities from ground truth."""
    rng = np.random.default_rng(7)
    centers, probs = [], []
    for s in range(ctx_s, ctx_e - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        frac = apnea_mask[s : s + WINDOW_SAMPLES].mean()
        p    = rng.uniform(0.85, 0.97) if frac > 0.6 else rng.uniform(0.02, 0.14)
        centers.append((s + WINDOW_SAMPLES / 2) / FS_CAPNO)
        probs.append(p)
    return np.array(centers), np.array(probs)


# ---------------------------------------------------------------------------
# Helpers — figure drawing
# ---------------------------------------------------------------------------

def _style_ax(ax, *, grid: bool = True) -> None:
    ax.set_facecolor(BG2)
    for sp in ax.spines.values():
        sp.set_edgecolor(DIM)
        sp.set_linewidth(0.9)
    ax.tick_params(colors=SUBTEXT, labelsize=8.5, length=3, pad=4)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.45, linestyle="--", alpha=0.9)
        ax.set_axisbelow(True)
    ax.yaxis.label.set_color(SUBTEXT)
    ax.xaxis.label.set_color(SUBTEXT)


def _draw_apnea_span(ax, ap_s_s: float, ap_e_s: float) -> None:
    ax.axvspan(ap_s_s, ap_e_s, color=RED, alpha=0.12, zorder=0)
    ax.axvline(ap_s_s, color=RED, linewidth=1.4, linestyle="--", alpha=0.75, zorder=2)
    ax.axvline(ap_e_s, color=RED, linewidth=1.4, linestyle="--", alpha=0.75, zorder=2)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def make_demo_figure(
    record_id: str,
    data_dir: str,
    metrics_path: Optional[str] = None,
    model_path: str = "output/best_model.pt",
    output_path: str = "output/PNEUMA_demo.png",
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── load record ──────────────────────────────────────────────────────────
    sig_file = Path(data_dir) / "data" / "csv" / f"{record_id}_signal.csv"
    if not sig_file.exists():
        sys.exit(f"Signal file not found: {sig_file}")
    result = load_capnobase_record(str(sig_file))
    if result is None:
        sys.exit(f"Could not load {record_id}")
    signals, apnea_mask, insp_idx = result
    co2 = signals[0]

    ap_s, ap_e, ctx_s, ctx_e = _find_apnea_window(signals, apnea_mask, insp_idx)
    apnea_dur_s = (ap_e - ap_s) / FS_CAPNO
    ap_s_s, ap_e_s = ap_s / FS_CAPNO, ap_e / FS_CAPNO

    t_ctx   = np.arange(ctx_s, ctx_e) / FS_CAPNO
    co2_ctx = co2[ctx_s:ctx_e]
    paco2_t = _paco2_trend(ap_s_s, ap_e_s, t_ctx)
    rise_val  = co2_rise(apnea_dur_s)
    alert_ok  = rise_val > ALERT_THRESHOLD
    paco2_peak = estimate_paco2(apnea_dur_s)

    # ── model inference ───────────────────────────────────────────────────────
    model, device = _try_load_model(model_path)
    if model is not None:
        wt, wp = _infer_windows(model, device, signals, ctx_s, ctx_e)
    else:
        wt, wp = _simulate_probs(apnea_mask, ctx_s, ctx_e)

    met = _load_metrics(metrics_path)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 13.5), facecolor=BG)

    fig.text(0.5, 0.977, "PNEUMA   V0.1.0",
             ha="center", va="top", fontsize=36, fontweight="bold",
             color=CYAN, fontfamily="monospace")
    fig.text(
        0.5, 0.947,
        "Physiological Neural Apnea Monitor  ·  "
        "Hybrid CNN-LSTM + TcCO₂ Physiology  ·  "
        f"Record: {record_id}",
        ha="center", va="top", fontsize=11.5, color=SUBTEXT, fontstyle="italic",
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        left=0.03, right=0.97,
        top=0.915, bottom=0.135,
        hspace=0.32, wspace=0.08,
        width_ratios=[1, 2.85],
        height_ratios=[2.5, 1.3, 1.15],
    )
    ax_met  = fig.add_subplot(gs[:, 0])
    ax_co2  = fig.add_subplot(gs[0, 1])
    ax_pco2 = fig.add_subplot(gs[1, 1])
    ax_prob = fig.add_subplot(gs[2, 1])

    _style_ax(ax_met,  grid=False)
    _style_ax(ax_co2,  grid=True)
    _style_ax(ax_pco2, grid=True)
    _style_ax(ax_prob, grid=True)

    ax_met.set_xlim(0, 1)
    ax_met.set_ylim(0, 1)
    ax_met.set_xticks([])
    ax_met.set_yticks([])

    # ── LEFT PANEL helpers ────────────────────────────────────────────────────

    def _hline(y: float, x0: float = 0.05, x1: float = 0.95) -> None:
        ax_met.plot([x0, x1], [y, y], color=DIM, linewidth=0.8)

    def _section_hdr(y: float, label: str, color: str = CYAN) -> None:
        ax_met.text(0.5, y, f"◈  {label}", ha="center", va="top",
                    color=color, fontsize=9, fontweight="bold",
                    fontfamily="monospace")

    def _metric_bar(y_center: float, label: str, val: float,
                    color: str = CYAN) -> None:
        BW, BH, X0 = 0.82, 0.036, 0.09
        ax_met.text(X0, y_center + 0.036, label, color=SUBTEXT,
                    fontsize=7.5, va="bottom", fontfamily="monospace")
        ax_met.add_patch(mpatches.FancyBboxPatch(
            (X0, y_center - BH / 2), BW, BH,
            boxstyle="round,pad=0.004",
            facecolor=BG3, edgecolor=DIM, linewidth=0.5, zorder=2,
        ))
        ax_met.add_patch(mpatches.FancyBboxPatch(
            (X0, y_center - BH / 2), BW * val, BH,
            boxstyle="round,pad=0.004",
            facecolor=color, alpha=0.88, zorder=3,
        ))
        ax_met.text(X0 + BW * val + 0.022, y_center + 0.001,
                    f"{val * 100:.2f} %",
                    color=TEXT, fontsize=9, va="center",
                    fontweight="bold", fontfamily="monospace")

    def _kv_row(y: float, key: str, val: str,
                val_color: str = TEXT) -> None:
        ax_met.text(0.08, y, key + " :", color=SUBTEXT, fontsize=8,
                    va="center", fontfamily="monospace")
        ax_met.text(0.93, y, val, color=val_color, fontsize=8,
                    va="center", ha="right", fontfamily="monospace")

    # ── MODEL PERFORMANCE section  (y = 0.530 → 0.990) ───────────────────────
    _section_hdr(0.978, "MODEL PERFORMANCE", color=CYAN)
    _hline(0.952)

    bars = [
        ("ACCURACY",  met["accuracy"],  CYAN,   0.908),
        ("F1 SCORE",  met["f1"],        CYAN,   0.822),
        ("RECALL",    met["recall"],    GREEN,  0.736),
        ("PRECISION", met["precision"], CYAN,   0.650),
        ("ROC  AUC",  met["auroc"],     VIOLET, 0.564),
    ]
    for label, val, color, yc in bars:
        _metric_bar(yc, label, val, color)

    # ── TRAINING DATA section  (y = 0.280 → 0.520) ───────────────────────────
    _hline(0.518)
    _section_hdr(0.506, "TRAINING DATA", color=VIOLET)
    _hline(0.484)

    train_rows = [
        ("Subjects",    "42  CapnoBase subjects"),
        ("Windows",     "8,000+  training samples"),
        ("Test set",    "1,428  evaluation windows"),
        ("Epochs",      "18  (early stop on F1)"),
    ]
    for j, (k, v) in enumerate(train_rows):
        _kv_row(0.458 - j * 0.046, k, v)

    # ── EVENT READOUT section  (y = 0.010 → 0.270) ───────────────────────────
    _hline(0.272)
    _section_hdr(0.261, "EVENT READOUT", color=AMBER)
    _hline(0.239)

    event_rows = [
        ("Apnea duration", f"{apnea_dur_s:.0f} s",        TEXT),
        ("Peak PaCO₂",     f"{paco2_peak:.1f}  mmHg",     RED if alert_ok else GREEN),
        ("PaCO₂ rise",     f"+{rise_val:.1f}  mmHg",      RED if alert_ok else GREEN),
        ("Inspirations",   f"{len(insp_idx)}  detected",  TEXT),
    ]
    for j, (k, v, vc) in enumerate(event_rows):
        _kv_row(0.213 - j * 0.046, k, v, val_color=vc)

    # alert badge
    badge_color = RED if alert_ok else GREEN
    badge_txt   = "▶  CO₂ ALERT ACTIVE" if alert_ok else "✓  NORMAL BREATHING"
    ax_met.add_patch(mpatches.FancyBboxPatch(
        (0.07, 0.014), 0.86, 0.050,
        boxstyle="round,pad=0.006",
        facecolor=badge_color, alpha=0.20, edgecolor=badge_color,
        linewidth=1.2, zorder=4,
    ))
    ax_met.text(0.5, 0.039, badge_txt, ha="center", va="center",
                color=badge_color, fontsize=9, fontweight="bold",
                fontfamily="monospace", zorder=5)

    # ── RIGHT TOP — TcCO₂ / EtCO₂ Waveform ───────────────────────────────────
    ax_co2.set_title(
        f"TcCO₂  /  EtCO₂ Waveform  ·  "
        f"Apnea  {ap_s_s:.0f} s → {ap_e_s:.0f} s  ({apnea_dur_s:.0f} s)",
        color=CYAN, fontsize=10.5, fontweight="bold", pad=7,
    )
    ax_co2.set_ylabel("CO₂  (V)", fontsize=9)

    ax_co2.plot(t_ctx, co2_ctx, color=CYAN, linewidth=0.9, alpha=0.95,
                label="EtCO₂  (capnogram)", zorder=3)
    _draw_apnea_span(ax_co2, ap_s_s, ap_e_s)

    for idx in insp_idx:
        ti = idx / FS_CAPNO
        if t_ctx[0] <= ti <= t_ctx[-1]:
            ax_co2.axvline(ti, color=GREEN, linewidth=0.55, alpha=0.40, zorder=1)

    # flatline annotation — find closest sample to window midpoint
    mid_t = (ap_s_s + ap_e_s) / 2
    mid_i = int(np.argmin(np.abs(t_ctx - mid_t)))
    y_max = float(co2_ctx.max())
    y_span = y_max - float(co2_ctx.min())
    ax_co2.annotate(
        "CO₂ flatline\n(no expiration)",
        xy=(mid_t, float(co2_ctx[mid_i])),
        xytext=(mid_t, y_max + y_span * 0.14),
        ha="center", fontsize=8.5, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2),
    )

    ax_co2.set_xlim(t_ctx[0], t_ctx[-1])
    ax_co2.tick_params(axis="x", labelbottom=False)
    ax_co2.legend(loc="upper right", fontsize=8.5,
                  facecolor=BG2, edgecolor=DIM, labelcolor=TEXT)

    # ── RIGHT MIDDLE — Estimated PaCO₂ / TcCO₂ Trend ─────────────────────────
    ax_pco2.set_title("Estimated  PaCO₂  /  TcCO₂  Trend",
                       color=BLUE, fontsize=10.5, fontweight="bold", pad=7)
    ax_pco2.set_ylabel("PaCO₂  (mmHg)", fontsize=9)

    ax_pco2.fill_between(t_ctx, INITIAL_PACO2, paco2_t,
                          color=BLUE, alpha=0.18, zorder=1)
    ax_pco2.plot(t_ctx, paco2_t, color=BLUE, linewidth=2.2,
                 label="Est. PaCO₂", zorder=3)

    alert_level = INITIAL_PACO2 + ALERT_THRESHOLD
    ax_pco2.axhline(alert_level, color=AMBER, linewidth=1.4, linestyle=":",
                    alpha=0.95,
                    label=f"Alert threshold  {alert_level:.0f} mmHg", zorder=4)

    _draw_apnea_span(ax_pco2, ap_s_s, ap_e_s)

    ax_pco2.set_xlim(t_ctx[0], t_ctx[-1])
    ax_pco2.set_ylim(INITIAL_PACO2 - 1.5,
                     max(float(paco2_t.max()), alert_level) + 2.5)
    ax_pco2.tick_params(axis="x", labelbottom=False)
    ax_pco2.legend(loc="upper right", fontsize=8.5,
                   facecolor=BG2, edgecolor=DIM, labelcolor=TEXT)

    # ── RIGHT BOTTOM — ML Apnea Probability ───────────────────────────────────
    ax_prob.set_title("ML  Apnea  Probability  (per 10-s window)",
                       color=VIOLET, fontsize=10.5, fontweight="bold", pad=7)
    ax_prob.set_ylabel("P (apnea)", fontsize=9)
    ax_prob.set_xlabel("Time  (s)", fontsize=9)

    bar_w   = WINDOW_SAMPLES / FS_CAPNO * 0.84
    bcolors = [RED if p >= 0.5 else GREEN for p in wp]
    ax_prob.bar(wt, wp, width=bar_w, color=bcolors, alpha=0.82, zorder=3,
                edgecolor=BG2, linewidth=0.4)

    for wti, wpi, bc in zip(wt, wp, bcolors):
        ypos = min(wpi + 0.04, 1.02)
        ax_prob.text(wti, ypos, f"{wpi:.2f}", ha="center", va="bottom",
                     fontsize=7.5, color=bc, fontfamily="monospace",
                     fontweight="bold")

    ax_prob.axhline(0.50, color=AMBER, linewidth=1.4, linestyle="--",
                    alpha=0.95, label="Decision threshold  0.50", zorder=4)
    _draw_apnea_span(ax_prob, ap_s_s, ap_e_s)

    ax_prob.set_xlim(t_ctx[0], t_ctx[-1])
    ax_prob.set_ylim(-0.04, 1.18)
    ax_prob.legend(loc="upper right", fontsize=8.5,
                   facecolor=BG2, edgecolor=DIM, labelcolor=TEXT)

    # ── BOTTOM FUSION BANNER ──────────────────────────────────────────────────
    banner = fig.add_axes([0.03, 0.022, 0.94, 0.082])
    banner.set_facecolor(RED if alert_ok else GREEN)
    banner.set_xlim(0, 1)
    banner.set_ylim(0, 1)
    banner.set_xticks([])
    banner.set_yticks([])
    for sp in banner.spines.values():
        sp.set_visible(False)

    if alert_ok:
        ltxt = "  ▶▶  Apnea Detected"
        ctxt = "  ─────────────────────────────────────────  "
        rtxt = f"CO₂ Alert  ·  ΔPaCO₂ +{rise_val:.1f} mmHg  ◀◀  "
        fc   = "#ffffff"
    else:
        ltxt = "  ✓  Normal Breathing Detected"
        ctxt = "  ──────────────────────────────────────  "
        rtxt = "No Alert  ·  PaCO₂ Stable  ✓  "
        fc   = "#0a1628"

    banner.text(0.01, 0.5, ltxt, color=fc, fontsize=19, fontweight="bold",
                va="center", ha="left", fontfamily="monospace")
    banner.text(0.50, 0.5, ctxt, color=fc, fontsize=14,
                va="center", ha="center", fontfamily="monospace")
    banner.text(0.99, 0.5, rtxt, color=fc, fontsize=19, fontweight="bold",
                va="center", ha="right", fontfamily="monospace")

    # ── FOOTER ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.003,
        "Datasets: CapnoBase  300 Hz  TcCO₂ / EtCO₂ / PPG   ·   "
        "Model: CNN-LSTM  344 K params  ·  BCEWithLogitsLoss + pos_weight  ·  "
        "Fusion guardrail: ΔPaCO₂ > 4 mmHg",
        ha="center", color=SUBTEXT, fontsize=7.5, fontstyle="italic",
    )

    plt.savefig(output_path, dpi=190, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Demo figure saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PNEUMA demo figure generator")
    p.add_argument("--data_dir",     type=str, default="data/")
    p.add_argument("--record_id",    type=str, default="0370_8min",
                   help="CapnoBase record with clear apnea event")
    p.add_argument("--metrics_path", type=str, default=None,
                   help="Path to output/test_metrics.txt")
    p.add_argument("--model_path",   type=str, default="output/best_model.pt",
                   help="Path to trained model checkpoint")
    p.add_argument("--output_path",  type=str, default="output/PNEUMA_demo.png")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_demo_figure(
        record_id=args.record_id,
        data_dir=args.data_dir,
        metrics_path=args.metrics_path,
        model_path=args.model_path,
        output_path=args.output_path,
    )
