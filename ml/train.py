"""
train.py
--------
Training pipeline for the CNN-LSTM apnea detector.

Usage
~~~~~
    python train.py --data_dir data/ --epochs 50 --batch_size 64

Target metrics: Recall > 95 %, Accuracy > 93 % on held-out test set.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_loader import get_dataloaders
from model import ApneaCNN_LSTM, model_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics


def _eval_epoch(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= 0.5).astype(int)
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)
    yt = np.array(all_labels)
    yp = np.array(all_preds)
    yprob = np.array(all_probs)
    return _compute_metrics(yt, yp, yprob), yt, yp, yprob


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.15,
        test_split=0.15,
        use_physionet=args.use_physionet,
        physionet_dir=args.physionet_dir or None,
        use_ucddb=args.use_ucddb,
        ucddb_dir=args.ucddb_dir or None,
        ucddb_auto_download=False,
        use_synthetic=not args.no_synthetic,
        n_synthetic_normal=args.n_synthetic,
        n_synthetic_apnea=args.n_synthetic,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ApneaCNN_LSTM(
        in_channels=2,
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.3,
    ).to(device)
    model_summary(model)

    # ------------------------------------------------------------------
    # Loss — BCEWithLogitsLoss with positive class weight for imbalance
    # ------------------------------------------------------------------
    pos_weight = class_weights[1].to(device)   # w_apnea
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ------------------------------------------------------------------
    # Early stopping (on validation F1 — balances precision + recall)
    # ------------------------------------------------------------------
    best_val_f1      = -1.0
    patience_counter = 0
    best_ckpt_path   = output_dir / "best_model.pt"

    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'AUROC':>7}  {'LR':>10}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for X, y in train_loader:
            X, y = X.to(device), y.to(device).float()
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y)

        epoch_loss /= len(train_loader.dataset)
        scheduler.step()

        # ---- Validate ----
        val_metrics, _, _, _ = _eval_epoch(model, val_loader, device)
        val_f1 = val_metrics["f1"]

        print(
            f"{epoch:>6}  {epoch_loss:>8.4f}  "
            f"{val_metrics['accuracy']:>7.4f}  "
            f"{val_metrics['precision']:>7.4f}  "
            f"{val_metrics['recall']:>7.4f}  "
            f"{val_metrics['f1']:>7.4f}  "
            f"{val_metrics['auroc']:>7.4f}  "
            f"{scheduler.get_last_lr()[0]:>10.2e}"
        )

        # ---- Checkpoint ----
        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_metrics": val_metrics,
                    "args":        vars(args),
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}).")
                break

    # ------------------------------------------------------------------
    # Test evaluation (load best checkpoint)
    # ------------------------------------------------------------------
    print(f"\nLoading best checkpoint (val F1={best_val_f1:.4f}) from {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_metrics, y_true, y_pred, y_prob = _eval_epoch(model, test_loader, device)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}  (target > 0.95)")
    print(f"  F1        : {test_metrics['f1']:.4f}")
    print(f"  AUROC     : {test_metrics['auroc']:.4f}")

    if test_metrics["accuracy"] >= 0.93:
        print("  ✓ Accuracy target met (>= 93 %)")
    else:
        print("  ✗ Accuracy below 93 % target")

    if test_metrics["recall"] >= 0.95:
        print("  ✓ Recall target met (>= 95 %)")
    else:
        print("  ✗ Recall below 95 % target — consider lowering threshold in inference.py")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Apnea"], digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"             Pred Normal  Pred Apnea")
    print(f"  True Normal   {cm[0,0]:>7}     {cm[0,1]:>7}")
    print(f"  True Apnea    {cm[1,0]:>7}     {cm[1,1]:>7}")

    # Save final metrics
    metrics_path = output_dir / "test_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("TEST SET METRICS\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
    print(f"\nMetrics saved to {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train CNN-LSTM apnea detector")
    p.add_argument("--data_dir",      type=str, default="data/",   help="Root data directory")
    p.add_argument("--output_dir",    type=str, default="output/",  help="Where to save checkpoints")
    p.add_argument("--epochs",        type=int, default=50)
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--patience",      type=int, default=10,         help="Early stopping patience")
    p.add_argument("--num_workers",   type=int, default=0)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--no_physionet",  action="store_true",    help="Skip Apnea-ECG data")
    p.add_argument("--physionet_dir", type=str, default="",  help="Local Apnea-ECG dir")
    p.add_argument("--no_ucddb",      action="store_true",    help="Skip UCDDB data")
    p.add_argument("--ucddb_dir",     type=str, default="",  help="Local UCDDB dir")
    p.add_argument("--no_synthetic",  action="store_true",    help="Skip synthetic data")
    p.add_argument("--n_synthetic",   type=int, default=3000, help="Synthetic windows per class")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.use_physionet = not args.no_physionet
    args.use_ucddb     = not args.no_ucddb
    # Disable PhysioNet downloads by default (use --no_physionet already handles this)
    if not args.physionet_dir:
        args.use_physionet = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)
