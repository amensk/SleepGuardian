"""
data_loader.py
--------------
Loads and preprocesses signals from three sources:

  1. CapnoBase IEEE TBME RR Benchmark  (local CSV files, 300 Hz)
  2. PhysioNet Apnea-ECG Database      (wfdb download, 8 records with resp, 100→300 Hz)
  3. PhysioNet UCDDB                   (EDF download, 25 records, airflow 32→300 Hz)

All sources are windowed into 10-second, non-overlapping segments, band-pass filtered
(0.05–0.7 Hz) and Z-score normalised before being combined into a PyTorch Dataset.

CapnoBase directory layout (after unzipping the user-provided archive):
  <data_dir>/data/csv/<record>_8min_signal.csv
  <data_dir>/data/csv/<record>_8min_labels.csv

PhysioNet data is downloaded automatically by wfdb on first use.
"""

import os
import warnings
from math import gcd
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt, resample_poly
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS_CAPNO        = 300        # CapnoBase / target sampling rate (Hz)
WINDOW_SEC      = 10         # window length (s)
WINDOW_SAMPLES  = FS_CAPNO * WINDOW_SEC   # 3 000 samples

APNEA_GAP_SAMPLES = WINDOW_SAMPLES       # gap ≥ 3 000 samples → apnea region

BP_LOW   = 0.05              # bandpass lower cutoff (Hz)
BP_HIGH  = 0.70              # bandpass upper cutoff (Hz)
BP_ORDER = 4

# Apnea-ECG records that have supplementary respiration channels
# Use the 'er' (combined) header; includes ECG + Resp C / A / N + SpO2
APNEA_ECG_RESP_RECORDS = ["a01er", "a02er", "a03er", "a04er",
                           "b01er", "c01er", "c02er", "c03er"]
# ECG-only records (no resp channel)
APNEA_ECG_ECG_RECORDS  = (
    [f"a{i:02d}" for i in range(5, 21)]   # a05-a20
    + [f"b{i:02d}" for i in range(2, 6)]  # b02-b05
    + [f"c{i:02d}" for i in range(4, 11)] # c04-c10
)

# UCDDB record names (25 subjects; IDs 001, 004, 016 do not exist)
UCDDB_RECORDS = (
    ["ucddb002", "ucddb003"]
    + [f"ucddb{i:03d}" for i in range(5, 16)]   # 005–015
    + [f"ucddb{i:03d}" for i in range(17, 29)]  # 017–028
)

# UCDDB apnea/hypopnea event types to treat as positive label
UCDDB_APNEA_TYPES = {"APNEA-C", "APNEA-O", "APNEA-M",
                     "HYP-C",   "HYP-O",   "HYP-M"}


# ---------------------------------------------------------------------------
# Signal preprocessing helpers
# ---------------------------------------------------------------------------

def _bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass using SOS format for numerical stability.

    The standard ba-format butter() is numerically catastrophic at high sampling
    rates (300 Hz) with narrow physiological cutoffs (0.05–0.7 Hz) because the
    8th-order bandpass polynomial coefficients suffer from catastrophic cancellation.
    SOS format avoids this by keeping biquad sections in second-order form.
    """
    nyq  = fs / 2.0
    low  = max(BP_LOW  / nyq, 1e-4)
    high = min(BP_HIGH / nyq, 1 - 1e-4)
    if low >= high:
        return signal.copy()
    if len(signal) < 3 * BP_ORDER:
        return signal.copy()
    sos = butter(BP_ORDER, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def _zscore(signal: np.ndarray) -> np.ndarray:
    mu, std = signal.mean(), signal.std()
    return np.zeros_like(signal) if std < 1e-8 else (signal - mu) / std


def preprocess_window(window: np.ndarray, fs: float = FS_CAPNO) -> np.ndarray:
    """
    Apply bandpass filter + Z-score to each channel of a window.
    window shape: (C, T) → returns (C, T) float32.
    """
    out = np.empty_like(window, dtype=np.float32)
    for c in range(window.shape[0]):
        filtered = _bandpass_filter(window[c].astype(np.float64), fs)
        out[c]   = _zscore(filtered).astype(np.float32)
    return out


def _resample_to_capno(signal: np.ndarray, fs_native: float) -> np.ndarray:
    """Resample signal from fs_native Hz to FS_CAPNO Hz using polyphase filter."""
    if abs(fs_native - FS_CAPNO) < 0.5:
        return signal.astype(np.float32)
    up   = int(FS_CAPNO)
    down = int(fs_native)
    g    = gcd(up, down)
    return resample_poly(signal, up // g, down // g).astype(np.float32)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def make_windows(
    signals: np.ndarray,
    apnea_mask: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create non-overlapping windows.

    Args:
        signals    : (C, N) float32
        apnea_mask : (N,)   bool — True = apnea sample
        window_samples : samples per window

    Returns:
        X : (W, C, window_samples) float32 — preprocessed
        y : (W,)                  int32
    """
    C, N = signals.shape
    n_win = N // window_samples
    X_list, y_list = [], []

    for i in range(n_win):
        s, e = i * window_samples, (i + 1) * window_samples
        win  = signals[:, s:e]
        X_list.append(preprocess_window(win))
        # Apnea label: the entire window must be within an apnea region
        y_list.append(1 if apnea_mask[s:e].all() else 0)

    if not X_list:
        return (np.empty((0, C, window_samples), dtype=np.float32),
                np.array([], dtype=np.int32))
    return np.stack(X_list), np.array(y_list, dtype=np.int32)


# ---------------------------------------------------------------------------
# ──────────────────────────  1. CapnoBase  ──────────────────────────────────
# ---------------------------------------------------------------------------

def _parse_space_sep_column(raw_value) -> np.ndarray:
    """Parse a space-separated integer list stored in one CSV cell."""
    if pd.isna(raw_value):
        return np.array([], dtype=np.int64)
    s = str(raw_value).strip().strip('"').strip()
    if not s:
        return np.array([], dtype=np.int64)
    try:
        return np.array([int(p) for p in s.split()], dtype=np.int64)
    except ValueError:
        return np.array([], dtype=np.int64)


def _apnea_mask_from_insp_indices(insp_indices: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Boolean mask where inter-breath gaps ≥ APNEA_GAP_SAMPLES are marked True.
    """
    mask = np.zeros(n_samples, dtype=bool)
    if len(insp_indices) < 2:
        return mask
    for i in range(len(insp_indices) - 1):
        start, end = int(insp_indices[i]), int(insp_indices[i + 1])
        if (end - start) >= APNEA_GAP_SAMPLES:
            mask[min(start, n_samples):min(end, n_samples)] = True
    last = int(insp_indices[-1])
    if (n_samples - last) >= APNEA_GAP_SAMPLES:
        mask[last:] = True
    return mask


def load_capnobase_record(
    signal_csv: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load one CapnoBase record.

    Returns:
        signals    : (2, N) float32  — [co2_y, pleth_y]
        apnea_mask : (N,)  bool
        insp_idx   : (M,)  int64     — breath onset sample indices (for overlay)
    """
    signal_path = Path(signal_csv)
    stem        = signal_path.name.replace("_signal.csv", "")
    labels_path = signal_path.parent / f"{stem}_labels.csv"

    try:
        sig_df = pd.read_csv(signal_path)
    except Exception as e:
        warnings.warn(f"Cannot read {signal_path}: {e}")
        return None

    if not {"co2_y", "pleth_y"}.issubset(sig_df.columns):
        warnings.warn(f"Missing columns in {signal_path}")
        return None

    co2   = sig_df["co2_y"].to_numpy(dtype=np.float32)
    pleth = sig_df["pleth_y"].to_numpy(dtype=np.float32)
    signals   = np.stack([co2, pleth])        # (2, N)
    n_samples = signals.shape[1]

    insp_idx = np.array([], dtype=np.int64)
    if labels_path.exists():
        try:
            lab_df   = pd.read_csv(labels_path, nrows=2)
            if "co2_startinsp_x" in lab_df.columns:
                raw      = lab_df["co2_startinsp_x"].iloc[
                    1 if len(lab_df) > 1 else 0
                ]
                insp_idx = _parse_space_sep_column(raw)
                insp_idx = np.sort(insp_idx[insp_idx < n_samples])
        except Exception as e:
            warnings.warn(f"Cannot parse labels for {stem}: {e}")

    apnea_mask = _apnea_mask_from_insp_indices(insp_idx, n_samples)
    return signals, apnea_mask, insp_idx


def load_capnobase_dataset(
    csv_dir: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    csv_dir      = Path(csv_dir)
    signal_files = sorted(csv_dir.glob("*_signal.csv"))
    if not signal_files:
        raise FileNotFoundError(f"No *_signal.csv files in {csv_dir}")

    all_X, all_y, record_ids = [], [], []
    for sf in signal_files:
        result = load_capnobase_record(str(sf))
        if result is None:
            continue
        signals, apnea_mask, _ = result
        X, y = make_windows(signals, apnea_mask)
        if len(X) == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        record_ids.append(sf.stem.replace("_signal", ""))

    n_total = sum(len(y) for y in all_y)
    n_apnea = int(sum(y.sum() for y in all_y))
    print(f"CapnoBase : {len(record_ids)} records | "
          f"{n_total} windows | {n_apnea} apnea ({100*n_apnea/max(n_total,1):.1f}%)")
    return all_X, all_y, record_ids


# ---------------------------------------------------------------------------
# ──────────────────────────  2. Apnea-ECG  ──────────────────────────────────
# ---------------------------------------------------------------------------

def load_apnea_ecg_record(
    record_name: str,
    physionet_dir: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load one Apnea-ECG record.

    Preference order for respiratory channel:
      1. 'Resp C' / 'Resp A' / 'Resp N'  (in `er` combined records)
      2. First ECG channel (fallback – morphology only)

    Returns (signals (1, N), apnea_mask (N,)) at FS_CAPNO Hz, or None.
    """
    try:
        import wfdb
    except ImportError:
        warnings.warn("wfdb not installed. Run: pip install wfdb")
        return None

    try:
        if physionet_dir:
            rec = wfdb.rdrecord(os.path.join(physionet_dir, record_name))
        else:
            rec = wfdb.rdrecord(record_name, pn_dir="apnea-ecg")
    except Exception as e:
        warnings.warn(f"Cannot load Apnea-ECG record '{record_name}': {e}")
        return None

    # Pick best respiratory channel
    sig_names = [s.lower() for s in rec.sig_name]
    resp_idx  = None
    for cand in ["resp c", "resp a", "resp n", "resp", "thorax", "thor"]:
        for i, name in enumerate(sig_names):
            if cand in name:
                resp_idx = i
                break
        if resp_idx is not None:
            break
    if resp_idx is None:
        resp_idx = 0   # fall back to ECG

    raw  = rec.p_signal[:, resp_idx].astype(np.float32)
    resp = _resample_to_capno(raw, rec.fs)
    n    = len(resp)
    signals = resp[np.newaxis, :]   # (1, N)

    # Apnea annotations (per-minute A/N labels)
    apnea_mask = np.zeros(n, dtype=bool)
    # Derive base record name (strip 'er' suffix)
    base = record_name.replace("er", "")
    try:
        if physionet_dir:
            ann = wfdb.rdann(os.path.join(physionet_dir, base), "apn")
        else:
            ann = wfdb.rdann(base, "apn", pn_dir="apnea-ecg")

        scale = FS_CAPNO / rec.fs
        for idx, sym in zip(ann.sample, ann.symbol):
            if sym == "A":
                s = int(idx * scale)
                e = int((idx + 60 * rec.fs) * scale)
                apnea_mask[max(0,s):min(e,n)] = True
    except Exception:
        pass

    return signals, apnea_mask


def load_apnea_ecg_dataset(
    physionet_dir: Optional[str] = None,
    use_ecg_only_records: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load Apnea-ECG records.

    By default only loads the 8 records with true respiratory channels (er records).
    Set use_ecg_only_records=True to also include the 35 ECG-only records.
    """
    records = list(APNEA_ECG_RESP_RECORDS)
    if use_ecg_only_records:
        records += APNEA_ECG_ECG_RECORDS

    all_X, all_y, record_ids = [], [], []
    for rec in records:
        result = load_apnea_ecg_record(rec, physionet_dir)
        if result is None:
            continue
        signals, apnea_mask = result
        X, y = make_windows(signals, apnea_mask)
        if len(X) == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        record_ids.append(f"apnea_ecg_{rec}")

    n_total = sum(len(y) for y in all_y)
    n_apnea = int(sum(y.sum() for y in all_y))
    print(f"Apnea-ECG : {len(record_ids)} records | "
          f"{n_total} windows | {n_apnea} apnea ({100*n_apnea/max(n_total,1):.1f}%)")
    return all_X, all_y, record_ids


# ---------------------------------------------------------------------------
# ──────────────────────────  3. UCDDB  ──────────────────────────────────────
# ---------------------------------------------------------------------------

def _parse_ucddb_respevt(
    respevt_path: str,
    n_samples: int,
    native_fs: float = 32.0,
    target_fs: float = FS_CAPNO,
) -> np.ndarray:
    """
    Parse a UCDDB *_respevt.txt file and return a boolean apnea mask
    at target_fs sampling rate.

    File format (tab-delimited):
        HH:MM:SS  <EventType>  <Duration_s>  ...
    """
    mask = np.zeros(n_samples, dtype=bool)
    if not os.path.exists(respevt_path):
        return mask

    with open(respevt_path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            time_str   = parts[0]
            event_type = parts[1].upper()
            try:
                duration_s = float(parts[2])
            except ValueError:
                continue
            if event_type not in UCDDB_APNEA_TYPES:
                continue
            try:
                h, m, s = map(int, time_str.split(":"))
                onset_s = h * 3600 + m * 60 + s
            except ValueError:
                continue
            start = int(onset_s * target_fs)
            end   = int((onset_s + duration_s) * target_fs)
            mask[max(0, start):min(end, n_samples)] = True

    return mask


def load_ucddb_record(
    rec_name: str,
    ucddb_dir: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load one UCDDB record from a local directory containing .rec and _respevt.txt files.

    Returns (signals (1, N), apnea_mask (N,)) at FS_CAPNO Hz, or None on failure.
    """
    rec_path     = os.path.join(ucddb_dir, f"{rec_name}.rec")
    respevt_path = os.path.join(ucddb_dir, f"{rec_name}_respevt.txt")

    if not os.path.exists(rec_path):
        warnings.warn(f"UCDDB file not found: {rec_path}")
        return None

    try:
        import pyedflib
    except ImportError:
        warnings.warn("pyedflib not installed. Run: pip install pyedflib")
        return None

    try:
        f       = pyedflib.EdfReader(rec_path)
        labels  = [l.strip().lower() for l in f.getSignalLabels()]
        freqs   = f.getSampleFrequencies()
    except Exception as e:
        warnings.warn(f"Cannot open {rec_path}: {e}")
        return None

    # Prefer: airflow > thorax > abdomen
    resp_ch, resp_fs = None, None
    for cand_group in [
        ["airflow", "flow", "oro-nasal", "oronasal"],
        ["thor", "ribcage", "thorax", "chest"],
        ["abdo", "abdomen", "abdominal"],
    ]:
        for i, label in enumerate(labels):
            if any(c in label for c in cand_group):
                resp_ch = i
                resp_fs = float(freqs[i])
                break
        if resp_ch is not None:
            break

    if resp_ch is None:
        warnings.warn(f"No respiratory channel found in {rec_name}; skipping.")
        f.close()
        return None

    try:
        raw = f.readSignal(resp_ch).astype(np.float32)
    except Exception as e:
        warnings.warn(f"Cannot read signal from {rec_name}: {e}")
        f.close()
        return None
    f.close()

    # Resample to FS_CAPNO
    resp     = _resample_to_capno(raw, resp_fs)
    n        = len(resp)
    signals  = resp[np.newaxis, :]   # (1, N)

    # Apnea mask from event file
    apnea_mask = _parse_ucddb_respevt(respevt_path, n, resp_fs, FS_CAPNO)
    return signals, apnea_mask


def download_ucddb(target_dir: str) -> bool:
    """
    Download UCDDB from PhysioNet using wfdb.
    Returns True on success, False on failure.
    """
    try:
        import wfdb
        os.makedirs(target_dir, exist_ok=True)
        print(f"Downloading UCDDB to {target_dir} …")
        wfdb.dl_database("ucddb", dl_dir=target_dir)
        return True
    except Exception as e:
        warnings.warn(f"UCDDB download failed: {e}")
        return False


def load_ucddb_dataset(
    ucddb_dir: str,
    records: Optional[List[str]] = None,
    auto_download: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load UCDDB records from ucddb_dir.

    If the directory is empty / missing and auto_download=True, downloads via wfdb.
    """
    ucddb_dir = str(ucddb_dir)
    if records is None:
        records = UCDDB_RECORDS

    # Download if needed
    if auto_download and not any(
        Path(ucddb_dir).glob("ucddb*.rec")
    ):
        ok = download_ucddb(ucddb_dir)
        if not ok:
            print("UCDDB download failed — skipping this source.")
            return [], [], []

    all_X, all_y, record_ids = [], [], []
    for rec_name in records:
        result = load_ucddb_record(rec_name, ucddb_dir)
        if result is None:
            continue
        signals, apnea_mask = result
        X, y = make_windows(signals, apnea_mask)
        if len(X) == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        record_ids.append(f"ucddb_{rec_name}")

    n_total = sum(len(y) for y in all_y)
    n_apnea = int(sum(y.sum() for y in all_y))
    print(f"UCDDB     : {len(record_ids)} records | "
          f"{n_total} windows | {n_apnea} apnea ({100*n_apnea/max(n_total,1):.1f}%)")
    return all_X, all_y, record_ids


# ---------------------------------------------------------------------------
# ──────────────────────────  PyTorch Dataset  ───────────────────────────────
# ---------------------------------------------------------------------------

def _pad_to_two_channels(arr: np.ndarray) -> np.ndarray:
    """Repeat single-channel arrays to produce (2, T) tensors."""
    if arr.shape[1] == 1:
        return np.concatenate([arr, arr], axis=1)
    return arr


# ---------------------------------------------------------------------------
# ──────────────────────── 4. Synthetic Dataset ──────────────────────────────
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_normal: int = 3000,
    n_apnea: int  = 3000,
    fs: float = float(FS_CAPNO),
    window_samples: int = WINDOW_SAMPLES,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate balanced synthetic apnea training data.

    Design principle
    ~~~~~~~~~~~~~~~~
    The preprocessing pipeline (SOS bandpass 0.05–0.7 Hz → Z-score) is the
    lens through which the model sees every window.  Both classes must therefore
    be discriminable AFTER that pipeline, not just in the raw domain.

    Normal windows
      CO₂ : strong periodic signal at respiratory rate (0.15–0.37 Hz).
            Fundamental + second harmonic give a capnogram-like shape.
            High amplitude relative to noise → periodic structure dominates
            after bandpass + Z-score.
      PPG : respiratory-frequency sinusoid at a different phase (simulates
            respiratory sinus arrhythmia / thoracic impedance).  Cardiac
            component (1–2 Hz) would sit above the 0.7 Hz bandpass cutoff,
            so we omit it here; the respiratory envelope IS the in-band feature.

    Apnea windows
      CO₂ : broadband Gaussian noise only — no spectral peak in 0.05–0.7 Hz.
      PPG : same broadband noise, different amplitude draw.

    After the SOS bandpass, normal windows retain a clear periodic spike in
    the FFT; apnea windows retain only the in-band noise floor.  Z-scoring
    maps both to unit variance but preserves the spectral shape difference,
    which is exactly what the CNN encoder is designed to detect.

    Returns
    -------
    X : (n_normal + n_apnea, 2, window_samples) float32   — raw, pre-filter
    y : (n_normal + n_apnea,)                  int32
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(window_samples) / fs       # time axis (s)

    X_list, y_list = [], []

    # ------------------------------------------------------------------ Normal
    for _ in range(n_normal):
        # Respiratory rate: 10–22 breaths/min → 0.167–0.367 Hz
        # (well inside the 0.05–0.7 Hz bandpass at every draw)
        f_resp  = rng.uniform(10, 22) / 60.0
        phase_c = rng.uniform(0, 2 * np.pi)
        phase_p = rng.uniform(0, 2 * np.pi)

        # Large amplitude ensures SNR >> 1 even after bandpass attenuates noise
        amp_co2 = rng.uniform(4.0, 8.0)
        amp_ppg = rng.uniform(3.0, 6.0)
        noise   = rng.uniform(0.05, 0.20)

        # CO₂: fundamental + 2nd harmonic (capnogram-like, no DC offset)
        co2 = amp_co2 * (
            np.sin(2 * np.pi * f_resp * t + phase_c)
            + 0.30 * np.sin(4 * np.pi * f_resp * t + phase_c)
        )
        co2 = co2 + rng.standard_normal(window_samples) * noise

        # PPG: respiratory-frequency sinus (in-band after bandpass)
        ppg = amp_ppg * np.sin(2 * np.pi * f_resp * t + phase_p)
        ppg = ppg + rng.standard_normal(window_samples) * noise

        X_list.append(np.stack([co2.astype(np.float32),
                                 ppg.astype(np.float32)]))
        y_list.append(0)

    # ------------------------------------------------------------------ Apnea
    for _ in range(n_apnea):
        # Broadband noise only — after bandpass only ~0.4 % of energy survives
        # (0.65 Hz passband out of 150 Hz Nyquist), giving a featureless residual
        noise_co2 = rng.uniform(0.5, 2.0)
        noise_ppg = rng.uniform(0.5, 2.0)

        co2 = rng.standard_normal(window_samples).astype(np.float32) * noise_co2
        ppg = rng.standard_normal(window_samples).astype(np.float32) * noise_ppg

        X_list.append(np.stack([co2, ppg]))
        y_list.append(1)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)

    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def load_synthetic_dataset(
    n_normal: int = 3000,
    n_apnea: int  = 3000,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Returns synthetic windows in the same list-of-records format as other loaders.
    Splits into 42 virtual "records" of ~143 windows each (mirroring CapnoBase size).
    """
    X, y = generate_synthetic_dataset(n_normal, n_apnea, seed=seed)

    n_records     = 42
    windows_each  = len(X) // n_records
    all_X, all_y, ids = [], [], []

    for i in range(n_records):
        s  = i * windows_each
        e  = s + windows_each if i < n_records - 1 else len(X)
        # Apply the same preprocessing pipeline used on real data
        Xi = X[s:e]                            # already (W, 2, T)
        Xi_proc = np.stack([preprocess_window(w) for w in Xi])
        all_X.append(Xi_proc)
        all_y.append(y[s:e])
        ids.append(f"synthetic_{i:03d}")

    n_total = sum(len(yy) for yy in all_y)
    n_ap    = int(sum(yy.sum() for yy in all_y))
    print(f"Synthetic : {n_records} virtual records | "
          f"{n_total} windows | {n_ap} apnea ({100*n_ap/max(n_total,1):.1f}%)")
    return all_X, all_y, ids


class ApneaDataset(Dataset):
    """
    Combined Dataset from CapnoBase + Apnea-ECG + UCDDB.

    All samples are presented as (2, 3000) float32 tensors.
    Single-channel records are duplicated across both channels.
    """

    def __init__(self, all_X: List[np.ndarray], all_y: List[np.ndarray]):
        if not all_X:
            self.X = np.empty((0, 2, WINDOW_SAMPLES), dtype=np.float32)
            self.y = np.array([], dtype=np.int64)
            return

        padded = [_pad_to_two_channels(x) for x in all_X]
        self.X = np.concatenate(padded, axis=0)
        self.y = np.concatenate(all_y, axis=0).astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.from_numpy(self.X[idx]),
                torch.tensor(self.y[idx], dtype=torch.long))

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights: [w_normal, w_apnea]."""
        n_total  = len(self.y)
        n_apnea  = int(self.y.sum())
        n_normal = n_total - n_apnea
        if n_apnea == 0 or n_normal == 0:
            return torch.ones(2)
        return torch.tensor([1.0, n_normal / n_apnea], dtype=torch.float32)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.15,
    use_physionet: bool = False,
    physionet_dir: Optional[str] = None,
    use_ucddb: bool = False,
    ucddb_dir: Optional[str] = None,
    ucddb_auto_download: bool = False,
    use_synthetic: bool = True,
    n_synthetic_normal: int = 3000,
    n_synthetic_apnea: int  = 3000,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train / val / test DataLoaders.

    Splitting is done at the record level to prevent data leakage.

    Returns: (train_loader, val_loader, test_loader, class_weights_tensor)
    """
    rng = np.random.default_rng(seed)

    # ---- CapnoBase ----
    csv_dir = Path(data_dir) / "data" / "csv"
    all_X_cap, all_y_cap, ids_cap = load_capnobase_dataset(str(csv_dir))

    cap_order = rng.permutation(len(ids_cap))
    n_val   = max(1, int(len(ids_cap) * val_split))
    n_test  = max(1, int(len(ids_cap) * test_split))
    n_train = len(ids_cap) - n_val - n_test

    def _split(order, Xs, ys):
        train_X = [Xs[i] for i in order[:n_train]]
        train_y = [ys[i] for i in order[:n_train]]
        val_X   = [Xs[i] for i in order[n_train:n_train+n_val]]
        val_y   = [ys[i] for i in order[n_train:n_train+n_val]]
        test_X  = [Xs[i] for i in order[n_train+n_val:]]
        test_y  = [ys[i] for i in order[n_train+n_val:]]
        return train_X, train_y, val_X, val_y, test_X, test_y

    train_X, train_y, val_X, val_y, test_X, test_y = _split(
        cap_order, all_X_cap, all_y_cap
    )

    def _add_source(Xs, ys):
        """Append a full dataset, splitting 70/15/15 among sets."""
        if not Xs:
            return
        order   = rng.permutation(len(Xs))
        nv      = max(1, int(len(order) * val_split))
        nt      = max(1, int(len(order) * test_split))
        ntrain  = len(order) - nv - nt
        train_X.extend(Xs[i] for i in order[:ntrain])
        train_y.extend(ys[i] for i in order[:ntrain])
        val_X.extend(  Xs[i] for i in order[ntrain:ntrain+nv])
        val_y.extend(  ys[i] for i in order[ntrain:ntrain+nv])
        test_X.extend( Xs[i] for i in order[ntrain+nv:])
        test_y.extend( ys[i] for i in order[ntrain+nv:])

    # ---- Apnea-ECG ----
    if use_physionet:
        Xs_pn, ys_pn, _ = load_apnea_ecg_dataset(physionet_dir)
        _add_source(Xs_pn, ys_pn)

    # ---- UCDDB ----
    if use_ucddb:
        _ucddb_dir = ucddb_dir or str(Path(data_dir) / "ucddb")
        Xs_uc, ys_uc, _ = load_ucddb_dataset(
            _ucddb_dir, auto_download=ucddb_auto_download
        )
        _add_source(Xs_uc, ys_uc)

    # ---- Synthetic ----
    if use_synthetic:
        Xs_syn, ys_syn, _ = load_synthetic_dataset(
            n_normal=n_synthetic_normal,
            n_apnea=n_synthetic_apnea,
            seed=seed,
        )
        _add_source(Xs_syn, ys_syn)

    train_ds = ApneaDataset(train_X, train_y)
    val_ds   = ApneaDataset(val_X,   val_y)
    test_ds  = ApneaDataset(test_X,  test_y)
    cw       = train_ds.class_weights()

    print(f"\nSplit  — train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}")
    print(f"Weights — normal: {cw[0]:.3f}  apnea: {cw[1]:.3f}\n")

    def _loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return (
        _loader(train_ds, shuffle=True),
        _loader(val_ds,   shuffle=False),
        _loader(test_ds,  shuffle=False),
        cw,
    )
