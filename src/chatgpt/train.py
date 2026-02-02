#!/usr/bin/env python3
"""
train.py
Train coconut-maturity classifier from the Caladcad dataset (xlsx).
Saves best pytorch state_dict and an ONNX model for inference with onnxruntime.

Usage examples:
  python train.py --xlsx coconut_acoustic_signals.xlsx
  python train.py --xlsx data.xlsx --export_dir models --epochs 30
"""
import argparse
import hashlib
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
from audiomentations import Compose, AddGaussianSNR, PitchShift, TimeStretch, Shift, AddGaussianNoise
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------
# Utilities: reading & caching
# ---------------------------
SAMPLE_RATE = 44100
DEFAULT_SIGNAL_LEN = 132300  # 3 seconds * 44100
LABEL_MAP = {"im": 0, "m": 1, "om": 2}

def sha1_of_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def sha1_of_file(path: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA1 hash of a file's contents (streamed).
    """
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def sanitize_wave(wave: np.ndarray, name: str = "<unknown>"):
    """
    Ensure wave contains finite float32 values. Replace NaN/Inf with 0 and return contiguous array.
    """
    if not isinstance(wave, np.ndarray):
        wave = np.array(wave, dtype=np.float32)
    wave = wave.astype(np.float32, copy=False)
    if not np.isfinite(wave).all():
        nbad = int((~np.isfinite(wave)).sum())
        print(f"[warning] {name}: {nbad}/{wave.size} non-finite samples — replacing with 0.0")
        wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(wave, dtype=np.float32)

def load_xlsx_columns(
    xlsx_path: str,
    max_rows: int = DEFAULT_SIGNAL_LEN,
    drop_if_frac_nan: float = 0.5,
    allow_interpolate: bool = True,
    min_signal_len: int = 0,
    report_enabled: bool = True
):
    """
    Read the three sheets and return list of (waveform (numpy float32), label, name, sheet, original_len, used_len).
    Behavior:
      - For each column, use L = min(len(column), max_rows)
      - If L < min_signal_len, pad zeros up to min_signal_len before MFCC
      - Drop columns with too many NaNs (>= drop_if_frac_nan)
      - Interpolate small NaN gaps if allowed
      - Return records and a report list for CSV
    """
    xlsx = pd.ExcelFile(xlsx_path)
    sheets = ["Ridge A", "Ridge B", "Ridge C"]
    records = []
    report_rows = []
    dropped = []

    for sh in sheets:
        if sh not in xlsx.sheet_names:
            continue
        df = xlsx.parse(sh, header=0)
        for col in df.columns:
            col_series = pd.to_numeric(df[col], errors="coerce")
            col_arr = col_series.values
            orig_len = len(col_arr)
            L = min(orig_len, max_rows)
            if L == 0:
                dropped.append((sh, col, "len0"))
                report_rows.append({"name": col, "sheet": sh, "original_len": orig_len, "used_len": 0, "mfcc_time_frames": None, "dropped_reason": "len0"})
                continue

            arr = col_arr[:L].astype(np.float32, copy=False)
            n_nan = int(np.isnan(arr).sum())
            frac_nan = n_nan / float(L) if L > 0 else 1.0

            if n_nan == L:
                dropped.append((sh, col, "all_nan"))
                report_rows.append({"name": col, "sheet": sh, "original_len": orig_len, "used_len": L, "mfcc_time_frames": None, "dropped_reason": "all_nan"})
                continue

            if frac_nan >= drop_if_frac_nan:
                dropped.append((sh, col, f"frac_nan={frac_nan:.3f}"))
                report_rows.append({"name": col, "sheet": sh, "original_len": orig_len, "used_len": L, "mfcc_time_frames": None, "dropped_reason": f"frac_nan={frac_nan:.3f}"})
                continue

            if n_nan > 0:
                if allow_interpolate and (n_nan / L) < 0.1:
                    s = pd.Series(arr)
                    s_interpolated = s.interpolate(limit_direction="both", limit=1000)
                    s_interpolated = s_interpolated.fillna(0.0)
                    arr = s_interpolated.values.astype(np.float32)
                    # continue, interpolation happened
                else:
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # apply min_signal_len: pad zeros BEFORE MFCC extraction
            used_len = max(len(arr), min_signal_len)
            if len(arr) < used_len:
                padded = np.zeros(used_len, dtype=np.float32)
                padded[:len(arr)] = arr
                arr = padded

            # determine label token from header
            token = None
            if isinstance(col, str) and "_" in col:
                token = col.split("_")[-1]
            label = None
            if token in LABEL_MAP:
                label = LABEL_MAP[token]
            else:
                lc = str(col).lower()
                if lc.endswith("_om"):
                    label = LABEL_MAP["om"]
                elif lc.endswith("_m"):
                    label = LABEL_MAP["m"]
                elif lc.endswith("_im"):
                    label = LABEL_MAP["im"]
                else:
                    # unknown header -> skip
                    report_rows.append({"name": col, "sheet": sh, "original_len": orig_len, "used_len": used_len, "mfcc_time_frames": None, "dropped_reason": "unknown_header"})
                    print(f"[warn] skipping unknown header token for {sh}:{col}")
                    continue

            arr = sanitize_wave(arr, name=f"{sh}:{col}")
            records.append((arr, label, col, sh, orig_len, used_len))
            report_rows.append({"name": col, "sheet": sh, "original_len": orig_len, "used_len": used_len, "mfcc_time_frames": None, "dropped_reason": ""})

    if dropped:
        print("[summary] dropped columns:", dropped)
    if report_enabled:
        # return records and report rows; report will be completed after MFCC step with time frames
        return records, report_rows
    else:
        return records, []

def compute_mfcc(wave: np.ndarray, sr=SAMPLE_RATE, n_mfcc=40, hop_length=512, n_fft=2048):
    wave = sanitize_wave(wave, name="compute_mfcc_input")
    if wave.size == 0:
        return np.zeros((n_mfcc, 1), dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=wave.astype(np.float32), sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc.astype(np.float32)

def preprocess_and_cache(xlsx_path: str, max_rows: int, cache_dir: str, n_mfcc=40, min_signal_len=0, report_enabled=True):
    """
    - Loads columns (variable length up to max_rows), pads to min_signal_len if requested,
      computes per-sample MFCCs, pads/truncates MFCC time axis to dataset max_t,
      writes a preprocess report CSV if report_enabled, and caches the processed data.
    """
    file_hash = sha1_of_file(xlsx_path)
    key = f"{file_hash}|{max_rows}|{n_mfcc}|{min_signal_len}"
    h = sha1_of_str(key)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{h}.pt"
    if out_path.exists():
        print("Loading preprocessed from:", out_path)
        return torch.load(out_path)

    print("Preprocessing and caching to:", out_path)
    records, report_rows = load_xlsx_columns(
        xlsx_path,
        max_rows=max_rows,
        min_signal_len=min_signal_len,
        report_enabled=report_enabled
    )

    X_mfcc = []
    y = []
    names = []
    # compute MFCCs and update report rows with time frames
    for idx, (wave, label, name, sheet, orig_len, used_len) in enumerate(tqdm(records, desc="computing mfcc")):
        mf = compute_mfcc(wave, n_mfcc=n_mfcc)
        X_mfcc.append(mf)
        y.append(int(label))
        names.append(name)
        # update corresponding report row (match by name & sheet)
        for rr in report_rows:
            if rr["name"] == name and rr["sheet"] == sheet:
                rr["mfcc_time_frames"] = mf.shape[1]
                break

    if len(X_mfcc) == 0:
        raise RuntimeError("No valid samples found after loading; aborting.")

    max_t = max(m.shape[1] for m in X_mfcc)
    n_mfcc_actual = X_mfcc[0].shape[0]
    Xp = np.zeros((len(X_mfcc), n_mfcc_actual, max_t), dtype=np.float32)
    for i, m in enumerate(X_mfcc):
        t = m.shape[1]
        if t < max_t:
            Xp[i, :, :t] = m
        else:
            Xp[i] = m[:, :max_t]

    data = {"X": Xp, "y": np.array(y, dtype=np.int64), "names": names, "n_mfcc": n_mfcc}
    torch.save(data, out_path)

    # write report if requested
    return_path = None
    if report_enabled:
        # fill any remaining None mfcc_time_frames with 0
        for rr in report_rows:
            if rr["mfcc_time_frames"] is None:
                rr["mfcc_time_frames"] = 0
        # caller is responsible for export_dir; but when caching we save a small report to cache folder
        report_df = pd.DataFrame(report_rows)
        report_path = Path(".preprocessed") / f"{h}_report.csv"
        report_df.to_csv(report_path, index=False)
        print("Preprocess report saved to:", report_path)
        return_path = report_path

    return {"data": data, "report_path": return_path} if report_enabled else data

# ---------------------------
# Augmentation (audiomentation + procedural lowpass)
# ---------------------------
def make_augmenter(sr=SAMPLE_RATE):
    aug = Compose([
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.6),
        PitchShift(min_semitones=-3, max_semitones=3, p=0.6),
        Shift(min_fraction=-0.1, max_fraction=0.1, p=0.5),
        AddGaussianNoise(min_amplitude=0.0, max_amplitude=0.05, p=0.5),
        AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=30, p=0.5),
    ])
    return aug

def procedural_lowpass(wave: np.ndarray, sr=SAMPLE_RATE, cutoff=8000):
    # simple FFT-based lowpass
    f = np.fft.rfft(wave)
    freqs = np.fft.rfftfreq(len(wave), 1.0 / sr)
    f[freqs > cutoff] = 0
    out = np.fft.irfft(f)
    # ensure same length
    return out.astype(np.float32)

# ---------------------------
# Dataset & Model
# ---------------------------
class CoconutDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), int(self.y[idx])

class ConvLSTMClassifier(nn.Module):
    def __init__(self, n_mfcc=40, conv_channels=[32,64], lstm_hidden=64, n_classes=3, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mfcc, conv_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=conv_channels[1], hidden_size=lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (B, T, C)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ---------------------------
# Training loop
# ---------------------------
def train(args):
    # preprocess_and_cache returns dict when report_enabled; handle both shapes
    pre = preprocess_and_cache(args.xlsx, args.max_rows, ".preprocessed", n_mfcc=args.n_mfcc, min_signal_len=args.min_signal_len, report_enabled=args.report)
    if isinstance(pre, dict) and args.report:
        data = pre["data"]
        report_path = pre["report_path"]
    else:
        data = pre

    X = data["X"]
    y = data["y"]

    # If report_enabled, copy report to export_dir for user visibility
    if args.report and isinstance(pre, dict) and pre["report_path"] is not None:
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        final_report = export_dir / "preprocess_report.csv"
        # load the report from cache and write to export_dir
        df = pd.read_csv(pre["report_path"])
        df.to_csv(final_report, index=False)
        print("Copied preprocess report to:", final_report)

    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=args.val_split, stratify=y, random_state=42)
    train_ds = CoconutDataset(X[train_idx], y[train_idx])
    val_ds = CoconutDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTMClassifier(n_mfcc=args.n_mfcc, lstm_hidden=args.lstm_hidden, n_classes=args.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    os.makedirs(args.export_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_train = total_loss / len(train_loader.dataset)

        model.eval()
        total_v = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_v += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
        avg_val = total_v / len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)

        print(f"Epoch {epoch}/{args.epochs} train_loss={avg_train:.4f} val_loss={avg_val:.4f} val_acc={acc:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            fname = os.path.join(args.export_dir, args.model_name or f"best_epoch_{epoch}.pt")
            print("Saving best model to", fname)
            torch.save(model.state_dict(), fname)
            # export ONNX
            onnx_name = fname.replace(".pt", ".onnx")
            dummy = torch.randn(1, args.n_mfcc, X.shape[2], device=device)
            model.eval()
            try:
                torch.onnx.export(model, dummy, onnx_name,
                                  export_params=True,
                                  opset_version=12,
                                  do_constant_folding=True,
                                  input_names=['input'],
                                  output_names=['output'],
                                  dynamic_axes={'input': {0: 'batch_size', 2: 'time'}, 'output': {0: 'batch_size'}})
                print("Exported ONNX model to", onnx_name)
            except Exception as e:
                print("ONNX export failed:", e)

    data = preprocess_and_cache(args.xlsx, args.max_rows, ".preprocessed", n_mfcc=args.n_mfcc)
    X = data["X"]
    y = data["y"]
    # split train/val
    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=args.val_split, stratify=y, random_state=42)
    train_ds = CoconutDataset(X[train_idx], y[train_idx], augment=False)
    val_ds = CoconutDataset(X[val_idx], y[val_idx], augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device.type)
    model = ConvLSTMClassifier(n_mfcc=args.n_mfcc, lstm_hidden=args.lstm_hidden, n_classes=args.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = 1e9
    os.makedirs(args.export_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_train = total_loss / len(train_loader.dataset)

        # val
        model.eval()
        total_v = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_v += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
        avg_val = total_v / len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)

        print(f"Epoch {epoch}/{args.epochs} train_loss={avg_train:.4f} val_loss={avg_val:.4f} val_acc={acc:.4f}")

        # save if better
        if avg_val < best_val:
            best_val = avg_val
            fname = os.path.join(args.export_dir, args.model_name or f"best_epoch_{epoch}.pt")
            print("Saving best model to", fname)
            torch.save(model.state_dict(), fname)
            # also export ONNX
            onnx_name = fname.replace(".pt", ".onnx")
            # build dummy input
            dummy = torch.randn(1, args.n_mfcc, X.shape[2], device=device)
            model.eval()
            try:
                torch.onnx.export(model, dummy, onnx_name,
                                  export_params=True,
                                  opset_version=12,
                                  do_constant_folding=True,
                                  input_names=['input'],
                                  output_names=['output'],
                                  dynamic_axes={'input': {0: 'batch_size', 2: 'time'}, 'output': {0: 'batch_size'}})
                print("Exported ONNX model to", onnx_name)
            except Exception as e:
                print("ONNX export failed:", e)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True, help="path to coconut_acoustic_signals.xlsx")
    p.add_argument("--max_rows", type=int, default=DEFAULT_SIGNAL_LEN, help="rows to read per sample (default = 132300)")
    p.add_argument("--min_signal_len", type=int, default=0, help="minimum signal length in samples (pads zeros before MFCC)")
    p.add_argument("--n_mfcc", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--export_dir", default="models")
    p.add_argument("--model_name", default=None, help="optional filename for best model (ends with .pt)")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--lstm_hidden", type=int, default=64)
    p.add_argument("--report", type=bool, default=True, help="write per-sample preprocess report to export_dir")
    return p.parse_args()
if __name__ == "__main__":
    args = parse_args()
    train(args)
