"""
Coconut Maturity Classification Training Script (Enhanced)
Features: Checkpointing, Resume Training, Early Stopping
Based on Caladcad et al. (2020) research using CNN + MFCC
"""

import argparse
import hashlib
import json
import random
import signal
import warnings
from pathlib import Path
from typing import List, Optional, Tuple
import math
import os
from collections import defaultdict

from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Gain,
    LowPassFilter,
    HighPassFilter,
    ClippingDistortion,
)
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Global flag for graceful interruption
training_interrupted = False

warnings.filterwarnings("ignore", category=UserWarning, module="numba")
warnings.filterwarnings("ignore", category=FutureWarning)


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global training_interrupted
    print("\n\n⚠️  Training interrupted! Saving checkpoint...")
    training_interrupted = True


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

class Config:
    N_MFCC = 128
    SIGNAL_COUNT = 132300
    SAMPLE_RATE = 44100

class CoconutDataset(Dataset):
    """Dataset for coconut acoustic signals with MFCC extraction and optional audiomentations pipeline."""

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        sample_rate: int = Config.SAMPLE_RATE,
        n_mfcc: int = Config.N_MFCC,
        max_len: int = 130,
        augment: bool = False,
        aug_prob: float = 0.5,
        aug_strength: float = 1.0,
        use_audiomentations: bool = True,
        seed: Optional[int] = None,
    ):
        self.signals = signals
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.augment = augment
        self.aug_prob = float(aug_prob)
        self.aug_strength = float(aug_strength)
        self.use_audiomentations = use_audiomentations and (Compose is not None)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Build audiomentations pipeline if requested
        self.augmenter = None
        if self.augment and self.use_audiomentations:
            # Strength scales intensity / ranges of transforms
            s = max(0.0, float(self.aug_strength))
            # Compose a set of transforms similar to the study (time-stretch, pitch, shift, noise, gain, filters)
            self.augmenter = Compose(
                [
                    # Add light gaussian noise
                    AddGaussianNoise(min_amplitude=0.001 * s, max_amplitude=0.015 * s, p=0.5),
                    # Time stretch (warp)
                    TimeStretch(min_rate=0.85 ** s, max_rate=1.15 ** s, p=0.4),
                    # Pitch shift in semitones
                    PitchShift(min_semitones=int(-2 * s), max_semitones=int(2 * s), p=0.4),
                    # Shift (roll)
                    Shift(min_shift=-0.1 * s, max_shift=0.1 * s, p=0.5),
                    # Random gain (compression / gain)
                    Gain(min_gain_db=-6.0 * s, max_gain_db=6.0 * s, p=0.4),
                    # Slight lowpass or highpass sometimes (procedural filtering)
                    LowPassFilter(min_cutoff_freq=3000, max_cutoff_freq=int(8000 * s), p=0.25),
                    HighPassFilter(min_cutoff_freq=20, max_cutoff_freq=200, p=0.15),
                    # Mild clipping / distortion occasionally to simulate recording artifacts
                    ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=int(5 * s), p=0.1),
                ]
            )
        elif self.augment and not self.use_audiomentations:
            raise RuntimeError(
                "audiomentations is not available. Install it with `pip install audiomentations` "
                "or run without --augment."
            )

    def __len__(self):
        return len(self.signals)

    def _maybe_augment(self, signal: np.ndarray) -> np.ndarray:
        """Apply audiomentations pipeline with per-sample probability self.aug_prob"""
        if not self.augment or self.augmenter is None:
            return signal
        if random.random() > self.aug_prob:
            return signal

        # audiomentations expects float32 samples in range [-1, 1]
        sig = signal.astype(np.float32)
        max_abs = np.max(np.abs(sig)) if sig.size > 0 else 1.0
        if max_abs > 1.0:
            sig = sig / max_abs

        augmented = self.augmenter(samples=sig, sample_rate=self.sample_rate)
        # Keep values in a safe float32 range; if original had larger dynamic range, rescale
        augmented = augmented.astype(np.float32)
        return augmented

    def __getitem__(self, idx):
        signal = self.signals[idx].astype(np.float32)
        label = self.labels[idx]

        # optionally augment raw waveform
        if self.augment:
            signal = self._maybe_augment(signal)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=self.n_mfcc, n_fft=2048, hop_length=512)

        # Pad or truncate to fixed length
        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, : self.max_len]

        return torch.FloatTensor(mfcc), torch.LongTensor([label])[0]

def build_audiomentations_pipeline(aug_strength: float = 1.0):
    """Return an audiomentations.Compose pipeline (same defaults used for on-the-fly)."""
    if Compose is None:
        raise RuntimeError(
            "audiomentations not installed. Please install with `pip install audiomentations`."
        )

    s = max(0.0, float(aug_strength))
    return Compose(
        [
            AddGaussianNoise(min_amplitude=0.001 * s, max_amplitude=0.015 * s, p=0.5),
            TimeStretch(min_rate=0.85 ** s, max_rate=1.15 ** s, p=0.4),
            PitchShift(min_semitones=int(-2 * s), max_semitones=int(2 * s), p=0.4),
            Shift(min_fraction=-0.1 * s, max_fraction=0.1 * s, p=0.5),
            Gain(min_gain_in_db=-6.0 * s, max_gain_in_db=6.0 * s, p=0.4),
            LowPassFilter(min_low_pass_freq=3000, max_low_pass_freq=int(8000 * s), p=0.25),
            HighPassFilter(min_high_pass_freq=20, max_high_pass_freq=200, p=0.15),
            ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=int(5 * s), p=0.1),
        ]
    )

def generate_offline_augmented_dataset(
    signals: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    target_total: int = 13950,
    sample_rate: int = Config.SAMPLE_RATE,
    aug_strength: float = 1.0,
    seed: int = 42,
    out_dir: str = ".preprocessed",
    original_source_id: Optional[str] = None,  # used for cache filename uniqueness
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Create an augmented dataset on disk using audiomentations.
    - signals: np.ndarray of raw waveforms (1D arrays)
    - labels: integer labels (same length)
    - label_names: mapping of label id -> name
    - target_total: desired total number of samples in the augmented dataset
    - returns (aug_signals, aug_labels, cache_path)
    """

    if Compose is None:
        raise RuntimeError("audiomentations is required for offline augmentation. Install: pip install audiomentations")

    rng = np.random.RandomState(seed)
    random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    # compute class counts and desired per-class counts (preserve proportions)
    unique, counts = np.unique(labels, return_counts=True)
    total_original = len(labels)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))

    # compute target per class (proportional rounding)
    target_per_class = {}
    for cls in unique:
        frac = class_counts[cls] / total_original
        target_per_class[cls] = int(round(frac * target_total))

    # adjust rounding to exactly match target_total (distribute error)
    assigned = sum(target_per_class.values())
    diff = target_total - assigned
    if diff != 0:
        # sort classes by descending original size to distribute remainder
        sorted_classes = sorted(unique.tolist(), key=lambda c: class_counts[c], reverse=True)
        idx = 0
        step = 1 if diff > 0 else -1
        while diff != 0:
            target_per_class[sorted_classes[idx % len(sorted_classes)]] += step
            diff -= step
            idx += 1

    # build augmenter
    augmenter = build_audiomentations_pipeline(aug_strength)

    # Build index lists for each class
    class_indices = defaultdict(list)
    for i, lab in enumerate(labels):
        class_indices[int(lab)].append(i)

    augmented_signals = []
    augmented_labels = []

    # Start with original samples (keep them)
    for i, sig in enumerate(signals):
        augmented_signals.append(sig.astype(np.float32))
        augmented_labels.append(int(labels[i]))

    print(f"🔁 Generating offline augmented dataset: original={len(signals)} target_total={target_total}")
    # For each class, generate required extra samples
    pbar_total = sum(max(0, target_per_class[cls] - class_counts[cls]) for cls in unique)
    from tqdm import tqdm

    pbar = tqdm(total=pbar_total, desc="Offline augmenting")
    max_trials = 1000000  # safety to avoid infinite loops
    for cls in unique:
        orig_count = class_counts[cls]
        desired = target_per_class[cls]
        needed = max(0, desired - orig_count)
        if needed == 0:
            continue

        indices = class_indices[cls]
        if len(indices) == 0:
            raise ValueError(f"No examples for class {cls}")

        produced = 0
        trials = 0

        # We'll sample with replacement from indices until we produce `needed` augmented samples
        while produced < needed and trials < max_trials:
            trials += 1
            idx = rng.choice(indices)
            base_sig = signals[idx].astype(np.float32)

            # ensure values in [-1, 1] range if not already
            max_abs = np.max(np.abs(base_sig)) if base_sig.size else 1.0
            if max_abs > 1.0:
                base_sig = base_sig / max_abs

            try:
                aug = augmenter(samples=base_sig, sample_rate=sample_rate)
            except Exception as e:
                # if any failure occurs, fallback to small gaussian noise
                aug = base_sig + 1e-6 * rng.randn(base_sig.shape[0]).astype(np.float32)

            # audiomentations keeps length same for the transforms used; still ensure matching length
            if len(aug) != len(base_sig):
                if len(aug) < len(base_sig):
                    aug = np.pad(aug, (0, len(base_sig) - len(aug)), mode="constant")
                else:
                    aug = aug[: len(base_sig)]

            # clip and cast
            aug = np.clip(aug, -1.0, 1.0).astype(np.float32)

            augmented_signals.append(aug)
            augmented_labels.append(int(cls))

            produced += 1
            pbar.update(1)

    pbar.close()

    aug_signals = np.array(augmented_signals, dtype=object)  # keep as object array of 1D arrays
    aug_labels = np.array(augmented_labels, dtype=np.int32)

    # Save cache file unique to original_source_id (hash) and target_total
    source_tag = original_source_id or "orig"
    cache_name = f"augmented_{source_tag}_{len(signals)}to{target_total}.npz"
    cache_path = os.path.join(out_dir, cache_name)

    # we will save as allow_pickle arrays for variable-length waveforms (but in your dataset signals likely same length)
    np.savez(cache_path, signals=aug_signals, labels=aug_labels, label_names=label_names)
    print(f"💾 Augmented dataset saved: {cache_path} (total={len(augmented_labels)})")

    return aug_signals, aug_labels, cache_path


class CoconutCNN(nn.Module):
    """
    Updated model to reflect architecture described in
    "Deep learning classification system for coconut maturity levels based on acoustic signals"
    (two Conv1d layers -> average pooling -> optionally RNN/LSTM -> FC).
    See Table IV in the paper for layer params (Conv1d: 128->32, 32->64; RNN/LSTM hidden=64; dropout=0.5). :contentReference[oaicite:1]{index=1}

    Args:
        n_mfcc: number of MFCC channels (input channels to Conv1d). The paper lists Conv1d(128,32,...),
                so set n_mfcc=128 if you want exact parity with the reported config. Default kept at 40 for compatibility.
        num_classes: number of target classes.
        dropout: dropout probability for FC head.
        use_recurrent: 'lstm', 'rnn', or None. If 'lstm' or 'rnn', a recurrent layer (hidden_size=64) is applied after convs.
        rnn_hidden: hidden size of recurrent layer (default 64 to match the paper).
        rnn_num_layers: number of recurrent layers.
        bidirectional: whether the recurrent layer is bidirectional.
    """
    def __init__(
        self,
        n_mfcc: int = Config.N_MFCC,
        num_classes: int = 3,
        dropout: float = 0.5,
        use_recurrent: Optional[str] = "lstm",   # "lstm", "rnn", or None
        rnn_hidden: int = 64,
        rnn_num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super(CoconutCNN, self).__init__()

        # --- Convolutional feature extractor (matches Table IV: conv -> conv -> avgpool) ---
        # First conv: in_channels = n_mfcc (paper reports 128->32; set n_mfcc=128 to match exactly)
        self.conv1 = nn.Conv1d(in_channels=n_mfcc, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        # Second conv
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        # Average pooling (paper lists an avg pool layer). We'll provide an adaptive pool option
        # to reduce temporal dimension to 1 when not using RNN; otherwise we feed sequence to RNN.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Recurrent layer (optional) — paper compares RNN and LSTM; hidden units = 64
        self.use_recurrent = use_recurrent.lower() if isinstance(use_recurrent, str) else None
        self.rnn_hidden = rnn_hidden
        self.rnn_num_layers = rnn_num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if self.use_recurrent == "lstm":
            self.rnn = nn.LSTM(
                input_size=64,  # conv output channels -> features per time-step
                hidden_size=rnn_hidden,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif self.use_recurrent == "rnn":
            self.rnn = nn.RNN(
                input_size=64,
                hidden_size=rnn_hidden,
                num_layers=rnn_num_layers,
                batch_first=True,
                nonlinearity="tanh",
                bidirectional=bidirectional,
            )
        else:
            self.rnn = None

        # Fully-connected classifier head:
        # - If recurrent used: fc input dim = rnn_hidden * num_directions
        # - Else (CNN-only): fc input dim = 64 (conv output channels after avgpool)
        fc_in_dim = rnn_hidden * self.num_directions if self.rnn is not None else 64
        self.fc1 = nn.Linear(fc_in_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

        # initialize weights (optional, small helpful init)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x expected shape: (batch, n_mfcc, seq_len)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # If a recurrent layer is used, send the conv feature sequence to it.
        if self.rnn is not None:
            # x shape currently: (batch, channels=64, seq_len)
            # transpose to (batch, seq_len, features)
            x = x.permute(0, 2, 1).contiguous()  # (B, T, F)

            # RNN/LSTM returns (output, hidden)
            output, hidden = self.rnn(x)  # output: (B, T, H * num_directions)

            # --- ONNX-friendly extraction of final timestep features ---
            # Use torch.select to pick last timestep along time dim (avoids negative-index Slice)
            seq_len = output.size(1)
            last_idx = seq_len - 1  # Python int
            # select returns shape (B, H*num_directions)
            last_step = output.select(1, last_idx)

            features = last_step  # (batch, hidden * num_directions)

        else:
            # CNN-only: pool across time to produce a single vector per example
            # x shape: (batch, channels=64, seq_len)
            x = self.adaptive_pool(x)    # -> (batch, 64, 1)

            # Use reshape instead of squeeze to avoid potential 0-D tensors
            features = x.reshape(x.size(0), x.size(1))  # (batch, 64)

        # FC head
        x = self.relu(self.fc1(features))
        x = self.dropout(x)
        x = self.fc2(x)  # logits

        return x


def load_coconut_data(
    xlsx_path: str, signal_count: Optional[int] = None, cache_dir: str = ".preprocessed"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load coconut acoustic data from Excel file with caching"""

    if signal_count is None:
        signal_count = 132300

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # Create hash for caching
    with open(xlsx_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    cache_file = cache_path / f"coconut_data_{file_hash}_{signal_count}.npz"

    # Check cache
    if cache_file.exists():
        print(f"📦 Loading cached data from {cache_file}")
        cached = np.load(cache_file, allow_pickle=True)
        return cached["signals"], cached["labels"], cached["label_names"].tolist()

    print(f"📂 Loading data from {xlsx_path}")

    # Read Excel file
    xl_file = pd.ExcelFile(xlsx_path)
    ridges = ["Ridge A", "Ridge B", "Ridge C"]

    all_signals = []
    all_labels = []

    for ridge in ridges:
        print(f"  Processing {ridge}...")
        df = pd.read_excel(xl_file, sheet_name=ridge, nrows=signal_count)

        for col in df.columns:
            parts = col.split("_")
            if len(parts) >= 2:
                maturity_code = parts[-1]
                signal = df[col].values
                signal = np.nan_to_num(signal, nan=0.0)

                all_signals.append(signal)
                all_labels.append(maturity_code)

    signals = np.array(all_signals, dtype=np.float32)

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(all_labels)
    label_names = label_encoder.classes_.tolist()

    print("\n📊 Dataset Summary:")
    print(f"  Total samples: {len(signals)}")
    print(f"  Signal shape: {signals[0].shape}")
    print("  Label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c, name in zip(unique, counts, label_names):
        print(f"    {name}: {c} samples")

    # Cache the data
    np.savez(cache_file, signals=signals, labels=labels, label_names=label_names)
    print(f"💾 Data cached to {cache_file}\n")

    return signals, labels, label_names


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    history: dict,
    best_val_loss: float,
    checkpoint_path: str,
    is_best: bool = False,
):
    """Save training checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, checkpoint_path)

    if is_best:
        print(f"💾 Best model saved to {checkpoint_path}")
    else:
        print(f"💾 Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer, scheduler
) -> Tuple[int, dict, float]:
    """Load training checkpoint"""

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    history = checkpoint["history"]
    best_val_loss = checkpoint["best_val_loss"]

    print(f"✅ Resumed from epoch {epoch}")
    print(f"   Best val loss so far: {best_val_loss:.4f}")

    return epoch, history, best_val_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 60,
    learning_rate: float = 0.001,
    device: str = "cpu",
    checkpoint_dir: str = "checkpoints",
    model_name: str = "coconut_classifier",
    resume: bool = False,
    early_stopping_patience: int = 10,
) -> Tuple[nn.Module, dict]:
    """
    Train the coconut classification model with checkpointing

    Features:
    - Saves best model immediately to disk
    - Can resume training after interruption
    - Early stopping to prevent overfitting
    - Graceful interruption handling (Ctrl+C)
    """

    global training_interrupted

    # Setup checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    best_checkpoint = checkpoint_path / f"{model_name}_best.pth"
    latest_checkpoint = checkpoint_path / f"{model_name}_latest.pth"

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Resume from checkpoint if requested
    if resume and latest_checkpoint.exists():
        print("\n🔄 Resuming training from checkpoint...")
        start_epoch, history, best_val_loss = load_checkpoint(
            str(latest_checkpoint), model, optimizer, scheduler
        )
        start_epoch += 1  # Start from next epoch

    print(f"\n🚀 Starting training from epoch {start_epoch + 1}/{num_epochs}")
    print(f"💾 Checkpoints will be saved to: {checkpoint_path}")
    print("⚠️  Press Ctrl+C to interrupt and save progress\n")

    for epoch in range(start_epoch, num_epochs):
        if training_interrupted:
            print("\n✋ Training interrupted by user")
            break

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for signals, labels in pbar:
            if training_interrupted:
                break

            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if training_interrupted:
            break

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for signals, labels in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"
            ):
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Update learning rate
        scheduler.step(val_loss)

        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model immediately to disk
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                history,
                best_val_loss,
                str(best_checkpoint),
                is_best=True,
            )
            print(f"  ⭐ New best model! (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        # Save latest checkpoint (for resume)
        save_checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            history,
            best_val_loss,
            str(latest_checkpoint),
            is_best=False,
        )

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"\n🛑 Early stopping triggered! No improvement for {early_stopping_patience} epochs"
            )
            print(f"   Best validation loss: {best_val_loss:.4f}")
            break

        print()  # Blank line for readability

    # Load best model before returning
    if best_checkpoint.exists():
        print("\n✅ Loading best model from checkpoint...")
        checkpoint = torch.load(str(best_checkpoint))
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"   Best epoch: {checkpoint['epoch'] + 1}")
        print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")

    return model, history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    label_names: List[str],
    device: str = "cpu",
) -> dict:
    """Evaluate model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Evaluating"):
            signals = signals.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(f"\n{'=' * 60}")
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=label_names))

    print("\nCONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    return {
        "predictions": all_preds,
        "labels": all_labels,
        "confusion_matrix": cm.tolist(),
    }


def export_to_onnx(
    model: nn.Module, export_path: str, n_mfcc: int = Config.N_MFCC, max_len: int = 130
):
    """Export model to ONNX format (ONNX-friendly export settings)."""
    model.eval()
    dummy_input = torch.randn(1, n_mfcc, max(2, max_len), dtype=torch.float32)

    dynamic_axes = {
        "input": {0: "batch_size", 2: "seq_len"},
        "output": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=14,  # recommend 13-15; 14 is a good default
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    print(f"✓ Model exported to ONNX: {export_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train coconut maturity classification model (Enhanced with checkpointing)"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to Excel dataset",
        default="coconut_acoustic_signals.xlsx",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="models",
        help="Directory to save final models",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="coconut_classifier",
        help="Base name for model files",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--signal_count", type=int, default=132300, help="Number of signals per sample"
    )
    parser.add_argument(
        "--n_mfcc", type=int, default=Config.N_MFCC, help="Number of MFCC coefficients"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--early_stopping", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument("--test_split", type=float, default=0.1, help="Test set split")
    parser.add_argument(
        "--val_split", type=float, default=0.1, help="Validation set split"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument(
        "--aug_prob", type=float, default=0.5, help="Per-sample augmentation probability (0-1)"
    )
    parser.add_argument(
        "--aug_strength",
        type=float,
        default=1.0,
        help="Augmentation strength multiplier (>=0.0). Lower -> milder; Higher -> stronger",
    )
    parser.add_argument(
        "--use_audiomentations",
        action="store_true",
        help="When --augment is used, use the audiomentations library (recommended)",
    )
    parser.add_argument(
        "--offline_augment",
        action="store_true",
        help="Generate an offline augmented dataset and save to disk before training",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=13950,
        help="When --offline_augment is used, target total dataset size (default 13950)",
    )
    parser.add_argument(
        "--aug_cache_dir",
        type=str,
        default=".preprocessed",
        help="Directory to save offline augmented dataset cache",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create directories
    export_dir = Path(args.export_dir)
    export_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")

    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    signals, labels, label_names = load_coconut_data(
        args.data, signal_count=args.signal_count
    )

    if args.offline_augment:
        print("\n" + "=" * 60)
        print("OFFLINE AUGMENTATION: Generating augmented dataset")
        print("=" * 60)
        # build augmenter (we use aug_strength and seed)
        aug_signals, aug_labels, cache_path = generate_offline_augmented_dataset(
            signals=signals,
            labels=labels,
            label_names=label_names,
            target_total=args.target_size,
            sample_rate=Config.SAMPLE_RATE,
            aug_strength=args.aug_strength,
            seed=args.seed,
            out_dir=args.aug_cache_dir,
            original_source_id=hashlib.md5(open(args.data, "rb").read()).hexdigest()[:8],
        )

        # Replace loaded signals/labels with augmented ones for subsequent splitting/training
        # If the saved format used object arrays, convert to a 2D array if shape consistent
        # Attempt to stack if possible
        try:
            # try to stack to 2D array (n_signals, signal_length)
            signals = np.stack(aug_signals).astype(np.float32)
        except Exception:
            # keep as object arrays; CoconutDataset handles variable lengths by padding/truncating during MFCC
            signals = np.array(aug_signals, dtype=object)
        labels = np.array(aug_labels, dtype=np.int32)

        print(f"   Using augmented dataset cache: {cache_path}")
        print(f"   New dataset size: {len(signals)} samples")

    # Split data
    print("=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)

    X_temp, X_test, y_temp, y_test = train_test_split(
        signals,
        labels,
        test_size=args.test_split,
        random_state=args.seed,
        stratify=labels,
    )

    val_size_adjusted = args.val_split / (1 - args.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=args.seed,
        stratify=y_temp,
    )

    print(f"  Train set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")

    if args.augment:
        print(f"🔀 Data augmentation enabled (prob={args.aug_prob}, strength={args.aug_strength})")
        if args.use_audiomentations:
            print("   Using audiomentations pipeline.")
        else:
            print("   Using audiomentations when available (fallback will error if not installed).")

    # Create datasets
    train_dataset = CoconutDataset(
        X_train,
        y_train,
        n_mfcc=args.n_mfcc,
        augment=args.augment,
        aug_prob=args.aug_prob,
        aug_strength=args.aug_strength,
        use_audiomentations=args.use_audiomentations or True,  # prefer audiomentations when available
        seed=args.seed,
    )
    val_dataset = CoconutDataset(X_val, y_val, n_mfcc=args.n_mfcc, augment=False)
    test_dataset = CoconutDataset(X_test, y_test, n_mfcc=args.n_mfcc, augment=False)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Create model
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    model = CoconutCNN(
        n_mfcc=args.n_mfcc, num_classes=len(label_names), dropout=args.dropout
    )
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Train model
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        resume=args.resume,
        early_stopping_patience=args.early_stopping,
    )

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    test_metrics = evaluate_model(model, test_loader, label_names, device)

    # Save final model
    pytorch_path = export_dir / f"{args.model_name}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_names": label_names,
            "n_mfcc": args.n_mfcc,
            "num_classes": len(label_names),
            "dropout": args.dropout,
        },
        pytorch_path,
    )
    print(f"\n✓ PyTorch model saved: {pytorch_path}")

    # Export to ONNX
    onnx_path = export_dir / f"{args.model_name}.onnx"
    # export_to_onnx(model, str(onnx_path), n_mfcc=args.n_mfcc)

    # Save metadata
    metadata = {
        "label_names": label_names,
        "n_mfcc": args.n_mfcc,
        "sample_rate": Config.SAMPLE_RATE,
        "max_len": 130,
        "num_classes": len(label_names),
        "training_params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "early_stopping_patience": args.early_stopping,
        },
        "dataset_info": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        },
        "history": history,
        "test_metrics": {"confusion_matrix": test_metrics["confusion_matrix"]},
    }

    metadata_path = export_dir / f"{args.model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📁 Models saved in: {export_dir}")
    print(f"  - PyTorch model: {pytorch_path.name}")
    print(f"  - ONNX model: {onnx_path.name}")
    print(f"  - Metadata: {metadata_path.name}")
    print(f"\n📁 Checkpoints saved in: {checkpoint_dir}")
    print(f"  - Best model: {args.model_name}_best.pth")
    print(f"  - Latest checkpoint: {args.model_name}_latest.pth")


if __name__ == "__main__":
    main()
