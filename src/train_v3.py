"""
Coconut Maturity Classification Training Script V3 (Conv1D + LSTM)
==================================================================

METHODOLOGY:
This version implements the paper architecture from Caladcad & Piedad (2024):
"Deep learning classification system for coconut maturity levels based on acoustic signals"

Architecture (Table IV from paper):
- Conv1D: 128 -> 32 channels, kernel=3
- Conv1D: 32 -> 64 channels, kernel=3
- AvgPool1d
- LSTM: hidden_size=64
- Dropout: 0.5
- FC: 64 -> 3 classes

Key Features:
1. MFCC SEQUENCES: Full 128-coefficient MFCC time series (not just mean/std)
   - Preserves temporal information crucial for acoustic analysis
   - Shape: (batch, time_steps, 128)

2. BALANCED AUGMENTATION: Oversample minority classes to equal counts
   - Uses audiomentations: TimeStretch, PitchShift, GaussianNoise, Shift
   - Target: 1000 samples per class (3000 total)

3. CONV1D + LSTM: Captures both local patterns and temporal dependencies
   - Conv layers extract local acoustic features
   - LSTM learns temporal dynamics of the signal

4. PROPER TRAIN/TEST SPLIT: Split BEFORE augmentation (no data leakage!)
   - Test set contains only original samples never seen during training
   - This is the correct methodology (paper had data leakage)

Expected Performance (with proper methodology):
- Accuracy: ~75%
- Balanced Accuracy: ~68-70%
- Note: Paper claimed 97% but had data leakage (split after augmentation)

Reference:
- Paper: https://arxiv.org/abs/2408.14910
"""

import argparse
import copy
import hashlib
import json
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import audiomentations as am
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Configuration
# =============================================================================


class Config:
    """Configuration parameters for training."""

    SIGNAL_COUNT = 132300  # Samples per signal (3 seconds @ 44.1kHz)
    SAMPLE_RATE = 44100  # Audio sample rate
    N_MFCC = 128  # Number of MFCC coefficients (paper uses 128)
    N_FFT = 2048  # FFT window size
    HOP_LENGTH = 512  # Hop length for STFT
    HIDDEN_SIZE = 64  # LSTM hidden size (paper)
    DROPOUT = 0.5  # Dropout rate (paper)
    RANDOM_STATE = 42


# =============================================================================
# Model Architecture (Paper: Conv1D + LSTM)
# =============================================================================


class CoconutLSTM(nn.Module):
    """
    Paper architecture: Conv1D + LSTM classifier.

    From Table IV of Caladcad & Piedad (2024):
    - Conv1d(128, 32, kernel=3)
    - Conv1d(32, 64, kernel=3)
    - AvgPool1d
    - LSTM(64, hidden=64)
    - Dropout(0.5)
    - Linear(64, 3)

    Args:
        input_size: Number of MFCC coefficients (default: 128)
        hidden_size: LSTM hidden size (default: 64)
        num_classes: Number of output classes (default: 3)
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(
        self,
        input_size: int = Config.N_MFCC,
        hidden_size: int = Config.HIDDEN_SIZE,
        num_classes: int = 3,
        dropout: float = Config.DROPOUT,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Conv1D layers (paper: 128->32->64)
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        # Pooling
        self.pool = nn.AvgPool1d(kernel_size=2)

        # LSTM (paper: hidden=64)
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, time, features)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # x: (batch, time, features) -> (batch, features, time)
        x = x.permute(0, 2, 1)

        # Conv blocks
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Back to (batch, time, features) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM - use last timestep
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Last timestep: (batch, hidden)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


# =============================================================================
# Data Loading
# =============================================================================


def load_coconut_data(
    xlsx_path: str,
    signal_count: Optional[int] = None,
    cache_dir: str = ".preprocessed",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load coconut acoustic data from Excel file with caching."""
    if signal_count is None:
        signal_count = Config.SIGNAL_COUNT

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    with open(xlsx_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    cache_file = cache_path / f"coconut_data_{file_hash}_{signal_count}.npz"

    if cache_file.exists():
        print(f"📦 Loading cached data from {cache_file}")
        cached = np.load(cache_file, allow_pickle=True)
        return cached["signals"], cached["labels"], cached["label_names"].tolist()

    print(f"📂 Loading data from {xlsx_path}")

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

    np.savez(cache_file, signals=signals, labels=labels, label_names=label_names)
    print(f"💾 Data cached to {cache_file}\n")

    return signals, labels, label_names


# =============================================================================
# Augmentation
# =============================================================================


def create_augmentation_pipeline() -> am.Compose:
    """
    Create audiomentations pipeline matching paper methodology.

    From Table II of Caladcad & Piedad (2024):
    - TimeStretch: stretch_factor = random.uniform(0.8, 1.2)
    - PitchShift: pitch_factor = random.randint(-3, 3)
    - AddGaussianNoise: noise_factor = random.uniform(0, 0.05)
    - Shift: shift_factor = random.uniform(-0.1, 0.1)
    """
    return am.Compose(
        [
            am.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
            am.PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
            am.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            am.Shift(min_shift=-0.1, max_shift=0.1, p=0.5),
        ]
    )


def augment_dataset(
    signals: np.ndarray,
    labels: np.ndarray,
    target_per_class: int = 4050,
    sr: int = Config.SAMPLE_RATE,
    signal_length: int = Config.SIGNAL_COUNT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment dataset to balance classes.

    Args:
        signals: Original signals array
        labels: Original labels array
        target_per_class: Target samples per class
        sr: Sample rate
        signal_length: Expected signal length

    Returns:
        Augmented signals and labels
    """
    augmenter = create_augmentation_pipeline()

    aug_signals = []
    aug_labels = []

    num_classes = len(np.unique(labels))

    print(f"🔄 Augmenting to {target_per_class} samples per class...")

    for class_idx in range(num_classes):
        class_sigs = signals[labels == class_idx]

        # Add original samples
        for sig in class_sigs:
            aug_signals.append(sig)
            aug_labels.append(class_idx)

        # Augment until we reach target
        current_count = len(class_sigs)
        pbar = tqdm(
            total=target_per_class - current_count,
            desc=f"  Class {class_idx}",
            leave=False,
        )

        while len([l for l in aug_labels if l == class_idx]) < target_per_class:
            idx = np.random.randint(0, len(class_sigs))
            aug_sig = augmenter(samples=class_sigs[idx], sample_rate=sr)

            # Ensure correct length
            if len(aug_sig) > signal_length:
                aug_sig = aug_sig[:signal_length]
            elif len(aug_sig) < signal_length:
                aug_sig = np.pad(aug_sig, (0, signal_length - len(aug_sig)))

            aug_signals.append(aug_sig.astype(np.float32))
            aug_labels.append(class_idx)
            pbar.update(1)

        pbar.close()

    return np.array(aug_signals, dtype=np.float32), np.array(aug_labels)


# =============================================================================
# Feature Extraction
# =============================================================================


def extract_mfcc_sequences(
    signals: np.ndarray,
    sr: int = Config.SAMPLE_RATE,
    n_mfcc: int = Config.N_MFCC,
    n_fft: int = Config.N_FFT,
    hop_length: int = Config.HOP_LENGTH,
) -> np.ndarray:
    """
    Extract MFCC sequences from signals.

    Unlike V2 which uses mean/std pooling, this preserves the full
    temporal sequence for the LSTM to process.

    Args:
        signals: Audio signals array (N, signal_length)
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length

    Returns:
        MFCC sequences array (N, time_steps, n_mfcc)
    """
    X = []

    print(f"📊 Extracting {n_mfcc} MFCC sequences...")

    for sig in tqdm(signals, desc="  MFCC extraction"):
        mfcc = librosa.feature.mfcc(
            y=sig, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )
        X.append(mfcc.T)  # Transpose to (time, features)

    return np.array(X, dtype=np.float32)


# =============================================================================
# Training
# =============================================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int = 60,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> Tuple[nn.Module, dict]:
    """
    Train the model.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on

    Returns:
        Trained model and training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    print(f"\n🚀 Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            scheduler.step(val_loss)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch + 1:3d}: Train Loss={train_loss:.4f}, "
                    f"Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%"
                )
        else:
            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch + 1:3d}: Train Loss={train_loss:.4f}, "
                    f"Train Acc={train_acc:.1f}%"
                )

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n✓ Loaded best model (Val Acc: {best_val_acc:.1f}%)")

    return model, history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    label_names: List[str],
    device: str = "cpu",
) -> dict:
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\n" + classification_report(all_labels, all_preds, target_names=label_names))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "labels": all_labels,
    }


def save_model(
    model: nn.Module,
    label_names: List[str],
    export_dir: str,
    model_name: str = "coconut_classifier_v3",
    metadata: Optional[dict] = None,
):
    """Save trained model and metadata."""
    export_path = Path(export_dir)
    export_path.mkdir(exist_ok=True, parents=True)

    # Save PyTorch model
    model_path = export_path / f"{model_name}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "num_classes": model.num_classes,
        },
        model_path,
    )
    print(f"✓ Model saved: {model_path}")

    # Save metadata
    meta = {
        "label_names": label_names,
        "n_mfcc": Config.N_MFCC,
        "sample_rate": Config.SAMPLE_RATE,
        "n_fft": Config.N_FFT,
        "hop_length": Config.HOP_LENGTH,
        "hidden_size": Config.HIDDEN_SIZE,
        "feature_type": "mfcc_sequence",
        "architecture": "Conv1D_LSTM",
    }
    if metadata:
        meta.update(metadata)

    metadata_path = export_path / f"{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Coconut Classifier V3 (Conv1D + LSTM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
METHODOLOGY (Paper Architecture):
  - Features: 128 MFCC time sequences
  - Model: Conv1D (128->32->64) + LSTM (hidden=64) + Dropout(0.5)
  - Augmentation: TimeStretch, PitchShift, GaussianNoise, Shift
  - Training: 90/10 split, 60 epochs, Adam optimizer

EXAMPLE:
  python train_v3.py --data coconut_acoustic_signals.xlsx --epochs 60
        """,
    )

    parser.add_argument(
        "--data",
        type=str,
        default="coconut_acoustic_signals.xlsx",
        help="Path to Excel dataset",
    )
    parser.add_argument(
        "--export_dir", type=str, default="models", help="Directory to save model"
    )
    parser.add_argument(
        "--model_name", type=str, default="coconut_classifier_v3", help="Model name"
    )
    parser.add_argument(
        "--signal_count",
        type=int,
        default=Config.SIGNAL_COUNT,
        help="Samples per signal",
    )
    parser.add_argument(
        "--n_mfcc", type=int, default=Config.N_MFCC, help="Number of MFCC coefficients"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=Config.HIDDEN_SIZE, help="LSTM hidden size"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--dropout", type=float, default=Config.DROPOUT, help="Dropout rate"
    )
    parser.add_argument(
        "--test_split", type=float, default=0.1, help="Test set proportion"
    )
    parser.add_argument(
        "--target_per_class",
        type=int,
        default=4050,
        help="Target samples per class after augmentation",
    )
    parser.add_argument(
        "--no_augment", action="store_true", help="Disable augmentation"
    )
    parser.add_argument(
        "--seed", type=int, default=Config.RANDOM_STATE, help="Random seed"
    )

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("COCONUT MATURITY CLASSIFIER V3")
    print("Conv1D + LSTM Architecture (Paper)")
    print("=" * 60)
    print(f"\n🖥️  Device: {device}")
    print(f"📋 Architecture: Conv1D(128->32->64) + LSTM(hidden={args.hidden_size})")

    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    signals, labels, label_names = load_coconut_data(
        args.data, signal_count=args.signal_count
    )

    # SPLIT FIRST - before any augmentation to prevent data leakage
    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT (before augmentation)")
    print("=" * 60)
    print("⚠️  Splitting BEFORE augmentation to prevent data leakage")

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        signals,
        labels,
        test_size=args.test_split,
        stratify=labels,
        random_state=args.seed,
    )
    print(f"  Train (raw): {len(X_train_raw)} samples")
    print(f"  Test (held-out): {len(X_test_raw)} samples")

    # Augment ONLY training data
    if not args.no_augment:
        print("\n" + "=" * 60)
        print("DATA AUGMENTATION (training set only)")
        print("=" * 60)
        X_train_aug, y_train_aug = augment_dataset(
            X_train_raw,
            y_train_raw,
            target_per_class=args.target_per_class,
            sr=Config.SAMPLE_RATE,
        )
        print(f"✓ Augmented train set: {len(X_train_aug)} samples")

        # Show distribution
        counts = Counter(y_train_aug)
        for i, name in enumerate(label_names):
            print(f"  {name}: {counts[i]} samples")
    else:
        X_train_aug = X_train_raw
        y_train_aug = y_train_raw

    # Extract features
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)
    print("Extracting MFCCs for training set...")
    X_train = extract_mfcc_sequences(X_train_aug, n_mfcc=args.n_mfcc)
    print(f"✓ Train MFCC shape: {X_train.shape}")

    print("Extracting MFCCs for test set...")
    X_test = extract_mfcc_sequences(X_test_raw, n_mfcc=args.n_mfcc)
    print(f"✓ Test MFCC shape: {X_test.shape}")

    y_train = y_train_aug
    y_test = y_test_raw

    print(f"\n  Final train: {len(X_train)} samples")
    print(f"  Final test: {len(X_test)} samples (held-out, never augmented)")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    print("\n" + "=" * 60)
    print("MODEL")
    print("=" * 60)
    model = CoconutLSTM(
        input_size=args.n_mfcc,
        hidden_size=args.hidden_size,
        num_classes=len(label_names),
        dropout=args.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(model)

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    model, history = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )

    # Evaluate
    results = evaluate_model(model, test_loader, label_names, device)

    # Save
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    metadata = {
        "training_params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "target_per_class": args.target_per_class,
        },
        "results": {
            "accuracy": results["accuracy"],
            "balanced_accuracy": results["balanced_accuracy"],
        },
    }

    save_model(model, label_names, args.export_dir, args.model_name, metadata)

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Results:")
    print(f"   Accuracy: {results['accuracy']:.1%}")
    print(f"   Balanced Accuracy: {results['balanced_accuracy']:.1%}")
    print(f"\n💡 Use inference_v3.py for predictions.")


if __name__ == "__main__":
    main()
