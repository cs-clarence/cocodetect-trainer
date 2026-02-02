"""
Coconut Maturity Classification Training Script (Enhanced)
Features: Checkpointing, Resume Training, Early Stopping
Based on Caladcad et al. (2020) research using CNN + MFCC
"""

import argparse
import hashlib
import json
import signal
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

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


class CoconutDataset(Dataset):
    """Dataset for coconut acoustic signals with MFCC feature extraction"""

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        sample_rate: int = 44100,
        n_mfcc: int = 40,
        max_len: int = 130,
    ):
        self.signals = signals
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_len = max_len

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=self.n_mfcc)

        # Pad or truncate to fixed length
        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, : self.max_len]

        return torch.FloatTensor(mfcc), torch.LongTensor([label])[0]


class CoconutCNN(nn.Module):
    """CNN model for coconut maturity classification"""

    def __init__(self, n_mfcc: int = 40, num_classes: int = 3, dropout: float = 0.5):
        super(CoconutCNN, self).__init__()

        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(n_mfcc, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))

        x = self.adaptive_pool(x)
        x = x.squeeze(-1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

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
    model: nn.Module, export_path: str, n_mfcc: int = 40, max_len: int = 130
):
    """Export model to ONNX format"""
    model.eval()
    dummy_input = torch.randn(1, n_mfcc, max_len)

    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
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
        "--n_mfcc", type=int, default=40, help="Number of MFCC coefficients"
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

    # Create datasets
    train_dataset = CoconutDataset(X_train, y_train, n_mfcc=args.n_mfcc)
    val_dataset = CoconutDataset(X_val, y_val, n_mfcc=args.n_mfcc)
    test_dataset = CoconutDataset(X_test, y_test, n_mfcc=args.n_mfcc)

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
    export_to_onnx(model, str(onnx_path), n_mfcc=args.n_mfcc)

    # Save metadata
    metadata = {
        "label_names": label_names,
        "n_mfcc": args.n_mfcc,
        "sample_rate": 44100,
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
