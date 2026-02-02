"""
Coconut Maturity Classification Training Script
Based on: "Acoustic signal dataset for detecting coconut maturity level"
by June Anne Caladcad and Eduardo Jr. Piedad
"""

import hashlib
import json
import pickle
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Configuration
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SIGNAL_LENGTH = 132300  # 3 seconds at 44100 Hz
    N_MFCC = 40
    N_FFT = 2048
    HOP_LENGTH = 512

    # Model parameters
    NUM_CLASSES = 3  # premature, mature, overmature
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001

    # Paths
    DATASET_PATH = "coconut_acoustic_signals.xlsx"
    CACHE_DIR = Path(".preprocessed")
    MODEL_DIR = Path("models")  # Directory to save models
    MODEL_FILENAME = "coconut_maturity_model.onnx"  # ONNX model filename
    PYTORCH_MODEL_FILENAME = "best_model.pth"  # PyTorch checkpoint filename
    SCALER_FILENAME = "scaler.pkl"
    LABEL_MAP_FILENAME = "label_map.json"

    # Class mapping
    LABEL_MAP = {"im": 0, "m": 1, "om": 2}  # premature, mature, overmature
    LABEL_NAMES = ["premature", "mature", "overmature"]

    @classmethod
    def get_model_path(cls):
        """Get full path for ONNX model."""
        return cls.MODEL_DIR / cls.MODEL_FILENAME

    @classmethod
    def get_pytorch_model_path(cls):
        """Get full path for PyTorch model."""
        return cls.MODEL_DIR / cls.PYTORCH_MODEL_FILENAME

    @classmethod
    def get_label_map_path(cls):
        """Get full path for label map."""
        return cls.MODEL_DIR / cls.LABEL_MAP_FILENAME

    @classmethod
    def get_scaler_path(cls):
        """Get full path for scaler."""
        return cls.MODEL_DIR / cls.SCALER_FILENAME


def get_file_hash(filepath):
    """Compute hash of file for caching."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def extract_mfcc_features(signal, sr=Config.SAMPLE_RATE, n_mfcc=Config.N_MFCC):
    """Extract MFCC features from audio signal."""
    # Ensure signal is the right length
    if len(signal) < Config.SIGNAL_LENGTH:
        signal = np.pad(signal, (0, Config.SIGNAL_LENGTH - len(signal)))
    else:
        signal = signal[: Config.SIGNAL_LENGTH]

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH
    )

    # Extract delta and delta-delta
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Combine features
    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

    return features  # Shape: (n_mfcc * 3, time_steps)


def load_excel_data(filepath, signal_length=Config.SIGNAL_LENGTH):
    """Load acoustic signals from Excel file."""
    print(f"Loading data from {filepath}...")

    # Check cache
    Config.CACHE_DIR.mkdir(exist_ok=True)
    file_hash = get_file_hash(filepath)
    cache_file = Config.CACHE_DIR / f"data_{file_hash}.pkl"

    if cache_file.exists():
        print("Loading from cache...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Load from Excel
    xl_file = pd.ExcelFile(filepath)
    sheets = ["Ridge A", "Ridge B", "Ridge C"]

    all_features = []
    all_labels = []

    for sheet_name in sheets:
        print(f"Processing {sheet_name}...")
        df = pd.read_excel(xl_file, sheet_name=sheet_name)

        # Process each column (each coconut sample)
        for col in tqdm(df.columns, desc=f"{sheet_name}"):
            # Extract label from column name (e.g., c85_om -> om)
            parts = col.split("_")
            if len(parts) < 2:
                continue

            label = parts[1]
            if label not in Config.LABEL_MAP:
                continue

            # Get signal data
            signal = df[col].values[:signal_length]

            # Skip if signal is too short or has NaN values
            if len(signal) < signal_length or np.any(np.isnan(signal)):
                continue

            # Extract MFCC features
            features = extract_mfcc_features(signal)

            all_features.append(features)
            all_labels.append(Config.LABEL_MAP[label])

    # Convert to numpy arrays
    X = np.array(all_features)  # Shape: (n_samples, n_features, time_steps)
    y = np.array(all_labels)

    print(f"Loaded {len(X)} samples")
    print(f"Feature shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Cache the data
    with open(cache_file, "wb") as f:
        pickle.dump((X, y), f)

    return X, y


class CoconutDataset(Dataset):
    """PyTorch Dataset for coconut acoustic features."""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CoconutCNN(nn.Module):
    """CNN model for coconut maturity classification."""

    def __init__(self, input_shape, num_classes=Config.NUM_CLASSES):
        super(CoconutCNN, self).__init__()

        n_features, time_steps = input_shape

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            # Conv block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            # Conv block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        # Calculate the size after convolutions
        # time_steps -> /2 -> /2 -> /2 = time_steps / 8
        pooled_time_steps = time_steps // 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * pooled_time_steps, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, val_loader, device, epochs=Config.EPOCHS):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_acc = 0.0

    # Ensure model directory exists
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = Config.get_pytorch_model_path()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%"
        )
        print(f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(best_model_path))
            print(f"✓ Saved best model to {best_model_path} (Val Acc: {val_acc:.2f}%)")

        print("-" * 60)

    return model


def export_to_onnx(model, input_shape, model_dir=None, model_filename=None):
    """Export model to ONNX format."""
    model.eval()

    # Use config defaults if not provided
    if model_dir is None:
        model_dir = Config.MODEL_DIR
    if model_filename is None:
        model_filename = Config.MODEL_FILENAME

    # Ensure model directory exists
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Full save path
    save_path = model_dir / model_filename

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)

    # Export with opset 11 for better compatibility
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            verbose=False,
        )
        print(f"✓ Model exported to {save_path}")

        # Verify the exported model
        import onnx

        onnx_model = onnx.load(str(save_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")

    except Exception as e:
        print(f"⚠ Warning: ONNX export with opset 11 failed: {e}")
        print("Trying with opset 10...")

        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(save_path),
                export_params=True,
                opset_version=10,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                verbose=False,
            )
            print(f"✓ Model exported to {save_path} (opset 10)")

        except Exception as e2:
            print(f"✗ ONNX export failed: {e2}")
            print("Saving PyTorch model only...")
            pytorch_path = model_dir / model_filename.replace(".onnx", "_pytorch.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_shape": input_shape,
                },
                str(pytorch_path),
            )
            raise


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    X, y = load_excel_data(Config.DATASET_PATH)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")

    # Create datasets and dataloaders
    train_dataset = CoconutDataset(X_train, y_train)
    val_dataset = CoconutDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Create model
    input_shape = X.shape[1:]  # (n_features, time_steps)
    model = CoconutCNN(input_shape).to(device)

    print(f"Model input shape: {input_shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    model = train_model(model, train_loader, val_loader, device)

    # Load best model
    best_model_path = Config.get_pytorch_model_path()
    model.load_state_dict(torch.load(str(best_model_path)))

    # Export to ONNX
    export_to_onnx(model, input_shape)

    # Save label mapping
    label_map_path = Config.get_label_map_path()
    with open(str(label_map_path), "w") as f:
        json.dump(
            {"label_map": Config.LABEL_MAP, "label_names": Config.LABEL_NAMES},
            f,
            indent=2,
        )

    print("\nTraining complete!")
    print(f"Models saved to directory: {Config.MODEL_DIR}")
    print(f"  - ONNX model: {Config.MODEL_FILENAME}")
    print(f"  - PyTorch checkpoint: {Config.PYTORCH_MODEL_FILENAME}")
    print(f"  - Label map: {Config.LABEL_MAP_FILENAME}")


if __name__ == "__main__":
    main()
