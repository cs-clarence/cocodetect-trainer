"""
Coconut Maturity Classification Training Script - Improved Version
Addresses overfitting with better regularization and data augmentation
"""

import hashlib
import json
import pickle
import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    BATCH_SIZE = 16  # Reduced from 32 to help with generalization
    EPOCHS = 150
    LEARNING_RATE = 0.0005  # Reduced learning rate
    WEIGHT_DECAY = 0.01  # L2 regularization
    DROPOUT_RATE = 0.5  # Increased dropout

    # Data augmentation
    USE_AUGMENTATION = True
    TIME_STRETCH_RANGE = (0.9, 1.1)
    PITCH_SHIFT_RANGE = (-2, 2)
    NOISE_LEVEL = 0.005

    # Early stopping
    EARLY_STOPPING_PATIENCE = 20

    # Paths
    DATASET_PATH = "coconut_acoustic_signals.xlsx"
    CACHE_DIR = Path(".preprocessed")
    MODEL_DIR = Path("models")
    MODEL_FILENAME = "coconut_maturity_model.onnx"
    PYTORCH_MODEL_FILENAME = "best_model.pth"
    SCALER_FILENAME = "scaler.pkl"
    LABEL_MAP_FILENAME = "label_map.json"

    # Class mapping
    LABEL_MAP = {"im": 0, "m": 1, "om": 2}
    LABEL_NAMES = ["premature", "mature", "overmature"]

    @classmethod
    def get_model_path(cls):
        return cls.MODEL_DIR / cls.MODEL_FILENAME

    @classmethod
    def get_pytorch_model_path(cls):
        return cls.MODEL_DIR / cls.PYTORCH_MODEL_FILENAME

    @classmethod
    def get_label_map_path(cls):
        return cls.MODEL_DIR / cls.LABEL_MAP_FILENAME

    @classmethod
    def get_scaler_path(cls):
        return cls.MODEL_DIR / cls.SCALER_FILENAME


def get_file_hash(filepath):
    """Compute hash of file for caching."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def extract_mfcc_features(signal, sr=Config.SAMPLE_RATE, n_mfcc=Config.N_MFCC):
    """Extract MFCC features from audio signal."""
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

    return features


def augment_audio(signal, sr=Config.SAMPLE_RATE):
    """Apply data augmentation to audio signal."""
    augmented = signal.copy()

    # Time stretching
    if random.random() > 0.5:
        rate = random.uniform(*Config.TIME_STRETCH_RANGE)
        augmented = librosa.effects.time_stretch(augmented, rate=rate)

    # Pitch shifting
    if random.random() > 0.5:
        n_steps = random.uniform(*Config.PITCH_SHIFT_RANGE)
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)

    # Add noise
    if random.random() > 0.5:
        noise = np.random.normal(0, Config.NOISE_LEVEL, augmented.shape)
        augmented = augmented + noise

    return augmented


def load_excel_data(
    filepath,
    signal_length=Config.SIGNAL_LENGTH,
    use_augmentation=Config.USE_AUGMENTATION,
):
    """Load acoustic signals from Excel file."""
    print(f"Loading data from {filepath}...")

    # Check cache
    Config.CACHE_DIR.mkdir(exist_ok=True)
    file_hash = get_file_hash(filepath)
    cache_suffix = "_aug" if use_augmentation else ""
    cache_file = Config.CACHE_DIR / f"data_{file_hash}{cache_suffix}.pkl"

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

        for col in tqdm(df.columns, desc=f"{sheet_name}"):
            parts = col.split("_")
            if len(parts) < 2:
                continue

            label = parts[1]
            if label not in Config.LABEL_MAP:
                continue

            signal = df[col].values[:signal_length]

            if len(signal) < signal_length or np.any(np.isnan(signal)):
                continue

            # Original sample
            features = extract_mfcc_features(signal)
            all_features.append(features)
            all_labels.append(Config.LABEL_MAP[label])

            # Augmented samples (only for training)
            if use_augmentation:
                for _ in range(2):  # Create 2 augmented versions per sample
                    aug_signal = augment_audio(signal)
                    aug_features = extract_mfcc_features(aug_signal)
                    all_features.append(aug_features)
                    all_labels.append(Config.LABEL_MAP[label])

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"Loaded {len(X)} samples (including augmented)")
    print(f"Feature shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Cache the data
    with open(cache_file, "wb") as f:
        pickle.dump((X, y), f)

    return X, y


class CoconutDataset(Dataset):
    """PyTorch Dataset for coconut acoustic features with normalization."""

    def __init__(self, features, labels, scaler=None, fit_scaler=False):
        # Normalize features
        original_shape = features.shape
        features_flat = features.reshape(len(features), -1)

        if fit_scaler:
            self.scaler = StandardScaler()
            features_flat = self.scaler.fit_transform(features_flat)
        elif scaler is not None:
            self.scaler = scaler
            features_flat = self.scaler.transform(features_flat)
        else:
            self.scaler = None

        features = features_flat.reshape(original_shape)

        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ImprovedCoconutCNN(nn.Module):
    """Improved CNN model with better regularization."""

    def __init__(
        self, input_shape, num_classes=Config.NUM_CLASSES, dropout=Config.DROPOUT_RATE
    ):
        super(ImprovedCoconutCNN, self).__init__()

        n_features, time_steps = input_shape

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
            # Conv block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.7),
            # Conv block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        pooled_time_steps = time_steps // 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * pooled_time_steps, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def compute_class_weights(labels):
    """Compute class weights for imbalanced dataset."""
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)


def train_model(model, train_loader, val_loader, device, epochs=Config.EPOCHS):
    """Train the model with early stopping."""

    # Compute class weights for imbalanced dataset
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    class_weights = compute_class_weights(np.array(train_labels)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

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
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%"
        )
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Per-class accuracy
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        for i, label in enumerate(Config.LABEL_NAMES):
            mask = val_targets == i
            if mask.sum() > 0:
                class_acc = (
                    100.0
                    * (val_predictions[mask] == val_targets[mask]).sum()
                    / mask.sum()
                )
                print(f"  {label} accuracy: {class_acc:.2f}%")

        scheduler.step(avg_val_loss)

        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), str(best_model_path))
            print(
                f"✓ Saved best model (Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%)"
            )
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

        print("-" * 60)

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    return model


def export_to_onnx(model, input_shape, model_dir=None, model_filename=None):
    """Export model to ONNX format."""
    model.eval()

    if model_dir is None:
        model_dir = Config.MODEL_DIR
    if model_filename is None:
        model_filename = Config.MODEL_FILENAME

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / model_filename

    dummy_input = torch.randn(1, *input_shape)

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Augmentation: {'Enabled' if Config.USE_AUGMENTATION else 'Disabled'}")
    print(f"Dropout rate: {Config.DROPOUT_RATE}")
    print(f"Weight decay: {Config.WEIGHT_DECAY}")

    # Load data
    X, y = load_excel_data(Config.DATASET_PATH)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Train distribution: {np.bincount(y_train)}")
    print(f"Val distribution: {np.bincount(y_val)}")

    # Create datasets with normalization
    train_dataset = CoconutDataset(X_train, y_train, fit_scaler=True)
    val_dataset = CoconutDataset(X_val, y_val, scaler=train_dataset.scaler)

    # Save scaler
    scaler_path = Config.get_scaler_path()
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(scaler_path), "wb") as f:
        pickle.dump(train_dataset.scaler, f)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Create model
    input_shape = X.shape[1:]
    model = ImprovedCoconutCNN(input_shape).to(device)

    print(f"\nModel input shape: {input_shape}")
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


if __name__ == "__main__":
    main()
