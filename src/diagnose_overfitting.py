"""
Diagnostic script to analyze model performance and detect overfitting
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def load_cached_data():
    """Load preprocessed data from cache."""
    cache_dir = Path(".preprocessed")

    # Find the most recent cache file
    cache_files = list(cache_dir.glob("data_*.pkl"))
    if not cache_files:
        print("No cached data found!")
        return None, None

    # Use the first cache file (non-augmented)
    cache_file = [f for f in cache_files if "_aug" not in f.name][0]
    print(f"Loading from: {cache_file}")

    with open(cache_file, "rb") as f:
        X, y = pickle.load(f)

    return X, y


def analyze_dataset(X, y):
    """Analyze dataset distribution and statistics."""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature shape: {X.shape}")

    # Class distribution
    class_names = ["premature (im)", "mature (m)", "overmature (om)"]
    class_counts = np.bincount(y)

    print("\nClass Distribution:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        percentage = 100 * count / len(y)
        print(f"  {name:20s}: {count:3d} samples ({percentage:5.2f}%)")

    # Check for class imbalance
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count

    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 5:
        print("⚠ SEVERE CLASS IMBALANCE DETECTED!")
        print("  This can cause the model to overfit to the majority class.")
    elif imbalance_ratio > 3:
        print("⚠ Moderate class imbalance detected.")
    else:
        print("✓ Classes are relatively balanced.")

    # Feature statistics
    print("\nFeature Statistics:")
    print(f"  Mean: {X.mean():.6f}")
    print(f"  Std: {X.std():.6f}")
    print(f"  Min: {X.min():.6f}")
    print(f"  Max: {X.max():.6f}")


def analyze_model_predictions(model, dataloader, device, label_names):
    """Analyze model predictions and compute metrics."""
    model.eval()

    all_predictions = []
    all_targets = []
    all_confidences = []

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)

            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = probs.max(1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_confidences = np.array(all_confidences)

    # Confusion matrix
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(all_targets, all_predictions)

    print("\n           Predicted")
    print("          ", end="")
    for name in label_names:
        print(f"{name[:10]:>10s}", end=" ")
    print()
    print("Actual")

    for i, name in enumerate(label_names):
        print(f"{name[:10]:>10s}", end=" ")
        for j in range(len(label_names)):
            print(f"{cm[i, j]:>10d}", end=" ")
        print()

    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_targets, all_predictions, target_names=label_names))

    # Confidence analysis
    print("\n" + "=" * 60)
    print("CONFIDENCE ANALYSIS")
    print("=" * 60)

    print("\nOverall confidence statistics:")
    print(f"  Mean confidence: {all_confidences.mean():.2%}")
    print(f"  Std confidence: {all_confidences.std():.2%}")
    print(f"  Min confidence: {all_confidences.min():.2%}")
    print(f"  Max confidence: {all_confidences.max():.2%}")

    # High confidence predictions
    high_conf_mask = all_confidences > 0.95
    high_conf_count = high_conf_mask.sum()
    high_conf_correct = (
        all_predictions[high_conf_mask] == all_targets[high_conf_mask]
    ).sum()

    print("\nHigh confidence predictions (>95%):")
    print(
        f"  Count: {high_conf_count} ({100 * high_conf_count / len(all_confidences):.1f}%)"
    )
    if high_conf_count > 0:
        print(f"  Accuracy: {100 * high_conf_correct / high_conf_count:.2f}%")

        if high_conf_count > len(all_confidences) * 0.8:
            print("  ⚠ WARNING: Too many high-confidence predictions!")
            print("  This is a strong indicator of overfitting.")

    # Per-class confidence
    print("\nPer-class confidence:")
    for i, name in enumerate(label_names):
        mask = all_targets == i
        if mask.sum() > 0:
            class_conf = all_confidences[mask].mean()
            class_acc = (all_predictions[mask] == all_targets[mask]).mean()
            print(
                f"  {name:15s}: confidence={class_conf:.2%}, accuracy={class_acc:.2%}"
            )


def diagnose_overfitting():
    """Main diagnostic function."""
    print("=" * 60)
    print("OVERFITTING DIAGNOSTIC TOOL")
    print("=" * 60)

    # Load data
    X, y = load_cached_data()
    if X is None:
        return

    # Analyze dataset
    analyze_dataset(X, y)

    # Check if model exists
    model_path = Path("models/best_model.pth")
    if not model_path.exists():
        print("\n⚠ Model not found. Run training first.")
        return

    # Load model
    print("\n" + "=" * 60)
    print("MODEL ANALYSIS")
    print("=" * 60)

    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    from train import CoconutCNN, CoconutDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split data (same as training)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create datasets
    val_dataset = CoconutDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load model
    input_shape = X.shape[1:]
    model = CoconutCNN(input_shape).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print(f"Model loaded from: {model_path}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Load label names
    label_map_path = Path("models/label_map.json")
    with open(label_map_path, "r") as f:
        label_data = json.load(f)
        label_names = label_data["label_names"]

    # Analyze validation set
    print("\n" + "=" * 60)
    print("VALIDATION SET PERFORMANCE")
    print("=" * 60)
    analyze_model_predictions(model, val_loader, device, label_names)

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    class_counts = np.bincount(y)
    imbalance_ratio = class_counts.max() / class_counts.min()

    print("\nTo address overfitting, consider:")
    print("1. ✓ Use the improved training script (train_improved.py)")
    print("2. ✓ Enable data augmentation (time stretching, pitch shifting, noise)")
    print("3. ✓ Increase dropout rate (0.5 or higher)")
    print("4. ✓ Add L2 regularization (weight_decay)")
    print("5. ✓ Use class weights to handle imbalance")
    print("6. ✓ Reduce model complexity")
    print("7. ✓ Early stopping based on validation loss")

    if imbalance_ratio > 3:
        print("\n⚠ Class imbalance is significant!")
        print("  Consider:")
        print("  - Using class weights (already in train_improved.py)")
        print("  - Oversampling minority classes")
        print("  - Stratified cross-validation")


if __name__ == "__main__":
    diagnose_overfitting()
