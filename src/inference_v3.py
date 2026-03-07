#!/usr/bin/env python3
"""
Coconut Maturity Classification Inference Script V3 (Conv1D + LSTM)
===================================================================

METHODOLOGY:
This inference script uses the paper architecture from train_v3.py:
- Features: 128 MFCC time sequences (full temporal information)
- Model: Conv1D + LSTM architecture from Caladcad & Piedad (2024)

Key Differences from V1/V2:
1. Uses MFCC sequences (not mean/std pooling) - preserves temporal info
2. Uses PyTorch Conv1D + LSTM (not SVM)
3. Better at capturing acoustic dynamics over time

DATA SOURCE OPTIONS:
1. --from_cache: Load directly from cached .npz data (same as training)
   - Ensures consistency with training data
   - Use --index N to select specific sample, or --random for random

2. --audio: Load from WAV file (original behavior)

Usage:
    # From cached data (recommended for testing)
    python inference_v3.py --from_cache --random --show_probs
    python inference_v3.py --from_cache --index 42 --show_probs

    # From audio file
    python inference_v3.py --audio path/to/coconut.wav --show_probs

Reference:
    Caladcad & Piedad (2024) - https://arxiv.org/abs/2408.14910
"""

import argparse
import hashlib
import json
import random
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Configuration (must match training)
# =============================================================================


class Config:
    """Configuration - must match train_v3.py"""

    SAMPLE_RATE = 44100
    N_MFCC = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    HIDDEN_SIZE = 64


# =============================================================================
# Model Architecture (must match train_v3.py)
# =============================================================================


class CoconutLSTM(nn.Module):
    """
    Paper architecture: Conv1D + LSTM classifier.
    Must match the model in train_v3.py exactly.
    """

    def __init__(
        self,
        input_size: int = Config.N_MFCC,
        hidden_size: int = Config.HIDDEN_SIZE,
        num_classes: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Conv1D layers
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        # Pooling
        self.pool = nn.AvgPool1d(kernel_size=2)

        # LSTM
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = lstm_out[:, -1, :]

        # Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


# =============================================================================
# Audio Utilities
# =============================================================================


class AudioConverter:
    """Utility class for loading and processing audio files."""

    @staticmethod
    def load_audio(
        audio_path: str,
        target_sr: int = Config.SAMPLE_RATE,
        duration: Optional[float] = None,
        offset: float = 0.0,
    ) -> Tuple[np.ndarray, int]:
        """Load audio file and convert to target sample rate."""
        try:
            data, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)

            if np.ndim(data) > 1:
                data = np.mean(data, axis=1)

            if duration is not None:
                start_sample = int(offset * file_sr)
                end_sample = start_sample + int(duration * file_sr)
                data = data[start_sample:end_sample]

            if file_sr != target_sr:
                data = librosa.resample(
                    np.asarray(data, dtype=np.float32),
                    orig_sr=file_sr,
                    target_sr=target_sr,
                )

            return AudioConverter.sanitize_audio(data), target_sr

        except Exception:
            try:
                audio, sr = librosa.load(
                    audio_path, sr=target_sr, duration=duration, offset=offset
                )
                return AudioConverter.sanitize_audio(audio), sr
            except Exception as e:
                raise RuntimeError(f"Failed to read audio '{audio_path}': {e}")

    @staticmethod
    def convert_to_wav(
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        sample_rate: int = Config.SAMPLE_RATE,
    ) -> str:
        """Convert audio file to WAV format."""
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix(".wav")

        audio, sr = AudioConverter.load_audio(str(input_path), target_sr=sample_rate)
        sf.write(str(output_path), audio, sr)
        return str(output_path)

    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    @staticmethod
    def sanitize_audio(audio: np.ndarray) -> np.ndarray:
        """Ensure audio contains finite float32 values."""
        if audio is None:
            return np.zeros(0, dtype=np.float32)

        audio = np.asarray(audio)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = np.clip(audio, -1.0, 1.0)

        return audio


# =============================================================================
# Feature Extraction
# =============================================================================


def extract_mfcc_sequence(
    signal: np.ndarray,
    sr: int = Config.SAMPLE_RATE,
    n_mfcc: int = Config.N_MFCC,
    n_fft: int = Config.N_FFT,
    hop_length: int = Config.HOP_LENGTH,
) -> np.ndarray:
    """
    Extract MFCC sequence from audio signal.

    Returns the full temporal sequence (not just statistics),
    which is required for the Conv1D + LSTM architecture.

    Args:
        signal: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length

    Returns:
        MFCC sequence (time_steps, n_mfcc)
    """
    signal = signal.astype(np.float32)

    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )

    return mfcc.T  # (time, features)


# =============================================================================
# Classifier
# =============================================================================


class CoconutClassifierV3:
    """
    Coconut maturity classifier using Conv1D + LSTM (Paper Architecture).

    This classifier uses the paper architecture from Caladcad & Piedad (2024):
    - MFCC sequence features (preserves temporal information)
    - Conv1D + LSTM model

    Attributes:
        model: Trained PyTorch model
        label_names: Class names ['im', 'm', 'om']
        n_mfcc: Number of MFCC coefficients
        device: PyTorch device
    """

    def __init__(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to .pth model file
            metadata_path: Path to metadata JSON (optional)
            device: PyTorch device (auto-detected if None)
        """
        self.model_path = model_path

        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load metadata
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.label_names = metadata.get("label_names", ["im", "m", "om"])
                self.n_mfcc = metadata.get("n_mfcc", Config.N_MFCC)
                self.sample_rate = metadata.get("sample_rate", Config.SAMPLE_RATE)
                self.n_fft = metadata.get("n_fft", Config.N_FFT)
                self.hop_length = metadata.get("hop_length", Config.HOP_LENGTH)
                self.hidden_size = metadata.get("hidden_size", Config.HIDDEN_SIZE)
        else:
            self.label_names = ["im", "m", "om"]
            self.n_mfcc = Config.N_MFCC
            self.sample_rate = Config.SAMPLE_RATE
            self.n_fft = Config.N_FFT
            self.hop_length = Config.HOP_LENGTH
            self.hidden_size = Config.HIDDEN_SIZE

        # Load model
        self._load_model(model_path)

        print(f"✓ Loaded model: {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Labels: {self.label_names}")
        print(f"  Architecture: Conv1D + LSTM (MFCC sequences)")

    def _load_model(self, path: str):
        """Load PyTorch model from checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(str(path), map_location=self.device)

        # Get model parameters from checkpoint
        input_size = checkpoint.get("input_size", self.n_mfcc)
        hidden_size = checkpoint.get("hidden_size", self.hidden_size)
        num_classes = checkpoint.get("num_classes", len(self.label_names))

        # Create model
        self.model = CoconutLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
        )

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC sequence from audio."""
        return extract_mfcc_sequence(
            audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

    def predict(
        self,
        audio: Union[str, np.ndarray, bytes],
        return_probs: bool = False,
    ) -> Union[str, Tuple[str, np.ndarray]]:
        """
        Predict coconut maturity from audio.

        Args:
            audio: Audio file path, numpy array, or binary data
            return_probs: Whether to return class probabilities

        Returns:
            Predicted label, optionally with probabilities
        """
        # Load audio
        if isinstance(audio, str):
            audio_data, _ = AudioConverter.load_audio(audio, target_sr=self.sample_rate)
        elif isinstance(audio, bytes):
            audio_data = np.frombuffer(audio, dtype=np.float32)
            audio_data = AudioConverter.sanitize_audio(audio_data)
        else:
            audio_data = AudioConverter.sanitize_audio(np.asarray(audio))

        # # Normalize
        # audio_data = AudioConverter.normalize_audio(audio_data)

        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Empty audio data")

        # Extract features
        mfcc_seq = self.extract_features(audio_data)  # (time, n_mfcc)

        # Prepare tensor
        x = (
            torch.FloatTensor(mfcc_seq).unsqueeze(0).to(self.device)
        )  # (1, time, n_mfcc)

        # Predict
        with torch.no_grad():
            outputs = self.model(x)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = self.label_names[pred_idx]

        if return_probs:
            return pred_label, probs
        return pred_label

    def predict_batch(
        self,
        audio_list: list,
        return_probs: bool = False,
    ) -> list:
        """Predict for multiple audio samples."""
        results = []
        for audio in audio_list:
            result = self.predict(audio, return_probs=return_probs)
            results.append(result)
        return results


# =============================================================================
# Helpers
# =============================================================================


def get_random_sample_file(samples_dir: str = "samples") -> Optional[str]:
    """Get a random WAV file from samples directory."""
    samples_path = Path(samples_dir)
    ridge_dirs = [
        samples_path / "ridge-a",
        samples_path / "ridge-b",
        samples_path / "ridge-c",
    ]

    all_wav_files = []
    for ridge_dir in ridge_dirs:
        if ridge_dir.exists():
            all_wav_files.extend(ridge_dir.glob("*.wav"))

    if not all_wav_files:
        print(f"No WAV files found in {samples_dir}/ridge-{{a,b,c}}")
        return None

    return str(random.choice(all_wav_files))


def load_cached_data(
    cache_dir: str = ".preprocessed",
    xlsx_path: str = "coconut_acoustic_signals.xlsx",
    signal_count: int = 132300,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load cached data from .npz file (same data used for training).

    Args:
        cache_dir: Directory containing cached .npz files
        xlsx_path: Path to original xlsx (used for hash matching)
        signal_count: Signal length (used for cache filename)

    Returns:
        signals: Audio signals array (N, signal_length)
        labels: Integer labels array (N,)
        label_names: List of class names ['im', 'm', 'om']
    """
    cache_path = Path(cache_dir)

    # Try to find cache file by hash
    if Path(xlsx_path).exists():
        with open(xlsx_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        cache_file = cache_path / f"coconut_data_{file_hash}_{signal_count}.npz"

        if cache_file.exists():
            print(f"📦 Loading cached data: {cache_file}")
            cached = np.load(cache_file, allow_pickle=True)
            return cached["signals"], cached["labels"], cached["label_names"].tolist()

    # Fallback: find any matching cache file
    cache_files = list(cache_path.glob(f"coconut_data_*_{signal_count}.npz"))
    if cache_files:
        cache_file = cache_files[0]
        print(f"📦 Loading cached data: {cache_file}")
        cached = np.load(cache_file, allow_pickle=True)
        return cached["signals"], cached["labels"], cached["label_names"].tolist()

    raise FileNotFoundError(
        f"No cached data found in {cache_dir}. "
        f"Run train_v3.py first to create the cache."
    )


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Coconut Maturity Inference V3 (Conv1D + LSTM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
METHODOLOGY (Paper Architecture):
  - Features: 128 MFCC time sequences
  - Model: Conv1D + LSTM (Caladcad & Piedad, 2024)

EXAMPLES:
  # From cached training data (recommended)
  python inference_v3.py --from_cache --random --show_probs
  python inference_v3.py --from_cache --index 42 --show_probs

  # From audio file
  python inference_v3.py --audio path/to/coconut.wav --show_probs
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/coconut_classifier_v3.pth",
        help="Path to model .pth file",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="models/coconut_classifier_v3_metadata.json",
        help="Path to metadata JSON",
    )
    parser.add_argument("--audio", type=str, help="Audio file for inference")
    parser.add_argument(
        "--samples_dir", type=str, default="samples", help="Directory with sample files"
    )
    parser.add_argument("--random", action="store_true", help="Use random sample")
    parser.add_argument("--convert", type=str, help="Convert audio to WAV")
    parser.add_argument("--output", type=str, help="Output path for conversion")
    parser.add_argument(
        "--show_probs", action="store_true", help="Show class probabilities"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="PyTorch device (cpu/cuda)"
    )

    # New: cached data options
    parser.add_argument(
        "--from_cache",
        action="store_true",
        help="Load from cached .npz data (same as training)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".preprocessed",
        help="Directory with cached .npz files",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coconut_acoustic_signals.xlsx",
        help="Path to xlsx (for cache hash matching)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Sample index from cached data (use with --from_cache)",
    )

    args = parser.parse_args()

    # Handle conversion
    if args.convert:
        print(f"Converting {args.convert} to WAV...")
        output = AudioConverter.convert_to_wav(args.convert, output_path=args.output)
        print(f"✓ Saved to: {output}")
        return

    print("=" * 60)
    print("COCONUT MATURITY CLASSIFIER V3")
    print("Conv1D + LSTM Architecture (Paper)")
    print("=" * 60)

    # Initialize classifier
    classifier = CoconutClassifierV3(
        model_path=args.model,
        metadata_path=args.metadata if Path(args.metadata).exists() else None,
        device=args.device,
    )

    # Determine data source
    if args.from_cache:
        # Load from cached .npz data
        print("\n📊 Loading from cached training data...")
        signals, labels, label_names = load_cached_data(
            cache_dir=args.cache_dir,
            xlsx_path=args.data,
        )
        print(f"   Total samples: {len(signals)}")
        print(f"   Labels: {label_names}")

        # Select sample
        if args.index is not None:
            if args.index < 0 or args.index >= len(signals):
                print(f"Error: Index {args.index} out of range [0, {len(signals) - 1}]")
                return
            sample_idx = args.index
        elif args.random:
            sample_idx = random.randint(0, len(signals) - 1)
        else:
            print("Error: Use --index N or --random with --from_cache")
            return

        audio_data = signals[sample_idx]
        true_label_idx = labels[sample_idx]
        true_label = label_names[true_label_idx]

        print(f"\n   Selected sample index: {sample_idx}")
        print(f"   True label: {true_label}")

        print("\n" + "=" * 60)
        print("RUNNING INFERENCE")
        print("=" * 60)

        if args.show_probs:
            pred_label, probs = classifier.predict(audio_data, return_probs=True)
            print(f"\nPredicted maturity: {pred_label}")
            print(
                f"True label: {true_label} {'✓' if pred_label == true_label else '✗'}"
            )
            print("\nClass probabilities:")
            for label, prob in zip(classifier.label_names, probs):
                marker = " ← true" if label == true_label else ""
                print(f"  {label:>3s}: {prob:.4f} ({prob * 100:.2f}%){marker}")
        else:
            pred_label = classifier.predict(audio_data)
            print(f"\nPredicted maturity: {pred_label}")
            print(
                f"True label: {true_label} {'✓' if pred_label == true_label else '✗'}"
            )

        label_mapping = {
            "im": "Premature (immature)",
            "m": "Mature",
            "om": "Overmature",
        }
        print(f"Classification: {label_mapping.get(pred_label, pred_label)}")

    else:
        # Original behavior: load from audio file
        audio_file = None
        if args.random:
            audio_file = get_random_sample_file(args.samples_dir)
            if audio_file:
                print(f"\nUsing random sample: {audio_file}")
        elif args.audio:
            audio_file = args.audio
            print(f"\nUsing specified file: {audio_file}")
        else:
            print("\nError: Please specify --audio, --random, or --from_cache")
            return

        if not audio_file or not Path(audio_file).exists():
            print(f"Error: Audio file not found: {audio_file}")
            return

        print("\n" + "=" * 60)
        print("RUNNING INFERENCE")
        print("=" * 60)

        if args.show_probs:
            pred_label, probs = classifier.predict(audio_file, return_probs=True)
            print(f"\nPredicted maturity: {pred_label}")
            print("\nClass probabilities:")
            for label, prob in zip(classifier.label_names, probs):
                print(f"  {label:>3s}: {prob:.4f} ({prob * 100:.2f}%)")
        else:
            pred_label = classifier.predict(audio_file)
            print(f"\nPredicted maturity: {pred_label}")

        label_mapping = {
            "im": "Premature (immature)",
            "m": "Mature",
            "om": "Overmature",
        }
        print(f"Classification: {label_mapping.get(pred_label, pred_label)}")

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
