#!/usr/bin/env python3
"""
Coconut Maturity Classification Inference Script V2 (Mel Spectrogram + SVM)
===========================================================================

METHODOLOGY:
This inference script uses the improved classification approach from train_v2.py:
- Features: Mel spectrogram with mean/std pooling (256 features total)
- Classifier: SVM with RBF kernel and balanced class weights
- Achieves ~76% balanced accuracy on the coconut maturity dataset

Key Differences from V1:
1. Uses Mel spectrogram instead of MFCC features
2. Uses scikit-learn SVM instead of PyTorch CNN+LSTM
3. Faster inference (no GPU required)
4. More interpretable probability estimates

Usage:
    # Classify a specific audio file
    python inference_v2.py --audio path/to/coconut.wav --show_probs
    
    # Classify random sample from samples directory
    python inference_v2.py --random --show_probs
    
    # Convert audio file to WAV
    python inference_v2.py --convert input.mp3 --output output.wav
"""

import argparse
import json
import pickle
import random
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Configuration (must match training parameters)
# =============================================================================

class Config:
    """Configuration parameters - must match train_v2.py"""
    SAMPLE_RATE = 44100        # Audio sample rate
    N_MELS = 128               # Number of mel bands for spectrogram
    N_FFT = 2048               # FFT window size
    HOP_LENGTH = 512           # Hop length for STFT


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
        """
        Load audio file and convert to target sample rate.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate
            duration: Duration to load in seconds (None = full file)
            offset: Start reading at this offset (seconds)
        
        Returns:
            audio: Audio signal as numpy array
            sr: Sample rate
        """
        audio = None
        sr = target_sr

        # Prefer soundfile for robust reading
        try:
            data, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            
            # If multi-channel, mix to mono
            if np.ndim(data) > 1:
                data = np.mean(data, axis=1)
            
            # Handle duration/offset
            if duration is not None:
                start_sample = int(offset * file_sr)
                end_sample = start_sample + int(duration * file_sr)
                data = data[start_sample:end_sample]
            
            # Resample if needed
            if file_sr != target_sr:
                data = librosa.resample(
                    np.asarray(data, dtype=np.float32),
                    orig_sr=file_sr,
                    target_sr=target_sr,
                )
                sr = target_sr
            else:
                sr = file_sr
            
            audio = data
            
        except Exception:
            # Fallback to librosa
            try:
                audio, sr = librosa.load(audio_path, sr=target_sr, duration=duration, offset=offset)
            except Exception as e:
                raise RuntimeError(f"Failed to read audio '{audio_path}': {e}")

        # Sanitize audio
        audio = AudioConverter.sanitize_audio(audio)

        return audio, sr

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
        """Ensure audio contains finite values and is float32."""
        if audio is None:
            return np.zeros(0, dtype=np.float32)
        
        audio = np.asarray(audio)
        
        # Collapse multi-channel to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Replace non-finite values
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(
                audio,
                nan=0.0,
                posinf=np.finfo(np.float32).max / 100.0,
                neginf=np.finfo(np.float32).min / 100.0,
            )
        
        audio = audio.astype(np.float32)
        
        # Clip to reasonable range
        if np.max(np.abs(audio)) > 0:
            audio = np.clip(audio, -1.0, 1.0)
        
        return audio


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_mel_spectrogram_features(
    signal: np.ndarray,
    sr: int = Config.SAMPLE_RATE,
    n_mels: int = Config.N_MELS,
    n_fft: int = Config.N_FFT,
    hop_length: int = Config.HOP_LENGTH,
) -> np.ndarray:
    """
    Extract mel spectrogram features from audio signal.
    
    This function must match the feature extraction in train_v2.py exactly.
    
    Args:
        signal: Audio signal as numpy array
        sr: Sample rate
        n_mels: Number of mel frequency bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
    
    Returns:
        Feature vector: mean and std of mel spectrogram (2 * n_mels features)
    """
    # Ensure signal is float32
    signal = signal.astype(np.float32)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Extract statistics across time axis
    mel_mean = np.mean(mel_spec_db, axis=1)  # (n_mels,)
    mel_std = np.std(mel_spec_db, axis=1)    # (n_mels,)
    
    # Concatenate for final feature vector
    features = np.concatenate([mel_mean, mel_std])  # (2 * n_mels,)
    
    return features


# =============================================================================
# Classifier
# =============================================================================

class CoconutClassifierV2:
    """
    Coconut maturity classifier using Mel Spectrogram + SVM.
    
    This classifier uses the improved approach from train_v2.py:
    - Mel spectrogram features (more informative than MFCC)
    - SVM with RBF kernel and balanced class weights
    - Achieves ~76% balanced accuracy
    
    Attributes:
        model: Trained SVM model
        scaler: StandardScaler for feature normalization
        label_names: List of class names ['im', 'm', 'om']
        n_mels: Number of mel bands used in feature extraction
    """

    def __init__(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
    ):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to .pkl model file from train_v2.py
            metadata_path: Path to metadata JSON (optional)
        """
        self.model_path = model_path
        
        # Load model and scaler
        self._load_model(model_path)
        
        # Load metadata
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.label_names = metadata.get("label_names", ["im", "m", "om"])
                self.n_mels = metadata.get("n_mels", Config.N_MELS)
                self.sample_rate = metadata.get("sample_rate", Config.SAMPLE_RATE)
                self.n_fft = metadata.get("n_fft", Config.N_FFT)
                self.hop_length = metadata.get("hop_length", Config.HOP_LENGTH)
        else:
            # Fallback defaults
            self.label_names = ["im", "m", "om"]
            self.n_mels = Config.N_MELS
            self.sample_rate = Config.SAMPLE_RATE
            self.n_fft = Config.N_FFT
            self.hop_length = Config.HOP_LENGTH
        
        print(f"✓ Loaded model: {model_path}")
        print(f"  Labels: {self.label_names}")
        print(f"  Feature type: Mel spectrogram (n_mels={self.n_mels})")

    def _load_model(self, path: str):
        """Load model and scaler from pickle file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.scaler = data["scaler"]

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram features from audio.
        
        Args:
            audio: Audio signal as numpy array
        
        Returns:
            Feature vector (2 * n_mels,)
        """
        return extract_mel_spectrogram_features(
            audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
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
            If return_probs=False: Predicted label ('im', 'm', or 'om')
            If return_probs=True: Tuple of (label, probabilities array)
        """
        # Load audio if path provided
        if isinstance(audio, str):
            audio_data, _ = AudioConverter.load_audio(audio, target_sr=self.sample_rate)
        elif isinstance(audio, bytes):
            audio_data = np.frombuffer(audio, dtype=np.float32)
            audio_data = AudioConverter.sanitize_audio(audio_data)
        else:
            audio_data = AudioConverter.sanitize_audio(np.asarray(audio))
        
        # Normalize audio
        audio_data = AudioConverter.normalize_audio(audio_data)
        
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Empty audio data provided")
        
        # Extract features
        features = self.extract_features(audio_data)
        features = features.reshape(1, -1)  # (1, n_features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        pred_label_idx = self.model.predict(features_scaled)[0]
        pred_label = self.label_names[pred_label_idx]
        
        if return_probs:
            probs = self.model.predict_proba(features_scaled)[0]
            return pred_label, probs
        
        return pred_label

    def predict_batch(
        self,
        audio_list: list,
        return_probs: bool = False,
    ) -> list:
        """
        Predict coconut maturity for multiple audio samples.
        
        Args:
            audio_list: List of audio file paths or numpy arrays
            return_probs: Whether to return class probabilities
        
        Returns:
            List of predictions (or tuples if return_probs=True)
        """
        results = []
        for audio in audio_list:
            result = self.predict(audio, return_probs=return_probs)
            results.append(result)
        return results


# =============================================================================
# Helper Functions
# =============================================================================

def get_random_sample_file(samples_dir: str = "samples") -> Optional[str]:
    """Get a random WAV file from the samples directory."""
    samples_path = Path(samples_dir)
    ridge_dirs = [
        samples_path / "ridge-a",
        samples_path / "ridge-b",
        samples_path / "ridge-c",
    ]
    
    all_wav_files = []
    for ridge_dir in ridge_dirs:
        if ridge_dir.exists():
            wav_files = list(ridge_dir.glob("*.wav"))
            all_wav_files.extend(wav_files)
    
    if not all_wav_files:
        print(f"No WAV files found in {samples_dir}/ridge-{{a,b,c}}")
        return None
    
    random_file = random.choice(all_wav_files)
    return str(random_file)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Coconut Maturity Classification Inference V2 (Mel Spectrogram + SVM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
METHODOLOGY:
  This script uses an improved classification approach:
  - Features: Mel spectrogram with mean/std pooling
  - Classifier: SVM with RBF kernel and balanced class weights
  
EXAMPLES:
  # Classify a specific audio file
  python inference_v2.py --audio path/to/coconut.wav --show_probs
  
  # Classify random sample from samples directory  
  python inference_v2.py --random --show_probs
  
  # Convert audio file to WAV format
  python inference_v2.py --convert input.mp3 --output output.wav
        """,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/coconut_classifier_v2.pkl",
        help="Path to model .pkl file (default: models/coconut_classifier_v2.pkl)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="models/coconut_classifier_v2_metadata.json",
        help="Path to metadata JSON file",
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file for inference",
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="samples",
        help="Directory containing sample files",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random sample from samples directory",
    )
    parser.add_argument(
        "--convert",
        type=str,
        help="Convert audio file to WAV format",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for converted audio",
    )
    parser.add_argument(
        "--show_probs",
        action="store_true",
        help="Show class probabilities",
    )

    args = parser.parse_args()

    # Handle audio conversion
    if args.convert:
        print(f"Converting {args.convert} to WAV format...")
        output_path = AudioConverter.convert_to_wav(args.convert, output_path=args.output)
        print(f"✓ Saved to: {output_path}")
        return

    print("=" * 60)
    print("COCONUT MATURITY CLASSIFIER V2")
    print("Mel Spectrogram + SVM Approach")
    print("=" * 60)

    # Initialize classifier
    classifier = CoconutClassifierV2(
        model_path=args.model,
        metadata_path=args.metadata if Path(args.metadata).exists() else None,
    )

    # Get audio file
    audio_file = None
    if args.random:
        audio_file = get_random_sample_file(args.samples_dir)
        if audio_file:
            print(f"\nUsing random sample: {audio_file}")
    elif args.audio:
        audio_file = args.audio
        print(f"\nUsing specified file: {audio_file}")
    else:
        print("\nError: Please specify --audio or use --random flag")
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

    # Human-readable classification
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
