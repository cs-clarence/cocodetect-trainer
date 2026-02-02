"""
Coconut Maturity Classification Inference Script
Supports real-time inference on audio files using ONNX Runtime
"""

import io
import json
import random
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf


# Configuration
class InferenceConfig:
    SAMPLE_RATE = 44100
    SIGNAL_LENGTH = 132300  # 3 seconds
    N_MFCC = 40
    N_FFT = 2048
    HOP_LENGTH = 512

    MODEL_DIR = Path("models")  # Directory where models are saved
    MODEL_FILENAME = "coconut_maturity_model.onnx"
    LABEL_MAP_FILENAME = "label_map.json"

    @classmethod
    def get_model_path(cls):
        """Get full path for ONNX model."""
        return cls.MODEL_DIR / cls.MODEL_FILENAME

    @classmethod
    def get_label_map_path(cls):
        """Get full path for label map."""
        return cls.MODEL_DIR / cls.LABEL_MAP_FILENAME


class AudioProcessor:
    """Utility class for audio format conversion and preprocessing."""

    @staticmethod
    def load_audio_from_file(
        filepath: Union[str, Path], sr: int = InferenceConfig.SAMPLE_RATE
    ) -> np.ndarray:
        """
        Load audio from various file formats.

        Supported formats: WAV, MP3, OGG, FLAC, M4A
        """
        audio, _ = librosa.load(filepath, sr=sr, mono=True)
        return audio

    @staticmethod
    def load_audio_from_binary(
        binary_data: bytes, sr: int = InferenceConfig.SAMPLE_RATE
    ) -> np.ndarray:
        """
        Load audio from binary data.

        Args:
            binary_data: Raw audio bytes (WAV, MP3, etc.)
            sr: Target sample rate

        Returns:
            Audio signal as numpy array
        """
        # Try soundfile first (faster for WAV)
        try:
            audio, orig_sr = sf.read(io.BytesIO(binary_data))
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert to mono
            if orig_sr != sr:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
            return audio
        except:
            pass

        # Fallback to librosa (supports more formats)
        audio, _ = librosa.load(io.BytesIO(binary_data), sr=sr, mono=True)
        return audio

    @staticmethod
    def load_audio_from_numpy(
        audio_array: np.ndarray,
        orig_sr: int,
        target_sr: int = InferenceConfig.SAMPLE_RATE,
    ) -> np.ndarray:
        """
        Load audio from numpy array with resampling.

        Args:
            audio_array: Audio signal as numpy array
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio signal
        """
        if orig_sr != target_sr:
            audio_array = librosa.resample(
                audio_array, orig_sr=orig_sr, target_sr=target_sr
            )

        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)  # Convert to mono

        return audio_array

    @staticmethod
    def normalize_signal_length(
        signal: np.ndarray, target_length: int = InferenceConfig.SIGNAL_LENGTH
    ) -> np.ndarray:
        """
        Normalize signal to target length by padding or truncating.

        Args:
            signal: Input audio signal
            target_length: Desired signal length

        Returns:
            Normalized signal
        """
        if len(signal) < target_length:
            # Pad with zeros
            signal = np.pad(signal, (0, target_length - len(signal)))
        else:
            # Truncate
            signal = signal[:target_length]

        return signal

    @staticmethod
    def extract_mfcc_features(
        signal: np.ndarray, sr: int = InferenceConfig.SAMPLE_RATE
    ) -> np.ndarray:
        """
        Extract MFCC features from audio signal.

        Args:
            signal: Audio signal
            sr: Sample rate

        Returns:
            MFCC features with shape (n_mfcc * 3, time_steps)
        """
        # Normalize signal length
        signal = AudioProcessor.normalize_signal_length(signal)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=InferenceConfig.N_MFCC,
            n_fft=InferenceConfig.N_FFT,
            hop_length=InferenceConfig.HOP_LENGTH,
        )

        # Extract delta and delta-delta
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Combine features
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

        return features


class CoconutMaturityClassifier:
    """ONNX Runtime-based classifier for coconut maturity."""

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        label_map_path: Union[str, Path, None] = None,
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to ONNX model. If None, uses default from config.
            label_map_path: Path to label mapping JSON. If None, uses default from config.
        """
        # Use config defaults if not provided
        if model_path is None:
            model_path = InferenceConfig.get_model_path()
        if label_map_path is None:
            label_map_path = InferenceConfig.get_label_map_path()

        # Convert to Path objects
        model_path = Path(model_path)
        label_map_path = Path(label_map_path)

        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please ensure the model has been trained and exported."
            )
        if not label_map_path.exists():
            raise FileNotFoundError(
                f"Label map file not found: {label_map_path}\n"
                f"Please ensure the model has been trained."
            )

        # Load ONNX model
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Load label mapping
        with open(label_map_path, "r") as f:
            label_data = json.load(f)
            self.label_names = label_data["label_names"]

        self.audio_processor = AudioProcessor()

    def predict_from_file(self, audio_path: Union[str, Path]) -> dict:
        """
        Predict maturity level from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with prediction results
        """
        # Load audio
        audio = self.audio_processor.load_audio_from_file(audio_path)

        # Extract features
        features = self.audio_processor.extract_mfcc_features(audio)

        # Run inference
        return self._run_inference(features)

    def predict_from_binary(self, audio_binary: bytes) -> dict:
        """
        Predict maturity level from binary audio data.

        Args:
            audio_binary: Raw audio bytes

        Returns:
            Dictionary with prediction results
        """
        # Load audio from binary
        audio = self.audio_processor.load_audio_from_binary(audio_binary)

        # Extract features
        features = self.audio_processor.extract_mfcc_features(audio)

        # Run inference
        return self._run_inference(features)

    def predict_from_numpy(
        self, audio_array: np.ndarray, sample_rate: int = InferenceConfig.SAMPLE_RATE
    ) -> dict:
        """
        Predict maturity level from numpy array.

        Args:
            audio_array: Audio signal as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Dictionary with prediction results
        """
        # Resample if needed
        if sample_rate != InferenceConfig.SAMPLE_RATE:
            audio_array = self.audio_processor.load_audio_from_numpy(
                audio_array, sample_rate, InferenceConfig.SAMPLE_RATE
            )

        # Extract features
        features = self.audio_processor.extract_mfcc_features(audio_array)

        # Run inference
        return self._run_inference(features)

    def _run_inference(self, features: np.ndarray) -> dict:
        """
        Run ONNX inference on features.

        Args:
            features: MFCC features

        Returns:
            Dictionary with prediction results
        """
        # Prepare input
        input_data = features[np.newaxis, :, :].astype(np.float32)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        logits = outputs[0][0]

        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        # Get prediction
        predicted_class = int(np.argmax(probabilities))
        predicted_label = self.label_names[predicted_class]
        confidence = float(probabilities[predicted_class])

        # Build result
        result = {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": {
                self.label_names[i]: float(probabilities[i])
                for i in range(len(self.label_names))
            },
        }

        return result


def demo_inference():
    """Demonstration of inference on random sample files."""

    print("=" * 60)
    print("Coconut Maturity Classification - Inference Demo")
    print("=" * 60)

    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = CoconutMaturityClassifier()
    print("✓ Classifier loaded successfully")

    # Define sample directories
    sample_dirs = [
        Path("samples/ridge-a"),
        Path("samples/ridge-b"),
        Path("samples/ridge-c"),
    ]

    # Collect all WAV files
    all_samples = []
    for sample_dir in sample_dirs:
        if sample_dir.exists():
            wav_files = list(sample_dir.glob("*.wav"))
            all_samples.extend(wav_files)

    if not all_samples:
        print("\n⚠ No sample files found in samples/ directories")
        print("Please place WAV files in:")
        for sample_dir in sample_dirs:
            print(f"  - {sample_dir}")
        return

    # Pick a random sample
    sample_file = random.choice(all_samples)

    print(f"\n📁 Selected sample: {sample_file}")
    print(f"   Directory: {sample_file.parent.name}")
    print(f"   Filename: {sample_file.name}")

    # Run inference
    print("\n🔍 Running inference...")
    result = classifier.predict_from_file(sample_file)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Predicted Maturity: {result['predicted_label'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nAll Probabilities:")
    for label, prob in result["probabilities"].items():
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {label:12s} [{bar}] {prob:.2%}")
    print("=" * 60)

    # Demonstrate binary input
    print("\n📦 Demonstrating binary input...")
    with open(sample_file, "rb") as f:
        audio_binary = f.read()

    result_binary = classifier.predict_from_binary(audio_binary)
    print(
        f"✓ Binary inference result: {result_binary['predicted_label']} ({result_binary['confidence']:.2%})"
    )

    # Demonstrate numpy input
    print("\n🔢 Demonstrating numpy array input...")
    audio_array, sr = librosa.load(sample_file, sr=None)
    result_numpy = classifier.predict_from_numpy(audio_array, sr)
    print(
        f"✓ Numpy inference result: {result_numpy['predicted_label']} ({result_numpy['confidence']:.2%})"
    )

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demo_inference()
