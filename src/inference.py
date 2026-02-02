"""
Coconut Maturity Classification Inference Script
Supports ONNX Runtime inference with audio format conversion utilities
"""

import argparse
import json
import random
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning, module="numba")
warnings.filterwarnings("ignore", category=FutureWarning)


class AudioConverter:
    """Utility class for converting various audio formats to binary format"""

    @staticmethod
    def load_audio(
        audio_path: str, target_sr: int = 44100, duration: Optional[float] = 3.0
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to target sample rate

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (default: 44100)
            duration: Duration to load in seconds (default: 3.0)

        Returns:
            audio: Audio signal as numpy array
            sr: Sample rate
        """
        # Load audio with librosa (handles most formats)
        audio, sr = librosa.load(audio_path, sr=target_sr, duration=duration)
        return audio, sr

    @staticmethod
    def convert_to_wav(
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 44100,
    ) -> str:
        """
        Convert audio file to WAV format

        Args:
            input_path: Path to input audio file
            output_path: Path to output WAV file (optional)
            sample_rate: Target sample rate

        Returns:
            output_path: Path to output WAV file
        """
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.with_suffix(".wav")

        # Load and convert
        audio, sr = AudioConverter.load_audio(str(input_path), target_sr=sample_rate)

        # Save as WAV
        sf.write(str(output_path), audio, sr)

        return str(output_path)

    @staticmethod
    def audio_to_binary(audio: np.ndarray, dtype: np.dtype = np.float32) -> bytes:
        """
        Convert audio array to binary format

        Args:
            audio: Audio signal as numpy array
            dtype: Data type for conversion

        Returns:
            Binary representation of audio
        """
        return audio.astype(dtype).tobytes()

    @staticmethod
    def binary_to_audio(binary_data: bytes, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Convert binary data back to audio array

        Args:
            binary_data: Binary audio data
            dtype: Data type for conversion

        Returns:
            Audio signal as numpy array
        """
        return np.frombuffer(binary_data, dtype=dtype)

    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio


class CoconutClassifier:
    """Coconut maturity classifier using ONNX Runtime"""

    def __init__(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        n_mfcc: int = 40,
        sample_rate: int = 44100,
        max_len: int = 130,
    ):
        """
        Initialize classifier

        Args:
            model_path: Path to ONNX model
            metadata_path: Path to metadata JSON (optional)
            n_mfcc: Number of MFCC coefficients
            sample_rate: Audio sample rate
            max_len: Maximum length for MFCC features
        """
        self.model_path = model_path
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.max_len = max_len

        # Load ONNX model
        self.session = ort.InferenceSession(model_path)

        # Load metadata if provided
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.label_names = metadata.get("label_names", ["im", "m", "om"])
                self.n_mfcc = metadata.get("n_mfcc", n_mfcc)
                self.max_len = metadata.get("max_len", max_len)
        else:
            self.label_names = ["im", "m", "om"]  # premature, mature, overmature

        print(f"Loaded model: {model_path}")
        print(f"Labels: {self.label_names}")

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio

        Args:
            audio: Audio signal

        Returns:
            MFCC features
        """
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)

        # Pad or truncate to fixed length
        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, : self.max_len]

        return mfcc

    def predict(
        self, audio: Union[str, np.ndarray, bytes], return_probs: bool = False
    ) -> Union[str, Tuple[str, np.ndarray]]:
        """
        Predict coconut maturity from audio

        Args:
            audio: Audio file path, numpy array, or binary data
            return_probs: Whether to return class probabilities

        Returns:
            Predicted class label (and probabilities if return_probs=True)
        """
        # Handle different input types
        if isinstance(audio, str):
            # Load from file
            audio_data, _ = AudioConverter.load_audio(audio, target_sr=self.sample_rate)
        elif isinstance(audio, bytes):
            # Convert from binary
            audio_data = AudioConverter.binary_to_audio(audio)
        else:
            # Already numpy array
            audio_data = audio

        # Normalize audio
        audio_data = AudioConverter.normalize_audio(audio_data)

        # Extract features
        features = self.extract_features(audio_data)

        # Prepare input for ONNX
        input_data = features[np.newaxis, :, :].astype(np.float32)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        outputs = self.session.run([output_name], {input_name: input_data})[0]

        # Get prediction
        probs = self._softmax(outputs[0])
        pred_idx = np.argmax(probs)
        pred_label = self.label_names[pred_idx]

        if return_probs:
            return pred_label, probs
        return pred_label

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def predict_batch(self, audio_list: list, return_probs: bool = False) -> list:
        """
        Predict maturity for multiple audio samples

        Args:
            audio_list: List of audio files/arrays
            return_probs: Whether to return class probabilities

        Returns:
            List of predictions
        """
        results = []
        for audio in audio_list:
            result = self.predict(audio, return_probs=return_probs)
            results.append(result)
        return results


def get_random_sample_file(samples_dir: str = "samples") -> Optional[str]:
    """
    Pick a random WAV file from samples/ridge-{a,b,c} directories

    Args:
        samples_dir: Base directory for samples

    Returns:
        Path to random sample file or None
    """
    samples_path = Path(samples_dir)

    # Look for ridge directories
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

    # Pick random file
    random_file = random.choice(all_wav_files)
    return str(random_file)


def main():
    parser = argparse.ArgumentParser(
        description="Coconut maturity classification inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/coconut_classifier.onnx",
        help="Path to ONNX model (default: models/coconut_classifier.onnx)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="models/coconut_classifier_metadata.json",
        help="Path to metadata JSON (default: models/coconut_classifier_metadata.json)",
    )
    parser.add_argument("--audio", type=str, help="Path to audio file for inference")
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="samples",
        help="Directory containing sample files (default: samples)",
    )
    parser.add_argument(
        "--random", action="store_true", help="Use random sample from samples directory"
    )
    parser.add_argument("--convert", type=str, help="Convert audio file to WAV format")
    parser.add_argument("--output", type=str, help="Output path for converted audio")
    parser.add_argument(
        "--show_probs", action="store_true", help="Show class probabilities"
    )

    args = parser.parse_args()

    # Handle audio conversion
    if args.convert:
        print(f"Converting {args.convert} to WAV format...")
        output_path = AudioConverter.convert_to_wav(
            args.convert, output_path=args.output
        )
        print(f"✓ Saved to: {output_path}")
        return

    # Initialize classifier
    print("=" * 60)
    print("COCONUT MATURITY CLASSIFIER")
    print("=" * 60)

    classifier = CoconutClassifier(
        model_path=args.model,
        metadata_path=args.metadata if Path(args.metadata).exists() else None,
    )

    # Determine audio file to use
    audio_file = None

    if args.random:
        # Use random sample
        audio_file = get_random_sample_file(args.samples_dir)
        if audio_file:
            print(f"\nUsing random sample: {audio_file}")
    elif args.audio:
        # Use specified file
        audio_file = args.audio
        print(f"\nUsing specified file: {audio_file}")
    else:
        print("\nError: Please specify --audio or use --random flag")
        return

    if not audio_file or not Path(audio_file).exists():
        print(f"Error: Audio file not found: {audio_file}")
        return

    # Run inference
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

    # Decode label
    label_mapping = {"im": "Premature (immature)", "m": "Mature", "om": "Overmature"}

    print(f"Classification: {label_mapping.get(pred_label, pred_label)}")

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
