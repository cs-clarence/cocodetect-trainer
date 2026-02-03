#!/usr/bin/env python3
"""
Coconut Maturity Classification Inference Script (PyTorch backend)
Replaces ONNX runtime with direct PyTorch model loading and inference.
"""

import argparse
import json
import random
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import CoconutCNN

warnings.filterwarnings("ignore", category=UserWarning, module="numba")
warnings.filterwarnings("ignore", category=FutureWarning)


class AudioConverter:
    """Utility class for converting various audio formats to binary format"""

    @staticmethod
    def load_audio(
        audio_path: str, target_sr: int = 44100, duration: Optional[float] = None, offset: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to target sample rate

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (default: 44100)
            duration: Duration to load in seconds (default: None -> load full file)
            offset: start reading at this offset (seconds)
        Returns:
            audio: Audio signal as numpy array
            sr: Sample rate
        """
        audio = None
        sr = target_sr

        # Prefer soundfile for robust reading
        try:
            import soundfile as sf

            # sf.read reads entire file; we handle duration via slicing if requested
            data, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            # If multi-channel, mix to mono
            if np.ndim(data) > 1:
                data = np.mean(data, axis=1)
            # If user requested a duration, slice accordingly (use offset)
            if duration is not None:
                start_sample = int(offset * file_sr)
                end_sample = start_sample + int(duration * file_sr)
                data = data[start_sample:end_sample]
            # Resample if needed
            if file_sr != target_sr:
                data = librosa.resample(np.asarray(data, dtype=np.float32), orig_sr=file_sr, target_sr=target_sr)
                sr = target_sr
            else:
                sr = file_sr
            audio = data
        except Exception:
            # Fallback to librosa.load if soundfile failed for some reason
            try:
                audio, sr = librosa.load(audio_path, sr=target_sr, duration=duration, offset=offset)
            except Exception as e:
                # Final fallback: return empty array and raise helpful message
                raise RuntimeError(f"Failed to read audio '{audio_path}' with soundfile and librosa: {e}")

        # Sanitize audio (fix NaNs/Infs, collapse channels, clip, etc.)
        audio = AudioConverter.sanitize_audio(audio)

        return audio, sr

    @staticmethod
    def convert_to_wav(
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 44100,
    ) -> str:
        """
        Convert audio file to WAV format
        """
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.with_suffix(".wav")

        # Load and convert
        audio, sr = AudioConverter.load_audio(str(input_path), target_sr=sample_rate, duration=None)

        # Save as WAV
        sf.write(str(output_path), audio, sr)

        return str(output_path)
    
    @staticmethod
    def audio_to_binary(audio: np.ndarray, dtype: np.dtype = np.float32) -> bytes:
        return audio.astype(dtype).tobytes()

    @staticmethod
    def binary_to_audio(binary_data: bytes, dtype: np.dtype = np.float32) -> np.ndarray:
        return np.frombuffer(binary_data, dtype=dtype)

    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    @staticmethod
    def sanitize_audio(audio: np.ndarray) -> np.ndarray:
        """Ensure audio contains finite values and is float32 in safe range."""
        if audio is None:
            return np.zeros(0, dtype=np.float32)
        # Convert to 1-D
        audio = np.asarray(audio)
        if audio.ndim > 1:
            # collapse multi-channel to mono
            audio = np.mean(audio, axis=1)

        # If any non-finite values exist, replace them and warn
        if not np.isfinite(audio).all():
            # Report
            num_bad = np.size(audio) - np.count_nonzero(np.isfinite(audio))
            warnings.warn(f"Audio contained non-finite values ({num_bad}); replacing NaN/Inf with finite numbers.", UserWarning)
            # Replace NaN with 0.0; posinf/neginf with large finite numbers (then scale/clip)
            audio = np.nan_to_num(
                audio,
                nan=0.0,
                posinf=np.finfo(np.float32).max / 100.0,
                neginf=np.finfo(np.float32).min / 100.0,
            )

        # Cast and clip to reasonable dynamic range for audio
        audio = audio.astype(np.float32)
        # If signal is silent / constant zero, leave it
        if np.max(np.abs(audio)) > 0:
            # Optionally normalize here or leave to classifier.normalize_audio
            # But ensure values not wildly huge
            audio = np.clip(audio, -1.0, 1.0)
        return audio


# --- Classifier using PyTorch for inference ---
class CoconutClassifier:
    """Coconut maturity classifier using PyTorch"""

    def __init__(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        n_mfcc: int = 40,
        sample_rate: int = 44100,
        max_len: int = 130,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.max_len = max_len

        # device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load metadata if provided
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.label_names = metadata.get("label_names", ["im", "m", "om"])
                self.n_mfcc = metadata.get("n_mfcc", self.n_mfcc)
                self.max_len = metadata.get("max_len", self.max_len)
                num_classes = metadata.get("num_classes", len(self.label_names))
                dropout = metadata.get("dropout", 0.5)
        else:
            # fallback defaults
            self.label_names = ["im", "m", "om"]
            num_classes = 3
            dropout = 0.5

        # Instantiate model architecture
        self.model = CoconutCNN(n_mfcc=self.n_mfcc, num_classes=num_classes, dropout=dropout, use_recurrent=None)
        self.model.to(self.device)

        # Load checkpoint (support saved dicts or full model)
        self._load_checkpoint(self.model_path)

        self.model.eval()
        print(f"Loaded PyTorch model: {model_path} -> device: {self.device}")
        print(f"Labels: {self.label_names}")

    def _load_checkpoint(self, path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = torch.load(str(path), map_location=self.device)

        # If a dict with 'model_state_dict' key (from your training save)
        if isinstance(data, dict) and "model_state_dict" in data:
            state = data["model_state_dict"]
            try:
                self.model.load_state_dict(state)
            except Exception as e:
                # try strict=False in case of minor mismatch
                self.model.load_state_dict(state, strict=False)
        elif isinstance(data, dict) and all(k in data for k in ("state_dict",)):
            # alternate key
            self.model.load_state_dict(data["state_dict"])
        elif isinstance(data, nn.Module):
            # saved entire model
            self.model = data.to(self.device)
        else:
            # maybe the file is a raw state_dict
            try:
                self.model.load_state_dict(data)
            except Exception as e:
                raise RuntimeError(f"Unable to load model checkpoint from {path}: {e}")

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, : self.max_len]
        return mfcc

    def predict_batch(self, audio_list: list, return_probs: bool = False) -> list:
        results = []
        for audio in audio_list:
            result = self.predict(audio, return_probs=return_probs)
            results.append(result)
        return results
    
    def predict(
        self,
        audio: Union[str, np.ndarray, bytes],
        return_probs: bool = False,
        infer_strategy: str = "single",  # "single" or "sliding"
        window_seconds: float = 3.0,
        hop_seconds: Optional[float] = None,
        aggregate: str = "mean",  # "mean", "max", "majority"
    ) -> Union[str, Tuple[str, np.ndarray]]:
        """
        Predict coconut maturity from audio.

        Args:
            audio: Audio file path, numpy array, or binary data
            return_probs: Whether to return class probabilities
            infer_strategy: "single" (default) or "sliding"
            window_seconds: window length in seconds used for sliding windows
            hop_seconds: hop length in seconds between windows. If None and sliding, hop = window_seconds/2
            aggregate: aggregation method over window predictions: "mean", "max", or "majority"
        """
        # Handle different input types
        if isinstance(audio, str):
            # Load full file (duration controlled by caller or default None)
            audio_data, _ = AudioConverter.load_audio(audio, target_sr=self.sample_rate, duration=None)
        elif isinstance(audio, bytes):
            # Convert from binary then sanitize
            audio_data = AudioConverter.binary_to_audio(audio)
            audio_data = AudioConverter.sanitize_audio(audio_data)
        else:
            # Already numpy array: sanitize (fix NaN/Inf, collapse channels if needed)
            audio_data = AudioConverter.sanitize_audio(np.asarray(audio))

        # Normalize audio
        audio_data = AudioConverter.normalize_audio(audio_data)

        # If empty or too short, handle gracefully
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Empty audio data provided to predict()")

        # Helper to get tensor for a given audio segment (numpy array)
        def segment_to_tensor(seg_audio: np.ndarray) -> torch.Tensor:
            mfcc = self.extract_features(seg_audio)  # shape (n_mfcc, frames)
            # If MFCC frames < max_len, pad; else truncate in extract_features already
            inp = torch.from_numpy(mfcc.astype(np.float32))[None, :, :].to(self.device)
            return inp

        # Single-pass inference: compute MFCC on whole audio and pad/truncate to max_len
        if infer_strategy == "single":
            inp = segment_to_tensor(audio_data)
            with torch.no_grad():
                outputs = self.model(inp)
                probs_t = F.softmax(outputs, dim=1)
            probs = probs_t.cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = self.label_names[pred_idx] if pred_idx < len(self.label_names) else str(pred_idx)
            if return_probs:
                return pred_label, probs
            return pred_label

        # Sliding-window inference
        elif infer_strategy == "sliding":
            # derive frames/sec for MFCC: approx frames = len(audio) / hop_length_in_samples
            # But simpler and robust: convert window_seconds directly to sample counts and MFCC frames implicitly
            hop_seconds = hop_seconds if hop_seconds is not None else (window_seconds / 2.0)
            window_samples = int(window_seconds * self.sample_rate)
            hop_samples = int(hop_seconds * self.sample_rate)

            # If audio shorter than window, pad the audio to window length
            if len(audio_data) <= window_samples:
                # pad
                pad_len = window_samples - len(audio_data)
                audio_data_padded = np.pad(audio_data, (0, pad_len), mode="constant")
                windows = [audio_data_padded]
            else:
                # build sliding windows
                windows = []
                start = 0
                while start < len(audio_data):
                    end = start + window_samples
                    if end <= len(audio_data):
                        win = audio_data[start:end]
                    else:
                        # last window: pad to full length
                        win = np.pad(audio_data[start:], (0, end - len(audio_data)), mode="constant")
                    windows.append(win)
                    start += hop_samples

            # Run model on every window and collect probs
            all_probs = []
            with torch.no_grad():
                for win in windows:
                    inp = segment_to_tensor(win)
                    out = self.model(inp)
                    p = F.softmax(out, dim=1).cpu().numpy()[0]
                    all_probs.append(p)

            all_probs = np.stack(all_probs, axis=0)  # (num_windows, num_classes)

            if aggregate == "mean":
                agg_probs = np.mean(all_probs, axis=0)
            elif aggregate == "max":
                # pick the window with the maximum confidence (max of maxes), then return that window's probs
                window_scores = np.max(all_probs, axis=1)  # max prob per window
                idx = int(np.argmax(window_scores))
                agg_probs = all_probs[idx]
            elif aggregate == "majority":
                # majority vote on argmax across windows
                argmaxes = np.argmax(all_probs, axis=1)
                vals, counts = np.unique(argmaxes, return_counts=True)
                winner = int(vals[np.argmax(counts)])
                # create one-hot-like probs for winner
                agg_probs = np.zeros(all_probs.shape[1], dtype=float)
                agg_probs[winner] = 1.0
            else:
                raise ValueError(f"Unknown aggregate method: {aggregate}")

            pred_idx = int(np.argmax(agg_probs))
            pred_label = self.label_names[pred_idx] if pred_idx < len(self.label_names) else str(pred_idx)

            if return_probs:
                return pred_label, agg_probs
            return pred_label

        else:
            raise ValueError(f"Unknown infer_strategy: {infer_strategy}")


def get_random_sample_file(samples_dir: str = "samples") -> Optional[str]:
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


def main():
    parser = argparse.ArgumentParser(description="Coconut maturity classification inference (PyTorch backend)")
    parser.add_argument(
        "--model",
        type=str,
        default="models/coconut_classifier.pth",
        help="Path to PyTorch .pth model (default: models/coconut_classifier.pth)",
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
    parser.add_argument("--random", action="store_true", help="Use random sample from samples directory")
    parser.add_argument("--convert", type=str, help="Convert audio file to WAV format")
    parser.add_argument("--output", type=str, help="Output path for converted audio")
    parser.add_argument("--show_probs", action="store_true", help="Show class probabilities")
    parser.add_argument("--device", type=str, default=None, help="torch device (e.g. cpu or cuda)")
    parser.add_argument(
        "--infer_strategy",
        type=str,
        choices=["single", "sliding"],
        default="single",
        help="Inference strategy for variable-length audio: single or sliding (default: single)",
    )
    parser.add_argument(
        "--window_seconds",
        type=float,
        default=3.0,
        help="Window length in seconds for sliding-window inference (default: 3.0)",
    )
    parser.add_argument(
        "--hop_seconds",
        type=float,
        default=None,
        help="Hop length (seconds) between sliding windows. If omitted, defaults to window_seconds/2",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["mean", "max", "majority"],
        default="mean",
        help="Aggregation method across sliding-window predictions (default: mean)",
    )
    parser.add_argument(
        "--full_audio",
        action="store_true",
        help="Load the full audio file (override default duration). Often used with sliding strategy.",
    )

    args = parser.parse_args()

    # Handle audio conversion
    if args.convert:
        print(f"Converting {args.convert} to WAV format...")
        output_path = AudioConverter.convert_to_wav(args.convert, output_path=args.output)
        print(f"✓ Saved to: {output_path}")
        return

    print("=" * 60)
    print("COCONUT MATURITY CLASSIFIER (PyTorch)")
    print("=" * 60)

    classifier = CoconutClassifier(
        model_path=args.model,
        metadata_path=args.metadata if Path(args.metadata).exists() else None,
        device=args.device,
    )

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
        pred_label, probs = classifier.predict(
            audio_file,
            return_probs=True,
            infer_strategy=args.infer_strategy,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            aggregate=args.aggregate,
        )
        print(f"\nPredicted maturity: {pred_label}")
        print("\nClass probabilities:")
        for label, prob in zip(classifier.label_names, probs):
            print(f"  {label:>3s}: {prob:.4f} ({prob * 100:.2f}%)")
    else:
        pred_label = classifier.predict(
            audio_file,
            infer_strategy=args.infer_strategy,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            aggregate=args.aggregate,
        )
        print(f"\nPredicted maturity: {pred_label}")

    label_mapping = {"im": "Premature (immature)", "m": "Mature", "om": "Overmature"}
    print(f"Classification: {label_mapping.get(pred_label, pred_label)}")

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
