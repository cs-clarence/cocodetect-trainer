"""
inference.py
Load ONNX model and predict coconut maturity from an audio input.
Supports:
  - path to .wav/.flac file
  - binary audio data read from stdin / bytes

Example:
  python inference.py --onnx models/best_epoch_10.onnx --file samples/ridge-a/c85.wav
  cat some.wav | python inference.py --onnx models/best.onnx --stdin

Outputs: predicted label and probabilities.
"""
import argparse
import io
import numpy as np
import onnxruntime as rt
import librosa
import soundfile as sf

LABEL_INV = {0: "premature", 1: "mature", 2: "overmature"}
SAMPLE_RATE = 44100

def bytes_to_wave(b: bytes, sr=SAMPLE_RATE, mono=True):
    """
    Convert raw audio bytes (wav/flac/mp3 if libsndfile supports) to numpy float32 mono waveform.
    """
    bio = io.BytesIO(b)
    data, fs = sf.read(bio, dtype='float32')
    if fs != sr:
        data = librosa.resample(data.T if data.ndim > 1 else data, orig_sr=fs, target_sr=sr)
        data = data.T if data.ndim > 1 else data
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32)

def path_to_wave(path, sr=SAMPLE_RATE):
    data, fs = sf.read(path, dtype='float32')
    if fs != sr:
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = librosa.resample(data, orig_sr=fs, target_sr=sr)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32)

def compute_mfcc_for_inference(wave, sr=SAMPLE_RATE, n_mfcc=40, hop_length=512, n_fft=2048, target_time=None):
    mfcc = librosa.feature.mfcc(y=wave.astype(np.float32), sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    # ensure same dimensions as training by padding or truncating time axis
    if target_time is not None:
        T = target_time
        if mfcc.shape[1] < T:
            pad = np.zeros((mfcc.shape[0], T - mfcc.shape[1]), dtype=np.float32)
            mfcc = np.concatenate([mfcc, pad], axis=1)
        else:
            mfcc = mfcc[:, :T]
    return mfcc.astype(np.float32)

def run_inference(onnx_path, waveform=None, mfcc=None, n_mfcc=40, target_time=None):
    sess = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    if mfcc is None:
        mfcc = compute_mfcc_for_inference(waveform, n_mfcc=n_mfcc, target_time=target_time)
    # model expects (B, n_mfcc, T)
    inp = mfcc[np.newaxis, :, :].astype(np.float32)
    outputs = sess.run(None, {input_name: inp})
    logits = outputs[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred = int(np.argmax(probs))
    return pred, probs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--file", help="path to audio file (.wav/.flac). If provided, will load it.")
    p.add_argument("--stdin", action="store_true", help="read raw audio bytes from stdin")
    p.add_argument("--n_mfcc", type=int, default=40)
    p.add_argument("--target_time", type=int, default=None, help="time frames to pad/truncate mfcc to (optional). If omitted, uses mfcc length from input.")
    args = p.parse_args()

    if args.stdin:
        import sys
        b = sys.stdin.buffer.read()
        wave = bytes_to_wave(b)
    elif args.file:
        wave = path_to_wave(args.file)
    else:
        raise ValueError("Either --file or --stdin must be given")

    # If target_time not given, compute a temporary mfcc to get time dimension
    mfcc0 = librosa.feature.mfcc(y=wave.astype(np.float32), sr=SAMPLE_RATE, n_mfcc=args.n_mfcc)
    T = mfcc0.shape[1] if args.target_time is None else args.target_time
    pred, probs = run_inference(args.onnx, waveform=wave, n_mfcc=args.n_mfcc, target_time=T)
    print("Prediction:", LABEL_INV[pred])
    print("Probs:", {LABEL_INV[i]: float(probs[i]) for i in range(len(probs))})

if __name__ == "__main__":
    main()
