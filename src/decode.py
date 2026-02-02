import os

import numpy as np
import pandas as pd
from scipy.io import wavfile

# Configuration based on dataset specs
SAMPLING_RATE = 44100  # 44.1 kHz
INPUT_FILE = "coconut_acoustic_signals.xlsx"  # Ensure the file name is correct
SHEETS = ["Ridge A", "Ridge B", "Ridge C"]


def decode_coconut_audio(
    input_file="coconut_acoustic_signals.xlsx", sheets=["Ridge A", "Ridge B", "Ridge C"]
):
    # 1. Load the Excel file
    try:
        excel_data = pd.ExcelFile(input_file)
    except FileNotFoundError:
        print(
            f"Error: {input_file} not found. Please ensure the file is in the same directory."
        )
        return

    for sheet_name in sheets:
        # 2. Create the structured folder (e.g., ridge-a)
        folder_name = f"samples/{sheet_name.lower().replace(' ', '-')}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        print(f"Processing {sheet_name}...")

        # Load the specific sheet
        df = pd.read_excel(excel_data, sheet_name=sheet_name)

        # 3. Iterate through each column (sample)
        for column_name in df.columns:
            # Get the 132,300 data points for this sample
            signal_data = df[column_name].values

            # 4. Normalization
            # The values are very small (e.g., 0.00003). To hear them,
            # we scale them so the peak value is 1.0 (float32 standard)
            max_val = np.max(np.abs(signal_data))
            if max_val > 0:
                normalized_signal = signal_data / max_val
            else:
                normalized_signal = signal_data

            # 5. Save as a .wav file
            # Using float32 format which maintains high fidelity
            file_path = os.path.join(folder_name, f"{column_name}.wav")
            wavfile.write(
                file_path, SAMPLING_RATE, normalized_signal.astype(np.float32)
            )

    print("\nDecoding complete! Check your folders for the .wav files.")


if __name__ == "__main__":
    decode_coconut_audio(input_file=INPUT_FILE, sheets=SHEETS)
