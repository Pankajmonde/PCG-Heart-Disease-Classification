import os
import numpy as np
import librosa

def load_pcg_spectrogram(file_path):
    """
    Load and preprocess a PhysioNet PCG spectrogram.
    
    Parameters:
    - file_path: Path to the spectrogram file.
    
    Returns:
    - normalized_spectrogram: Normalized spectrogram data.
    """
    try:
        # Load the spectrogram
        spectrogram, sr = librosa.load(file_path, sr=None)
        
        # Normalize the spectrogram
        normalized_spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
        
        return normalized_spectrogram
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Example usage: load a spectrogram
    sample_file_path = 'path_to_spectrogram.wav'  # Change to your file path
    spectrogram = load_pcg_spectrogram(sample_file_path)
    print("Loaded and normalized spectrogram:", spectrogram)