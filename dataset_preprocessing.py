import numpy as np
import librosa
from datasets import load_dataset
import os
import json
from sklearn.model_selection import train_test_split

# Audio preprocessing constants
SR = 16000
N_MELS = 64
N_FFT = 1024
HOP = 256
DUR = 1.0  # seconds per training example

def download_dads_dataset(cache_dir="./data/dads"):
    """Download DADS dataset from Hugging Face"""
    os.makedirs(cache_dir, exist_ok=True)
    return load_dataset("geronimobasso/drone-audio-detection-samples", cache_dir=cache_dir)

def load_and_preprocess(audio_array, sr=SR, dur=DUR):
    """Preprocess audio from numpy array instead of file path"""
    x = audio_array
    if len(x) < int(sr*dur):
        x = np.pad(x, (0, max(0, int(sr*dur) - len(x))))
    x = x[:int(sr*dur)]
    # Optional pre‑emphasis
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    # Mel spectrogram (power)
    S = librosa.feature.melspectrogram(x, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Normalise per-sample
    S_db_norm = (S_db - S_db.mean()) / (S_db.std() + 1e-9)
    return S_db_norm.astype(np.float32)  # shape: (n_mels, time_frames)

def preprocess_dads_sample(sample):
    """Preprocess a single DADS sample"""
    if 'audio' in sample:
        audio_data = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']
    else:
        audio_data = sample['array']
        sample_rate = sample.get('sampling_rate', SR)
    
    audio_data = np.array(audio_data)
    mel_spec = load_and_preprocess(audio_data, sr=sample_rate)
    
    return {
        'mel_spectrogram': mel_spec,
        'label': sample.get('label', 0)
    }

def save_data_to_json(data, save_path):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data_to_save = []
    for sample in data:
        data_to_save.append({
            'mel_spectrogram': sample['mel_spectrogram'].tolist(),
            'label': sample['label']
        })
    
    with open(save_path, 'w') as f:
        json.dump(data_to_save, f)

def split_and_save_data(processed_samples, train_path="./data/dads/train.json", test_path="./data/dads/test.json", test_size=0.2):
    """Split data into train/test and save separately"""
    # Split the data
    train_data, test_data = train_test_split(processed_samples, test_size=test_size, random_state=42, stratify=[s['label'] for s in processed_samples])
    
    # Save both datasets
    save_data_to_json(train_data, train_path)
    save_data_to_json(test_data, test_path)
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

if __name__ == "__main__":
    dataset = download_dads_dataset()
    split_data = dataset[list(dataset.keys())[0]]
    
    processed_samples = []
    for i in range(min(100, len(split_data))):
        sample = split_data[i]
        processed = preprocess_dads_sample(sample)
        processed_samples.append(processed)
    
    split_and_save_data(processed_samples)
