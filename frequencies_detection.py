import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
from pathlib import Path

def extract_frequency_peaks(filename, min_prominence=1, min_distance=50):
    """
    Extract dominant frequency peaks from audio file.
    
    Args:
        filename: Path to WAV file
        min_prominence: Minimum prominence for peak detection
        min_distance: Minimum distance between peaks (Hz)
    
    Returns:
        peaks: Array of peak frequencies
        magnitudes: Array of peak magnitudes
    """
    # Load audio
    y, sr = librosa.load(filename, sr=None)
    
    print(f"  Audio: {len(y)} samples at {sr} Hz")
    print(f"  Duration: {len(y)/sr:.2f} seconds")
    print(f"  RMS energy: {np.sqrt(np.mean(y**2)):.6f}")
    
    # Generate frequency spectrum
    n_fft = 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Average spectrum over time
    avg_spectrum = np.mean(S, axis=1)
    
    # Limit frequencies to 4000 Hz
    max_freq_idx = np.where(freqs <= 4000)[0][-1]
    freqs = freqs[:max_freq_idx + 1]
    avg_spectrum = avg_spectrum[:max_freq_idx + 1]
    
    print(f"  Spectrum max: {np.max(avg_spectrum):.6f}")
    print(f"  Spectrum mean: {np.mean(avg_spectrum):.6f}")
    print(f"  Frequency range: 0 - {freqs[-1]:.1f} Hz")
    
    # Find peaks in frequency domain
    distance_samples = max(1, int(min_distance * len(freqs) / (sr/2)))
    peaks, properties = find_peaks(
        avg_spectrum, 
        prominence=min_prominence,
        distance=distance_samples
    )
    
    # Convert to frequency values
    peak_freqs = freqs[peaks]
    peak_mags = avg_spectrum[peaks]
    
    if len(peaks) == 0:
        # Try with lower prominence to see if there are any peaks at all
        for test_prominence in [0.1, 0.5, 1, 2]:
            test_peaks, _ = find_peaks(
                avg_spectrum, 
                prominence=test_prominence,
                distance=10
            )
            if len(test_peaks) > 0:
                print(f"  Found {len(test_peaks)} peaks with prominence {test_prominence}")
                break
    
    return peak_freqs, peak_mags, avg_spectrum, freqs

def check_harmonics(peak_freqs, tolerance=0.05):
    """
    Check if frequency peaks form harmonic series.
    
    Args:
        peak_freqs: Array of peak frequencies
        tolerance: Relative tolerance for harmonic matching (5% default)
    
    Returns:
        fundamental: Estimated fundamental frequency
        harmonics: List of harmonic relationships found
    """
    if len(peak_freqs) < 2:
        return None, []
    
    # Try each peak as potential fundamental
    best_fundamental = None
    best_harmonic_count = 0
    best_harmonics = []
    
    for i, candidate_fund in enumerate(peak_freqs):
        harmonics = []
        
        # Check for harmonics (2f, 3f, 4f, etc.)
        for peak_freq in peak_freqs:
            if peak_freq == candidate_fund:
                harmonics.append((peak_freq, 1, 0))  # Fundamental
                continue
                
            # Check harmonic ratios
            ratio = peak_freq / candidate_fund
            harmonic_num = round(ratio)
            
            if harmonic_num >= 2:  # Skip sub-harmonics
                expected_freq = candidate_fund * harmonic_num
                error = abs(peak_freq - expected_freq) / expected_freq
                
                if error <= tolerance:
                    harmonics.append((peak_freq, harmonic_num, error * 100))
        
        # Score this fundamental
        if len(harmonics) > best_harmonic_count:
            best_harmonic_count = len(harmonics)
            best_fundamental = candidate_fund
            best_harmonics = harmonics
    
    return best_fundamental, best_harmonics


def analyze_audio(filename, prominence=1):
    """
    Analyze a single audio file for harmonic patterns.
    """
    print(f"\nAnalyzing: {filename}")
    
    # Extract peaks
    peak_freqs, peak_mags, spectrum, freqs = extract_frequency_peaks(filename, prominence)
    
    if len(peak_freqs) == 0:
        print("  No significant peaks found")
        return None
    
    # Print top 5 peaks sorted by frequency value
    top_indices = np.argsort(peak_mags)[-5:][::-1]
    top_freqs = peak_freqs[top_indices]
    top_mags = peak_mags[top_indices]
    
    # Sort by frequency value
    sorted_indices = np.argsort(top_freqs)
    
    print("  Top 5 frequency peaks (sorted by frequency):")
    for idx in sorted_indices:
        print(f"    {top_freqs[idx]:.1f} Hz (magnitude: {top_mags[idx]:.1f})")
    
    # Check for harmonics
    fundamental, harmonics = check_harmonics(peak_freqs)
    
    if fundamental:
        print(f"  🎵 HARMONIC DETECTED!")
        print(f"  Fundamental: {fundamental:.1f} Hz")
        print("  Harmonic series:")
        for freq, harmonic_num, error in sorted(harmonics, key=lambda x: x[1]):
            print(f"    {harmonic_num}f = {freq:.1f} Hz (error: {error:.1f}%)")
        
        return {
            'file': Path(filename).name,
            'fundamental': fundamental,
            'harmonics': harmonics,
            'peak_count': len(peak_freqs)
        }
    else:
        print("  No clear harmonic pattern found")
        return None

def analyze_batch(segments_dir, pattern="*segment*.wav", prominence=1):
    """
    Analyze multiple audio segments for harmonic patterns using analyze_audio.
    """
    segments_path = Path(segments_dir)
    segment_files = sorted(segments_path.glob(pattern))
    
    results = []
    
    for segment_file in segment_files:
        result = analyze_audio(segment_file, prominence)
        if result:
            results.append(result)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect harmonic patterns in audio segments")
    parser.add_argument("input_path", help="Directory containing audio segments or single audio file")
    parser.add_argument("--prominence", type=float, default=1, help="Minimum peak prominence")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Harmonic tolerance (0.05 = 5%)")
    parser.add_argument("--batch", action="store_true", help="Process directory in batch mode")
    
    args = parser.parse_args()
    
    if args.batch:
        # Directory analysis using analyze_batch
        results = analyze_batch(args.input_path, prominence=args.prominence)
        
        print(f"\n📊 SUMMARY:")
        print(f"Segments with harmonics: {len(results)}")
        if results:
            print("Fundamental frequencies detected:")
            for result in results:
                print(f"  {result['file']}: {result['fundamental']:.1f} Hz ({result['peak_count']} peaks)")
    else:
        # Single file analysis (default behavior)
        result = analyze_audio(args.input_path, args.prominence)
        if result:
            print(f"\n📊 RESULT:")
            print(f"File: {result['file']}")
            print(f"Fundamental: {result['fundamental']:.1f} Hz")
            print(f"Peaks found: {result['peak_count']}")
