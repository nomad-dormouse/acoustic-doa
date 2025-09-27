import librosa
import soundfile as sf
import numpy as np
import argparse
from pathlib import Path

def segment_audio(input_file="data/shahed/shahed.wav", segment_length=2, interval=1, output_dir="data/shahed/segments"):
    """
    Split audio into segments of specified length at specified intervals.
    
    Args:
        input_file: Path to input WAV file
        segment_length: Length of each segment in seconds
        interval: Interval between segment starts in seconds
        output_dir: Directory to save segments
    """
    # Load audio
    y, sr = librosa.load(input_file, sr=None)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate samples per segment and interval
    segment_samples = int(segment_length * sr)
    interval_samples = int(interval * sr)
    
    # Generate segments
    segment_num = 0
    start_sample = 0
    
    while start_sample + segment_samples <= len(y):
        end_sample = start_sample + segment_samples
        segment = y[start_sample:end_sample]
        
        # Save segment
        output_file = Path(output_dir) / f"{Path(input_file).stem}_segment_{segment_num:03d}.wav"
        sf.write(output_file, segment, sr)
        
        print(f"Saved: {output_file}")
        segment_num += 1
        start_sample += interval_samples
    
    print(f"Created {segment_num} segments")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split audio into segments")
    parser.add_argument("--input_file", default="data/shahed/shahed.wav", help="Input WAV file (default: data/shahed/shahed.wav)")
    parser.add_argument("--length", type=float, default=2, help="Segment length in seconds (default: 2)")
    parser.add_argument("--interval", type=float, default=1, help="Interval between segments in seconds (default: 1)")
    parser.add_argument("--output", default="data/shahed/segments", help="Output directory (default: data/shahed/segments)")
    
    args = parser.parse_args()
    
    segment_audio(args.input_file, args.length, args.interval, args.output)
