# Acoustic Direction of Arrival (DOA) Project

Real-time drone detection and localization using acoustic Direction of Arrival (DOA) estimation with dual-microphone arrays.

## Setup

```bash
brew install portaudio ffmpeg
uv sync
```

## Use

- `acoustic_footprint.py` - Generate frequency spectrum, waveform, and spectrogram visualizations
- `audio_segmentation.py` - Split chosen audio into segments segments of chosen lenght and with chosen interval
- `frequencies_detection.py` - Detect harmonic patterns and fundamental frequencies in audio segments
- `doa_visualisation.py` - Real-time DOA estimation with dual-microphone array and camera overlay

## Audio Processing with FFmpeg

```bash
# Convert video to audio
ffmpeg -i data/shahed/shahec.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 1 data/shahed/shahed.wav

# Crop audio files
ffmpeg -ss 00:00:00 -to 00:00:02 -i data/shahed/shahed.wav -vn -acodec pcm_s16le data/shahed/shahed_0to2s.wav

# Increase volume
ffmpeg -i data/shahed/shahed.wav -acodec pcm_s16le -ar 44100 -ac 1 -af "volume=5.0" data/shahed/shahed_volume_x5.wav
```
