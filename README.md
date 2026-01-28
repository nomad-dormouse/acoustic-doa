# Acoustic Direction of Arrival (DOA) Project

Acoustic Direction of Arrival (DOA) estimation for Shahed drones using a dual‑microphone array and signal processing.

**Live Demo:** [Watch the demonstration](https://bit.ly/acoustic_doa_demo)  
**Presentation:** [View slides](https://bit.ly/acoustic_doa_slides)

## Setup

```bash
brew install portaudio ffmpeg
uv sync
```

## Use

- `acoustic_footprint.py` - Generate frequency spectrum, waveform, and spectrogram visualizations for a WAV file
- `audio_segmentation.py` - Split an audio file into fixed-length, overlapping segments
- `frequencies_detection.py` - Detect harmonic patterns and estimate fundamental frequencies in audio (single file or batch)
- `doa_visualisation.py` - Real-time DOA estimation with a dual-microphone array and webcam overlay

## Project Overview

This project explores acoustic Direction of Arrival (DOA) estimation for Shahed drones by combining offline audio analysis with real-time microphone-array processing and visualisation.  
The workflow covers:

- **Preprocessing and segmentation** of drone recordings into short, partially overlapping windows for easier analysis.
- **Spectral and time–frequency analysis** to understand the acoustic footprint (waveform, spectrum, and spectrogram).
- **Harmonic pattern detection** to characterise drone engine tones and their harmonics.
- **Real-time DOA estimation** from a two-microphone array, visualised directly on a live camera feed.

## Technical Approach

- **Acoustic footprint visualisation (`acoustic_footprint.py`)**
  - Loads WAV audio with `librosa`, computes a Short-Time Fourier Transform (STFT), and limits analysis to \(\leq 4\,\text{kHz}\).
  - Aggregates the spectrogram over time to build an average frequency spectrum.
  - Renders an interactive Plotly dashboard with three aligned views: frequency spectrum, raw waveform, and spectrogram, saved as a standalone HTML report and opened in a browser.

- **Audio segmentation (`audio_segmentation.py`)**
  - Uses `librosa` and `soundfile` to cut long recordings into fixed-duration segments with configurable hop/interval.
  - Writes numbered segment files (e.g. `*_segment_000.wav`) suitable for batch analysis of different parts of a drone fly‑by.

- **Frequency and harmonic analysis (`frequencies_detection.py`)**
  - Computes an STFT-based magnitude spectrum, averaged over time and truncated to 0–4 kHz.
  - Uses `scipy.signal.find_peaks` to detect dominant spectral peaks, then searches for harmonic relationships between them (e.g. \(2f, 3f, 4f\)).
  - For single files and batches of segments, reports candidate fundamental frequencies, harmonic series, and peak statistics to help identify stable drone tone patterns.

- **Real-time DOA estimation and visualisation (`doa_visualisation.py`)**
  - Captures a stereo audio stream via `pyaudio` at 48 kHz from a two-microphone array with known spacing (\(d = 0.1\,\text{m}\)).
  - In each processing chunk, applies an FFT, performs simple band-pass filtering around selected harmonics, and then computes the relative shift between channels via time-domain correlation (`naive_shift`).
  - Estimates the inter-microphone time delay \(\tau\) as the lag (in samples) that maximises the cross-correlation between the two band-limited channels, and smooths it over several chunks to reduce jitter.
  - Interprets that delay as a path-length difference \(\Delta L = c \tau\) between microphones and converts it to azimuth using the far-field plane-wave model \(\Delta L = d \sin(\theta)\), giving \(\theta = \arcsin\left(\frac{c \tau}{d}\right)\) with values clamped for numerical stability.
  - Maps the resulting azimuth angle into the camera’s horizontal field of view so that left/right direction in the audio corresponds to horizontal position in the video frame.
  - Uses OpenCV (`cv2`) to overlay:
    - a vertical arrow indicating the current DOA angle,
    - an uncertainty cone (\(\pm\) configurable degrees) across the camera frame,
    - a confidence bar derived from recent correlation (overlap) values.
  - Runs continuously until the user exits, providing an intuitive live display of the acoustic direction of arrival.

## Audio

```bash
# Convert video to audio
ffmpeg -i data/shahed/shahec.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 1 data/shahed/shahed.wav

# Crop audio files
ffmpeg -ss 00:00:00 -to 00:00:02 -i data/shahed/shahed.wav -vn -acodec pcm_s16le data/shahed/shahed_0to2s.wav

# Increase volume
ffmpeg -i data/shahed/shahed.wav -acodec pcm_s16le -ar 44100 -ac 1 -af "volume=5.0" data/shahed/shahed_volume_x5.wav
```
