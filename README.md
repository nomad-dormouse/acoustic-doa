# Acoustic Direction of Arrival (DOA) Project

This project focuses on acoustic analysis and Direction of Arrival (DOA) estimation for drone detection.

## Setup

1. Install system dependencies:
```bash
brew install portaudio ffmpeg
```

2. Install Python dependencies using `uv`:
```bash
uv sync
```

## Audio Conversion

To convert MP4 audio files to WAV format:

```bash
ffmpeg -i data/shahed_drone.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 1 data/shahed_drone.wav
```

This command:
- `-i input.mp4` - Input MP4 file
- `-vn` - No video (audio only)
- `-acodec pcm_s16le` - 16-bit PCM audio codec
- `-ar 44100` - Sample rate of 44.1 kHz
- `-ac 1` - Mono channel
- `output.wav` - Output WAV file

## Usage

- `acoustic_footprint.py` - Audio visualization and analysis
- `azymuth_visualisation.py` - Real-time DOA visualization with camera overlay
- `notebooks/` - Jupyter notebooks for analysis
