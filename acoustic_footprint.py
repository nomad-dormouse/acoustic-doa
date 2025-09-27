import numpy as np
import librosa
import plotly.graph_objects as go
import plotly.subplots as sp
import webbrowser
import os
import argparse
from pathlib import Path

def create_acoustic_footprint(filename):
    # Load audio (WAV file - librosa will use soundfile backend)
    y, sr = librosa.load(filename, sr=None)

    # Time vector for waveform
    time_waveform = np.arange(len(y)) / sr

    # Generate spectrogram
    n_fft = 2048
    hop_length = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Time and frequency axes for spectrogram
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Calculate frequency spectrum (average over time)
    freq_spectrum = np.mean(S_db, axis=1)

    # Create subplot figure
    fig = sp.make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.15,
        subplot_titles=("Frequency Spectrum", "Waveform", "Spectrogram (dB)"),
        row_heights=[0.3, 0.3, 0.4]
    )

    # Plot frequency spectrum (Amplitude vs Frequency)
    fig.add_trace(
        go.Scatter(x=freqs, y=freq_spectrum, mode='lines', name='Frequency Spectrum'),
        row=1, col=1
    )

    # Plot waveform
    fig.add_trace(
        go.Scatter(x=time_waveform, y=y, mode='lines', name='Waveform'),
        row=2, col=1
    )

    # Plot spectrogram as heatmap
    fig.add_trace(
        go.Heatmap(
            z=S_db,
            x=times,
            y=freqs,
            colorscale='Viridis',
            zmin=-100, zmax=0,
            showscale=False
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        showlegend=False,
        xaxis_title='Frequency (Hz)',
        yaxis_title='Amplitude (dB)',
        xaxis2_title='Time (s)',
        yaxis2_title='Amplitude (normalized)',
        xaxis3_title='Time (s)',
        yaxis3_title='Frequency (Hz)',
    )

    # Limit y-axis of spectrogram (optional)
    fig.update_yaxes(range=[0, 8000], row=3, col=1)

    # Limit x-axis of frequency spectrum to focus on audible range
    fig.update_xaxes(range=[0, 8000], row=1, col=1)

    # Align y-axis titles for all plots
    fig.update_yaxes(title_standoff=25, row=1, col=1)
    fig.update_yaxes(title_standoff=15, row=2, col=1)
    fig.update_yaxes(title_standoff=20, row=3, col=1)

    # Generate output filename
    input_path = Path(filename)
    output_filename = f"{input_path.stem}_acoustic_footprint.html"
    output_path = Path("outputs/plotly") / output_filename
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save plot as HTML file and open in browser
    fig.write_html(str(output_path))
    webbrowser.open(f"file://{os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create acoustic footprint visualization")
    parser.add_argument("input_file", help="Input WAV file")
    
    args = parser.parse_args()
    create_acoustic_footprint(args.input_file)