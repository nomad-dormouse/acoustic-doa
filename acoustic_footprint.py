import numpy as np
import librosa
import plotly.graph_objects as go
import plotly.subplots as sp

# Load audio
filename = 'shahed_drone.mp4'  # replace with your file
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

# Create subplot figure
fig = sp.make_subplots(
    rows=2, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.15,
    subplot_titles=("Waveform", "Spectrogram (dB)"),
    row_heights=[0.3, 0.7]
)

# Plot waveform
fig.add_trace(
    go.Scatter(x=time_waveform, y=y, mode='lines', name='Waveform'),
    row=1, col=1
)

# Plot spectrogram as heatmap
fig.add_trace(
    go.Heatmap(
        z=S_db,
        x=times,
        y=freqs,
        colorscale='Viridis',
        zmin=-100, zmax=0,
        colorbar=dict(title='Power (dB)')
    ),
    row=2, col=1
)

# Update layout
fig.update_layout(
    height=600,
    width=1000,
    title='Audio Analysis',
    xaxis_title='Time (s)',
    yaxis_title='Amplitude',
    xaxis2_title='Time (s)',
    yaxis2_title='Frequency (Hz)',
)

# Limit y-axis of spectrogram (optional)
fig.update_yaxes(range=[0, 8000], row=2, col=1)

# Show plot
fig.show()