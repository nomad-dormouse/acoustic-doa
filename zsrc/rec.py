import os

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import sounddevice as sd

from config import FILT_F, FILT_BW, SECONDS, SR, W_REC
from utils import (
    bandpass_filter,
    compute_doa_properties,
    magnitude_spectrum,
    to_db,
)

pio.renderers.default = os.environ.get("PLOTLY_RENDERER", "browser")


def record_stereo(seconds=5.0, channels=2, samplerate=SR):
    print(f"Recording {seconds} s of stereo at {samplerate} Hz…")
    x = sd.rec(
        frames = int(seconds * samplerate),
        samplerate=samplerate,
        channels=channels,
        dtype='float32',
        blocking=True,
    )
    return x, samplerate


def plot_frequency_spectrum(x, samplerate):
    freqs, mag = magnitude_spectrum(x, samplerate)
    mag_db = to_db(mag)
    print(mag_db.shape, freqs.shape)
    
    fig = go.Figure()
    for i in range(mag_db.shape[0]):
        fig.add_trace(go.Scatter(x=freqs, y=mag_db[i], mode='lines', name=f'CH{i}'))
    
    fig.update_xaxes(range=[0, 5000])
    
    fig.update_layout(
        title='Frequency Decomposition (Magnitude Spectrum)',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude (dB, rel. full scale)',
        hovermode='x unified',
    )
    
    return fig

def plot_time_domain_waveforms(x, samplerate, max_seconds=None):
    if max_seconds is not None:
        n_show = min(x.shape[0], int(max_seconds * samplerate))
    else:
        n_show = x.shape[0]
    
    t = np.arange(n_show) / float(samplerate)
    
    fig = go.Figure()
    for i in range(x.shape[1]):
        fig.add_trace(go.Scatter(x=t, y=x[:n_show, i], mode='lines', name=f'CH{i}'))
    
    fig.update_layout(
        title='Raw Time-Domain Waveforms',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (relative)',
        hovermode='x unified'
    )
    
    return fig

def plot_doa_analysis(x, samplerate, window_size=W_REC):
    # Compute DOA properties for the first two channels (stereo pair)
    time_centers, lags, taus, thetas, ccs = compute_doa_properties(
        x[:, :2],
        samplerate,
        window_size=window_size,
    )
    
    # Create subplot figure for DOA properties (5 rows: time domain + 4 DOA plots)
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=('Time Domain Waveforms', 'Sample Lag', 'Time Delay (τ) in µs', 'Azimuth Angle (θ) in degrees', 'Cross-Correlation'),
        vertical_spacing=0.05
    )
    
    # Plot time domain waveforms (first subplot)
    n_show = min(x.shape[0], int(SECONDS * samplerate))
    t = np.arange(n_show) / float(samplerate)
    for i in range(x.shape[1]):
        fig.add_trace(
            go.Scatter(x=t, y=x[:n_show, i], mode='lines', name=f'CH{i}'),
            row=1, col=1
        )
    
    # Plot sample lag
    fig.add_trace(
        go.Scatter(x=time_centers, y=lags, mode='lines+markers', 
                  name='Sample Lag', line=dict(color='blue')),
        row=2, col=1
    )
    
    # Plot time delay in microseconds
    fig.add_trace(
        go.Scatter(x=time_centers, y=taus*1e6, mode='lines+markers', 
                  name='Time Delay (µs)', line=dict(color='red')),
        row=3, col=1
    )
    
    # Plot azimuth angle
    fig.add_trace(
        go.Scatter(x=time_centers, y=thetas, mode='lines+markers', 
                  name='Azimuth (deg)', line=dict(color='green')),
        row=4, col=1
    )
    
    # Plot cross-correlation
    fig.add_trace(
        go.Scatter(x=time_centers, y=ccs, mode='lines+markers', 
                  name='Cross-Correlation', line=dict(color='purple')),
        row=5, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='DOA Analysis - Time Domain and Windowed Properties',
        height=1200,
        showlegend=False
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Time (s)", row=5, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Samples", row=2, col=1)
    fig.update_yaxes(title_text="Time Delay (µs)", row=3, col=1)
    fig.update_yaxes(title_text="Angle (degrees)", row=4, col=1)
    fig.update_yaxes(title_text="CC Value", row=5, col=1)
    
    print(f"DOA Analysis: {len(time_centers)} windows processed")
    print(f"Average τ: {np.mean(taus)*1e6:.1f} µs, Average θ: {np.mean(thetas):.1f}°")
    
    return fig


def main():
    # Load input either from WAV (if path is provided) or record live
    x, samplerate = record_stereo(seconds=SECONDS, channels=2, samplerate=SR)
    x_filtered = bandpass_filter(x, samplerate, f0=FILT_F, bandwidth=FILT_BW)
    print(f"Captured audio: shape={x.shape}  sr={samplerate}")

    # Create and show all plots
    fig_freq = plot_frequency_spectrum(x, samplerate)
    fig_freq_filtered = plot_frequency_spectrum(x_filtered, samplerate)
    fig_doa = plot_doa_analysis(x_filtered, samplerate, window_size=W_REC)  # Now includes time domain plot

    # Show all figures
    fig_freq.show()
    fig_freq_filtered.show()
    fig_doa.show()

if __name__ == "__main__":
    main()
