import math
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.fft import rfftfreq, rfft, irfft
import plotly.graph_objects as go

from config import D, C, W_REC

def gcc_phat_delay(
    x,
    y,
    sr,
    max_tau,
    time_center=1,
    use_fft=False,
    plot_between_times=(3,3.1),
):
    """
    Return TDOA in seconds and cross-correlation value using GCC-PHAT with parabolic sub-sample refinement.

    """
    max_shift_samples = int(max_tau * sr)
    num_samples = len(x)
    
    if use_fft:
        X = np.fft.fft(x)
        Y = np.fft.fft(y)
        R = X * np.conj(Y)
        R_mag = np.abs(R)
        R_mag[R_mag < 1e-12] = 1e-12  # avoid division by zero
        R = R / R_mag
        
        # Compute cross-correlation via inverse FFT
        cc = np.fft.ifft(R).real  
    
        # Create search indices: [0, 1, 2, ..., max_shift] and [n-max_shift, ..., n-1]
        positive_indices = np.arange(max_shift_samples + 1)
        negative_indices = np.arange(max(0, num_samples - max_shift_samples), num_samples)
        search_indices = np.concatenate([positive_indices, negative_indices])
        cc_search_values = cc[search_indices]

        peak_search_idx = np.argmax(np.abs(cc_search_values))
        global_peak_idx = search_indices[peak_search_idx]
        
        # Get the cross-correlation value at the peak
        cc_value = cc[global_peak_idx]
        if global_peak_idx <= num_samples // 2:
            lag_samples = global_peak_idx
        else:
            lag_samples = global_peak_idx - num_samples
    else:
        cc_values = []
        norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
        for shift in np.arange(-max_shift_samples,max_shift_samples+1):
            cc_values.append(np.roll(x, shift = shift)@y / norm)
        cc = np.array(cc_values)
        lag_samples = np.argmax(np.abs(cc))
        cc_value = cc[lag_samples]

        if lag_samples <= max_shift_samples:
            signed_lag_samples = lag_samples
        else:
            signed_lag_samples = lag_samples - (2*max_shift_samples + 1)

    if plot_between_times[0] <= time_center <= plot_between_times[1]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(cc)), y=cc, mode='lines'))
        fig.update_layout(title=f"{time_center}")
        fig.show()
    return signed_lag_samples / sr, cc_value

def azimuth_from_tau(tau, d=D, c=C):
    s = c * tau / d
    theta = math.degrees(math.asin(s))
    return theta

def bandpass_filter(signal, samplerate, f0, bandwidth, order=4, square=False):
    low = max(1.0, f0 - bandwidth/2.0)
    high = min(samplerate/2 - 1.0, f0 + bandwidth/2.0)
    mult = signal.shape[0] / samplerate
    if square:
        def bandpass_filter_1d(signal):
            f = rfft(signal)
            f[:int(low*mult)] = 0
            f[int(high*mult):] = 0
            return irfft(f)
    else:
        def bandpass_filter_1d(signal):
            sos = butter(order, [low, high], btype='band', fs=samplerate, output='sos')
            return sosfiltfilt(sos, signal, axis=0)

    return np.apply_along_axis(bandpass_filter_1d, axis=0, arr=signal)

def magnitude_spectrum(x, samplerate):
    n_samples, n_channels = x.shape

    fft = np.array([rfft(x[:, i]) for i in range(n_channels)])

    mag = np.abs(fft)
    freqs = rfftfreq(mag.shape[1], d=1.0 / samplerate)
    num_positive_freqs = int(n_samples // 4)
    print(freqs, freqs[:num_positive_freqs])
    return freqs[:num_positive_freqs], mag[:, :num_positive_freqs]

def to_db(mag, floor_db=-120.0):
    # magnitude to dBFS-ish scale (relative to full scale 1.0)
    mag_capped = np.maximum(mag, 1e-12)
    db = 20.0 * np.log10(mag_capped)
    db = np.maximum(db, floor_db)
    return db

def compute_doa_properties(x, samplerate, window_size=W_REC, d=D, c=C):
    n_samples, n_channels = x.shape
    
    # Calculate number of windows
    n_windows = n_samples // window_size
    
    if n_windows == 0:
        raise ValueError(f"Audio data too short for window size {window_size}")
    
    # Initialize arrays
    time_centers = np.zeros(n_windows)
    lags = np.zeros(n_windows)
    taus = np.zeros(n_windows)
    thetas = np.zeros(n_windows)
    ccs = np.zeros(n_windows)
    
    # Process each window
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        
        # Extract window data for both channels
        ch0 = x[start_idx:end_idx, 0]
        ch1 = x[start_idx:end_idx, 1]
        
        # Calculate time center of this window
        time_centers[i] = (start_idx + window_size // 2) / samplerate
        
        # Compute TDOA using GCC-PHAT and get cross-correlation
        max_tau = d / c  # maximum possible time delay
        tau, cc_value = gcc_phat_delay(ch0, ch1, samplerate, max_tau, time_centers[i])
        
        # Calculate sample lag
        lag = tau * samplerate
        
        # Calculate azimuth angle
        theta = azimuth_from_tau(tau, d=d, c=c)
        
        # Store results
        lags[i] = lag
        taus[i] = tau
        thetas[i] = theta
        ccs[i] = cc_value
    
    return time_centers, lags, taus, thetas, ccs
