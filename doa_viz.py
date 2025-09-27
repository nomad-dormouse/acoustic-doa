import pyaudio
import wave
import numpy as np
import logging
from collections import deque
import matplotlib.pyplot as plt
import math
import cv2
from scipy.fft import fft, ifft, fftfreq, rfft, irfft

logging.basicConfig(level=logging.INFO)

c = 343.0  # Speed of sound in m/s
d = 0.1  # Distance between microphones in meters
fs = 48000  # Sampling frequency in Hz
max_shift = int(d / c * fs) + 1  # Maximum shift in samples
shift_values = np.arange(-max_shift, max_shift + 1)

rate = fs
computations_frequency = 5  # per second
chunk_size = int(rate/computations_frequency)
dtype = pyaudio.paInt16
n_channels = 2
record_seconds = 100
target_frequency_raw = 1000
target_frequency = target_frequency_raw / computations_frequency

# camera setup
HFOV = 70.0  # horizontal field of view of your webcam (in degrees)
W = 1280     # width of camera feed
H = 720      # height of camera feed
ARROW_COLOUR = (0, 255, 0)     # green
CONE_COLOUR = (0, 200, 200)    # yellowish
FONT = cv2.FONT_HERSHEY_SIMPLEX
UNCERTAINTY_DEG = 10.0         # DoA uncertainty cone width (± degrees)

def naive_shift(data):
    data_0 = data[0] / np.linalg.norm(data[0])
    data_1 = data[1] / np.linalg.norm(data[1])
    scalar_products = np.array([data_0 @ np.roll(data_1, shift=i) for i in shift_values])
    shift_raw = np.argmax(scalar_products)
    shift = shift_values[shift_raw]

    rolled_values = np.roll(scalar_products, -(shift_raw-3))[:7]
    confidence = np.max(rolled_values) -np.mean(rolled_values)
    return shift, confidence

def box_filter(X, filtration_values, filtration_width):
    X_filtered = X.copy()
    filtration_intervals = []
    running_index = 0
    for val in filtration_values:
        filtration_intervals.append((running_index, val-filtration_width))
        running_index = val + filtration_width
    filtration_intervals.append((running_index, X.shape[1]))

    for interval in filtration_intervals:
        X_filtered[:, interval[0]:interval[1]] = 0

    return X_filtered

def azimuth_from_shift(shift):
    s = np.clip(c * shift / fs / d, -1, 1)
    theta = math.degrees(math.asin(s))
    return theta

def theta_from_stream(stream, shift_history):
    raw_data = stream.read(chunk_size)
    data = np.frombuffer(raw_data, dtype=np.int16)
    channel_0 = data[0::n_channels]
    channel_1 = data[1::n_channels]
    channels = np.vstack((channel_0, channel_1))#, channel_2, channel_3))

    # filter
    X = rfft(channels)
    filtered_X = box_filter(X, filtration_values=[235, 470, 705, 940, 1175, 1410], filtration_width=40)
    filtered_data = irfft(filtered_X, n=channels.shape[1])
    
    # compute shift
    shift, confidence = naive_shift(filtered_data)
    shift_history.append(shift)
    smooth_shift = int(np.round(np.mean(shift_history)))
    theta = azimuth_from_shift(smooth_shift)
    logging.info(f"{smooth_shift}, {theta}")
    return theta, shift_history

def wrap_to_180(angle):
    """Wrap angle to [-180, 180] degrees"""
    return ((angle + 180) % 360) - 180

def azimuth_to_x(azimuth_deg):
    """Convert a world azimuth to x-coordinate in image frame"""
    return int((0.5 + azimuth_deg / HFOV) * W)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

p = pyaudio.PyAudio()
stream = p.open(format=dtype,
                channels=n_channels,
                rate=rate,
                input=True,
                # input_device_index=2,
                frames_per_buffer=chunk_size)
logging.info("* recording")

shift_history = deque(maxlen=5)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally to correct the mirror effect
        frame = cv2.flip(frame, 1)

        doa_azimuth, shift_history = theta_from_stream(stream, shift_history)
        x = azimuth_to_x(doa_azimuth)

        # Draw uncertainty cone (± degrees)
        left = azimuth_to_x(doa_azimuth - UNCERTAINTY_DEG)
        right = azimuth_to_x(doa_azimuth + UNCERTAINTY_DEG)
        left = max(0, left)
        right = min(W - 1, right)
        cv2.rectangle(frame, (left, 0), (right, H), CONE_COLOUR, thickness=2)

        # Draw DoA arrow (vertical)
        arrow_y = H // 2
        arrow_len = 100
        cv2.arrowedLine(
            frame,
            (x, arrow_y + arrow_len // 2),
            (x, arrow_y - arrow_len),
            ARROW_COLOUR, thickness=4, tipLength=0.3
        )

        # Draw text
        label = f"Azimuth: {doa_azimuth:6.1f}"
        cv2.putText(frame, label, (10, H - 20), FONT, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow('Azimuth DoA Overlay', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    stream.stop_stream()
    stream.close()
    p.terminate()