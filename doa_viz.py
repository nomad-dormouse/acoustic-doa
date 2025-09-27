import pyaudio
import wave
import numpy as np
import logging
import matplotlib.pyplot as plt
import math
import cv2
from scipy.fft import fft, ifft, fftfreq, rfft, irfft

logging.basicConfig(level=logging.INFO)

c = 343.0  # Speed of sound in m/s
d = 0.125  # Distance between microphones in meters
fs = 48000  # Sampling frequency in Hz
max_shift = int(d / c * fs) + 1  # Maximum shift in samples
shift_values = np.arange(-max_shift, max_shift + 1)

rate = 48000
computations_frequency = 5  # per second
chunk_size = int(rate/computations_frequency)
dtype = pyaudio.paInt16
n_channels = 2
record_seconds = 100
target_frequency_raw = 1000
target_frequency = target_frequency_raw / computations_frequency

def naive_shift(filtered_data):
    scalar_products = np.array([filtered_data[0] @ np.roll(filtered_data[1], shift=i) for i in shift_values])
    shift_raw = np.argmax(scalar_products)
    shift = shift_values[shift_raw]
    return shift

def azimuth_from_shift(shift):
    s = np.clip(c * shift / fs / d, -1, 1)
    theta = math.degrees(math.asin(s))
    return theta

def theta_from_stream(stream):
    raw_data = stream.read(chunk_size)
    data = np.frombuffer(raw_data, dtype=np.int16)
    channel_0 = data[0::n_channels]
    channel_1 = data[1::n_channels]
    channels = np.vstack((channel_0, channel_1))#, channel_2, channel_3))

    # filter
    X = rfft(channels)
    X[:, :int(target_frequency-10)] = 0
    X[:, int(target_frequency + 10):] = 0
    filtered_data = irfft(X, n=channels.shape[1])
    
    # compute shift
    shift = naive_shift(filtered_data)
    theta = azimuth_from_shift(shift)
    logging.info(f"{shift}, {theta}")
    return theta

HFOV = 70.0  # horizontal field of view of your webcam (in degrees)
W = 1280     # width of camera feed
H = 720      # height of camera feed
ARROW_COLOUR = (0, 255, 0)     # green
CONE_COLOUR = (0, 200, 200)    # yellowish
FONT = cv2.FONT_HERSHEY_SIMPLEX
UNCERTAINTY_DEG = 10.0         # DoA uncertainty cone width (± degrees)

def wrap_to_180(angle):
    """Wrap angle to [-180, 180] degrees"""
    return ((angle + 180) % 360) - 180

def azimuth_to_x(azimuth_deg, cam_heading_deg, frame_width, hfov_deg):
    """Convert a world azimuth to x-coordinate in image frame, based on camera yaw"""
    rel_azimuth = wrap_to_180(azimuth_deg - cam_heading_deg)
    if abs(rel_azimuth) > hfov_deg / 2:
        return None  # outside of field of view
    x = int((0.5 + rel_azimuth / hfov_deg) * frame_width)
    return x, rel_azimuth

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

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally to correct the mirror effect
        frame = cv2.flip(frame, 1)

        doa_azimuth = theta_from_stream(stream)
        cam_heading = 0.0

        # Convert azimuth to image x-coord
        result = azimuth_to_x(doa_azimuth, cam_heading, W, HFOV)
        if result is not None:
            x, rel_angle = result

            # Draw uncertainty cone (± degrees)
            left = int((0.5 + (rel_angle - UNCERTAINTY_DEG) / HFOV) * W)
            right = int((0.5 + (rel_angle + UNCERTAINTY_DEG) / HFOV) * W)
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
        else:
            # Out of field of view
            cv2.putText(frame, "Drone out of camera field of view", (10, H - 20), FONT, 0.8, (0,0,255), 2, cv2.LINE_AA)

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