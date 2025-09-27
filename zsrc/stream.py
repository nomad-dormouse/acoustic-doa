# pip install sounddevice numpy
import time

import sounddevice as sd

from utils import azimuth_from_tau, bandpass_filter, gcc_phat_delay
from config import SR, D, C, FILT_F, FILT_BW, W_LIVE

def callback(indata, frames, timeinfo, status):
    if status: print(status)
    y = bandpass_filter(indata, SR, f0=FILT_F, bandwidth=FILT_BW)
    ch0, ch1 = y[:,0], y[:,1]
    
    tau, _ = gcc_phat_delay(ch0, ch1, SR, max_tau=D/C)
    theta = azimuth_from_tau(tau)
    print(f"τ={tau*1e6:7.1f} µs  →  θ≈{theta:6.1f}°", end="\r")

if __name__ == "__main__":
    device = None # choose your Windows "Microphone Array" device index if needed

    with sd.InputStream(device=device, channels=2, samplerate=SR, blocksize=W_LIVE, callback=callback):
        print("Listening… Ctrl+C to stop.")
        while True: time.sleep(2000.0)
