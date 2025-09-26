import cv2
import numpy as np
import math
import time

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

start = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simulated input — replace this with real DoA
        t = time.time() - start
        doa_azimuth = (30.0 * math.sin(0.4 * t) + 90.0) % 360  # simulate 60–120° sweep
        cam_heading = 90.0  # assume camera facing East; change if needed

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
            label = f"Azimuth: {doa_azimuth:.1f}° (Rel: {rel_angle:+.1f}°)"
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