import cv2
import numpy as np
import math
import time

# CONFIG
HFOV = 70.0  # camera horizontal FOV in degrees — change to your webcam spec
VFOV = 43.0  # vertical FOV (optional)
W = 1280
H = 720
arrow_color = (0, 255, 0)  # BGR, green
cone_color = (0, 200, 200)
conf_font = cv2.FONT_HERSHEY_SIMPLEX

def wrap_to_180(angle):
    return ((angle + 180) % 360) - 180

def bearing_to_x(bearing_world, cam_yaw, W, HFOV):
    bearing_rel = wrap_to_180(bearing_world - cam_yaw)
    half = HFOV / 2.0
    if bearing_rel < -half or bearing_rel > half:
        return None  # outside FOV
    x = int((0.5 + bearing_rel / HFOV) * W)
    return x, bearing_rel

# Camera capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

start = time.time()
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simulated inputs (replace with real sensor inputs)
        t = time.time() - start
        # Simulate a world bearing that slowly moves 0->360
        bearing_world = (30.0 * math.sin(0.3 * t) + 90.0) % 360
        cam_yaw = 90.0  # assume camera faces East; replace with real camera heading

        # Convert bearing to image x
        result = bearing_to_x(bearing_world, cam_yaw, W, HFOV)
        if result is not None:
            x, bearing_rel = result
            # draw cone for uncertainty (±10 degrees)
            uncertainty_deg = 12.0
            # compute left/right x for cone edges
            left = int((0.5 + (bearing_rel - uncertainty_deg) / HFOV) * W)
            right = int((0.5 + (bearing_rel + uncertainty_deg) / HFOV) * W)
            cv2.rectangle(frame, (max(0, left), 0), (min(W-1, right), H), cone_color, thickness=2)
            # draw arrow at centre
            arrow_y = H // 2
            arrow_len = 80
            cv2.arrowedLine(frame, (x, arrow_y+arrow_len//2), (x, arrow_y - arrow_len), arrow_color, 5, tipLength=0.3)
            # draw label
            label = f"Bearing: {bearing_world:.1f}°  Rel: {bearing_rel:.1f}°"
            cv2.putText(frame, label, (10, H - 20), conf_font, 0.8, (255,255,255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Target outside HFOV", (10, H - 20), conf_font, 0.8, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow('Camera with DoA overlay', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()