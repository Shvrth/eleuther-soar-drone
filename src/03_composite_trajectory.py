import cv2
import numpy as np
import os
import subprocess

# --- CONFIGURATION ---
INPAINTED_VIDEO = "FINAL_INPAINTED_DRONE_SHOT.mp4"
ASSET_IMAGE = "data/ferrari.png"
COORD_FILE = "car_coordinates.txt"
OUTPUT_VIDEO = "FINAL_FERRARI_SHOT.mp4"

def smooth_data_heavy(data_array, window_size=21):
    pad_size = window_size // 2
    padded = np.pad(data_array, (pad_size, pad_size), mode='edge')
    return np.convolve(padded, np.ones(window_size)/window_size, mode='valid')

def rotate_asset_safely(image, angle_degrees):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle_degrees, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def main():
    if not os.path.exists(COORD_FILE):
        raise FileNotFoundError(f"Missing {COORD_FILE}! Run script 01 first.")

    raw_f_idx, raw_cx, raw_cy, raw_b_angles = [], [], [], []
    
    with open(COORD_FILE, "r") as f:
        next(f)
        for line in f:
            f_idx, cx, cy, b_angle = line.strip().split(",")
            raw_f_idx.append(int(f_idx))
            raw_cx.append(int(cx) * 0.5) # Scale to 360p background
            raw_cy.append(int(cy) * 0.5)
            raw_b_angles.append(float(b_angle))

    # --- ANTI-FLIP LOGIC ---
    aligned_angles = []
    prev_angle = None
    for i, curr_angle in enumerate(raw_b_angles):
        if prev_angle is None:
            lookahead = min(i + 15, len(raw_f_idx) - 1)
            dx, dy = raw_cx[lookahead] - raw_cx[i], raw_cy[lookahead] - raw_cy[i]
            motion_angle = 0 if (dx == 0 and dy == 0) else np.degrees(np.arctan2(dy, dx))
            if 90 < ((motion_angle - curr_angle) % 360) < 270:
                curr_angle = (curr_angle + 180) % 360
        else:
            diff1 = abs((curr_angle - prev_angle + 180) % 360 - 180)
            curr_angle_flipped = (curr_angle + 180) % 360
            diff2 = abs((curr_angle_flipped - prev_angle + 180) % 360 - 180)
            if diff2 < diff1:
                curr_angle = curr_angle_flipped
                
        aligned_angles.append(curr_angle)
        prev_angle = curr_angle

    # --- SMOOTHING ---
    smooth_cx = smooth_data_heavy(np.array(raw_cx), window_size=21)
    smooth_cy = smooth_data_heavy(np.array(raw_cy), window_size=21)
    trajectory = {raw_f_idx[i]: (smooth_cx[i], smooth_cy[i]) for i in range(len(raw_f_idx))}

    rad_angles = np.radians(aligned_angles)
    unwrapped_rads = np.unwrap(rad_angles)
    smoothed_rads = smooth_data_heavy(unwrapped_rads, window_size=15)
    final_angles = {raw_f_idx[i]: np.degrees(smoothed_rads[i]) for i in range(len(raw_f_idx))}

    # --- VIDEO COMPOSITING ---
    base_asset = cv2.imread(ASSET_IMAGE, cv2.IMREAD_UNCHANGED)
    asset_size = 60
    base_asset = cv2.resize(base_asset, (asset_size, asset_size))

    cap = cv2.VideoCapture(INPAINTED_VIDEO)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("Compositing target asset onto trajectory...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_idx in trajectory:
            cx, cy = trajectory[frame_idx]
            oriented_asset = rotate_asset_safely(base_asset, final_angles.get(frame_idx, 0.0))
            
            asset_bgr = oriented_asset[:, :, :3]
            asset_alpha = oriented_asset[:, :, 3] / 255.0  
            
            pos_x, pos_y = int(cx) - (asset_size // 2), int(cy) - (asset_size // 2)
            
            y1, y2 = max(0, pos_y), min(height, pos_y + asset_size)
            x1, x2 = max(0, pos_x), min(width, pos_x + asset_size)

            ay1 = 0 if pos_y >= 0 else -pos_y
            ay2 = asset_size if (pos_y + asset_size) <= height else asset_size - ((pos_y + asset_size) - height)
            ax1 = 0 if pos_x >= 0 else -pos_x
            ax2 = asset_size if (pos_x + asset_size) <= width else asset_size - ((pos_x + asset_size) - width)

            if y1 < y2 and x1 < x2:
                roi = frame[y1:y2, x1:x2]
                roi_alpha = asset_alpha[ay1:ay2, ax1:ax2]
                roi_bgr = asset_bgr[ay1:ay2, ax1:ax2]
                
                for c in range(3): 
                    roi[:, :, c] = (roi_alpha * roi_bgr[:, :, c] + (1 - roi_alpha) * roi[:, :, c])
                frame[y1:y2, x1:x2] = roi

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
        
    print(f"\n--- Pipeline Complete! Generated {OUTPUT_VIDEO} ---")

if __name__ == "__main__":
    main()