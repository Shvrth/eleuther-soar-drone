import os
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
TARGET_TRACK_ID = 1
INPUT_VIDEO = "data/drone_clip_two.mp4" 
OUTPUT_DIR = "tracked_output"
YOLO_WEIGHTS = "models/top_view_drone_yolo_11_weights.pt"
SAM2_CHECKPOINT = "models/sam2_hiera_tiny.pt"
SAM2_CONFIG = "sam2_hiera_t.yaml"

os.makedirs(OUTPUT_DIR, exist_ok=True)
COORD_FILE_PATH = "car_coordinates.txt"
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "sam2_drone_clip_two.mp4")
MASK_VIDEO_PATH = "mask_video.mp4"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing tracking on device: {device}")

    # Load Models
    yolo_model = YOLO(YOLO_WEIGHTS)
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    coord_file = open(COORD_FILE_PATH, "w")
    coord_file.write("frame_idx,center_x,center_y,body_angle\n")

    # Initialize Video Stream
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {INPUT_VIDEO}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    mask_writer = cv2.VideoWriter(MASK_VIDEO_PATH, fourcc, fps, (width, height))

    print(f"Processing video. Segmenting Target ID: {TARGET_TRACK_ID}...")
    
    frame_idx = 0
    target_mean = None

    custom_botsort = {
        "tracker_type": "botsort", "track_high_thresh": 0.25, "track_low_thresh": 0.05,
        "new_track_thresh": 0.3, "track_buffer": 60, "match_thresh": 0.8, "fuse_score": True,
        "gmc_method": "sparseOptFlow", "proximity_thresh": 0.5, "appearance_thresh": 0.25, "with_reid": False
    }
    with open("gmc_botsort.yaml", "w") as f:
        yaml.safe_dump(custom_botsort, f)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- Auto-Exposure Stabilization ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        current_mean = np.mean(v)
        
        if target_mean is None:
            target_mean = current_mean
            
        brightness_ratio = target_mean / (current_mean + 1e-5) 
        v_stabilized = np.clip(v * brightness_ratio, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(cv2.merge((h, s, v_stabilized)), cv2.COLOR_HSV2BGR)

        # --- YOLO Tracking ---
        yolo_results = yolo_model.track(
            source=frame, persist=True, tracker="gmc_botsort.yaml", conf=0.2, iou=0.65, imgsz=1280, verbose=False
        )[0]

        binary_mask_frame = np.zeros_like(frame, dtype=np.uint8)

        if yolo_results.boxes is None or yolo_results.boxes.id is None:
            video_writer.write(frame)
            mask_writer.write(binary_mask_frame)
            frame_idx += 1
            continue

        bboxes = yolo_results.boxes.xyxy.cpu().numpy()
        track_ids = yolo_results.boxes.id.cpu().numpy().astype(int)
        sam_image_initialized = False

        for bbox, track_id in zip(bboxes, track_ids):
            if track_id != TARGET_TRACK_ID: continue

            # --- SAM 2 Segmentation ---
            if not sam_image_initialized:
                sam2_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                sam_image_initialized = True

            masks, _, _ = sam2_predictor.predict(box=np.array(bbox), multimask_output=False)

            best_mask = masks[0]
            if torch.is_tensor(best_mask):
                best_mask = best_mask.cpu().numpy()
            best_mask = np.squeeze(best_mask).astype(bool) 

            # Visualization Blending
            frame[best_mask] = (frame[best_mask] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
            binary_mask_frame[best_mask] = [255, 255, 255]
            
            # Extract Rotated Bounding Box for physical angle
            contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                car_contour = max(contours, key=cv2.contourArea)
                (cx, cy), (rect_w, rect_h), angle = cv2.minAreaRect(car_contour)
                body_angle = angle + 90 if rect_w < rect_h else angle
            else:
                cx, cy, body_angle = 0, 0, 0
            
            coord_file.write(f"{frame_idx},{int(cx)},{int(cy)},{body_angle:.2f}\n")

            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"TARGET ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        video_writer.write(frame)
        mask_writer.write(binary_mask_frame)
        
        frame_idx += 1
        if frame_idx % 100 == 0: print(f"Processed {frame_idx} frames...")

    cap.release()
    video_writer.release()
    mask_writer.release()
    coord_file.close()
    print(f"\n--- Tracking Finished! Data saved to {OUTPUT_VIDEO_PATH} and {COORD_FILE_PATH} ---")

if __name__ == "__main__":
    main()