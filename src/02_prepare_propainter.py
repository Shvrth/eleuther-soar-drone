import os
import glob
import cv2
import shutil
import subprocess
import numpy as np 

# --- CONFIGURATION ---
INPUT_VIDEO = "data/drone_clip_two.mp4"
MASK_VIDEO = "mask_video.mp4"
PROPAINTER_DIR = "ProPainter" 

def clear_and_create_dirs(paths):
    for path in paths:
        if os.path.exists(path): shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

def main():
    print("--- Phase 1: Directory Setup & Frame Extraction ---")
    base_dirs = ["video_frames", "mask_frames", "part1_vid", "part1_mask", "part2_vid", "part2_mask", "final_inpainted_frames"]
    clear_and_create_dirs(base_dirs)

    # Extract & Downscale to 360p
    subprocess.run(f"ffmpeg -y -i {INPUT_VIDEO} -vf scale=640:360 video_frames/%05d.png", shell=True, check=True)
    subprocess.run(f"ffmpeg -y -i {MASK_VIDEO} -vf scale=640:360 mask_frames/%05d.png", shell=True, check=True)

    print("\n--- Phase 2: Dilating Masks to Swallow Shadows ---")
    mask_files = sorted(glob.glob("mask_frames/*.png"))
    kernel = np.ones((25, 25), np.uint8)
    for file in mask_files:
        mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            cv2.imwrite(file, cv2.dilate(mask, kernel, iterations=1))

    print("\n--- Phase 3: Slicing Frames for VRAM Efficiency ---")
    vid_files = sorted(glob.glob("video_frames/*.png"))

    split_idx = len(vid_files) // 2 
    for f in vid_files[:split_idx]: shutil.copy(f, "part1_vid/")
    for f in mask_files[:split_idx]: shutil.copy(f, "part1_mask/")
    for f in vid_files[split_idx:]: shutil.copy(f, "part2_vid/")
    for f in mask_files[split_idx:]: shutil.copy(f, "part2_mask/")

    print("\n--- Phase 4: Executing ProPainter Inference ---")
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    # Run Part 1
    subprocess.run(
        "python inference_propainter.py --video ../part1_vid --mask ../part1_mask "
        "--output ../inpainted_results --fp16 --subvideo_length 60 --raft_iter 10",
        cwd=PROPAINTER_DIR, shell=True, check=True
    )
    # Run Part 2
    subprocess.run(
        "python inference_propainter.py --video ../part2_vid --mask ../part2_mask "
        "--output ../inpainted_results --fp16 --subvideo_length 60 --raft_iter 10",
        cwd=PROPAINTER_DIR, shell=True, check=True
    )

    print("\n--- Phase 5: Recompiling Output ---")
    # Gather generated frames and rename them sequentially
    p1_out = sorted(glob.glob("inpainted_results/part1_vid/inpaint_out/*.png") or glob.glob("inpainted_results/part1_vid/*.png"))
    p2_out = sorted(glob.glob("inpainted_results/part2_vid/inpaint_out/*.png") or glob.glob("inpainted_results/part2_vid/*.png"))
    
    counter = 1
    for f in p1_out + p2_out:
        shutil.copy(f, f"final_inpainted_frames/{counter:05d}.png")
        counter += 1

    subprocess.run(
        "ffmpeg -y -r 30 -i final_inpainted_frames/%05d.png -c:v libx264 -pix_fmt yuv420p FINAL_INPAINTED_DRONE_SHOT.mp4",
        shell=True, check=True
    )
    
    print("\nSuccess! Generated FINAL_INPAINTED_DRONE_SHOT.mp4")

if __name__ == "__main__":
    main()