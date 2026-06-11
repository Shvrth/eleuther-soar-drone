# An Initial Drone Scene Editing Pipeline
**An EleutherAI SOAR Application Implementation**

This repository contains an end-to-end computer vision pipeline designed to seamlessly track, erase, and replace moving vehicles in dynamic aerial drone footage. 

By combining state-of-the-art segmentation, optical-flow inpainting, and custom cinematic trajectory smoothing, this pipeline allows for  2D asset injection (e.g., swapping a standard car for a Ferrari) into complex video environments.

---

## 🚀 Pipeline Architecture

The system is broken down into three modular stages:

1. **Tracking & Masking (`YOLOv11` + `SAM 2`):** Utilizes a fine-tuned YOLO model for detection and SAM 2 for pixel-perfect segmentation. Wraps the SAM 2 mask in a rotated bounding box (`cv2.minAreaRect`) to extract the *true physical heading angle* of the vehicle, bypassing the limitations of standard axis-aligned bounding boxes.
2. **Deep Temporal Inpainting (`ProPainter`):** Mathematically dilates masks to swallow drop shadows, downscales footage for memory safety, and utilizes recurrent optical flow with a deep 60-frame look-ahead window to hallucinate perfectly clean road textures.
3. **Trajectory Compositing:** Applies anti-flip mathematical continuity logic and a heavy **21-frame moving average filter** to unwrapped radians and spatial coordinates, ensuring the injected asset glides flawlessly across the frame.

---

## 💻 System Prerequisites

* **OS:** Linux (Ubuntu 22.04+ recommended) or Windows (WSL2 required for FFmpeg and PyTorch compatibility).
* **GPU:** NVIDIA GPU with at least **12GB VRAM** (RTX 3060/4070 or higher).
* **System Tools:** `ffmpeg` must be installed on your system. 
  * *Ubuntu/WSL:* `sudo apt update && sudo apt install ffmpeg`

---

## 🛠️  Installation & Setup

### Step 1: Clone the Repository & Setup Environment
First, clone this codebase and create a fresh Python environment to avoid dependency conflicts.
```bash
git clone https://github.com/Shvrth/eleuther-soar-drone.git
cd eleuther-soar-drone

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

```
### Step 2: Install Core Dependencies
Install PyTorch, Ultralytics (YOLO), OpenCV, and compile the Segment Anything 2 (SAM 2) library directly from Meta's GitHub.

```bash
pip install -r requirements.txt
```
### Step 3: Setup Models & Download SAM 2 Weights
Navigate to src/models, then use wget to pull down the lightweight SAM 2 Hiera Tiny model.

```bash
wget -P models/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

### Step 4: Clone & Configure ProPainter
ProPainter acts as a standalone application. 

```bash
git clone https://github.com/sczhou/ProPainter.git

# Install ProPainter's specific sub-dependencies
cd ProPainter
pip install -r requirements.txt
cd ..
```

## 🎬 Execution Guide
Ensure your source video (drone_clip_two.mp4) and 2D asset (ferrari.png) are located in the data/ folder. Run the scripts sequentially from the root of the repository:

1. Track, Mask, and Log Trajectory
Extracts SAM 2 masks, handles auto-exposure stabilization, and generates the car_coordinates.txt spatial log file.

```bash
python src/01_track_and_mask.py
```
2. Generate Clean Background
Dilates the shadows, slices the video into VRAM-safe chunks, and executes ProPainter.

```bash
python src/02_prepare_propainter.py
```
3. Composite and Smooth Asset
Injects the 2D asset, applies the 21-frame trajectory filter, and exports the final .mp4.

```bash
python src/03_composite_trajectory.py
```
Check your root directory for the final masterpiece: FINAL_FERRARI_SHOT.mp4!

## 📁 Repository Structure

Before running the pipeline, ensure your directory looks exactly like this:

```text
├── data/                               # Put your input video and 2D assets here
│   ├── drone_clip_two.mp4              
│   └── ferrari.png                     
├── models/                             # Model Weights
│   ├── sam2_hiera_tiny.pt              # (SAM2 Weights)
│   └── top_view_drone_yolo_11_weights.pt # (YOLO pre-trained weights)
├── ProPainter/                         # External Inpainting Engine
├── src/                                # Core Pipeline Logic 
│   ├── 01_track_and_mask.py
│   ├── 02_prepare_propainter.py
│   └── 03_composite_trajectory.py
├── .gitignore                          
├── requirements.txt                    
└── README.md
```

## 🤝 Acknowledgments
**EleutherAI SOAR** for project framework and inspiration.

**Meta Research** for Segment Anything 2.

**Ultralytics** for YOLOv11 framework.

**sczhou** for the incredible ProPainter video inpainting architecture.
