# UESTC4006P Individual Project — Road Crack Detection System (YOLO-based)

## Project Description
This project develops a road crack detection system by analysing the functional requirements of an automated pavement inspection workflow and clarifying its development objectives and application background. It studies key deep-learning principles and YOLO training techniques, prepares relevant public datasets, constructs detection/segmentation models, and performs system-level and detailed design. The system is implemented and validated through coding and testing, delivering an initial road crack detection pipeline based on YOLO models.

> **Code availability:** All source code for this final-year project is maintained in this repository.

---

## Background & Motivation
With the continuous expansion of road networks, automated and standardised detection of road defects is increasingly important for road maintenance and safety management. Manual inspection is often inefficient and subjective, especially under complex lighting, background textures, and large-scale road scenes.

---

## Method Overview (Detection → ROI → Segmentation)
This repository adopts a cascaded design built on the Ultralytics YOLO ecosystem (YOLOv8/YOLO11 as baselines):

1. **Detection (YOLOv8n / YOLO11n)**: quickly localises candidate crack/defect regions (ROIs) in the full image.
2. **ROI processing**: crops and rescales ROIs; for oversized ROIs, optional **tiling + overlap stitching** can be applied to mitigate scale collapse.
3. **Segmentation (YOLOv8n-seg / YOLO11-seg)**: performs pixel-level delineation on ROIs.
4. **Stitch-back & visualisation**: remaps ROI masks back to the original image to generate overlays and binary masks.

This cascade improves efficiency and helps the segmentation model operate closer to its training-scale distribution.

---

## Datasets
Typical dataset pairing used in this project:

- **RDD2022** (China_Drone & China_MotorBike): used for object detection of road damage categories (e.g., D00/D10/D20/D40).
- **CRACK500**: used for crack segmentation with pixel-level annotation masks.

> Notes:
> - Datasets are not included in this repository. Please download them from their official sources and convert them into Ultralytics/YOLO formats as needed.

---

## Getting Started

### Environment
- Python 3.8+ (recommended: 3.10+)
- PyTorch (CUDA optional but recommended for training)
- Ultralytics dependencies (inherited from upstream)

### Installation (recommended)
Clone this repository and install in editable mode:
```bash
git clone https://github.com/ScatteredWood/UESTC4006P-Individual-Project.git
cd UESTC4006P-Individual-Project
pip install -e .
