# UESTC4006P Final-Year Project Main Repository

## Project Overview
This repository is the main codebase for the UESTC4006P final-year project on road-damage and crack analysis based on Ultralytics YOLO.  
It contains:
- YOLO framework code (`ultralytics/`)
- project-specific training, validation, prediction, and cascade inference scripts (`uestc4006p/scripts/`)
- project configs and experiment management helpers (`uestc4006p/configs/`, `uestc4006p/scripts/`)

## Repository Role In The Final-Year Project
This is the **primary research/training repository** used for model development and experiment execution.  
The focus here is model-side work: detection, segmentation, evaluation, and cascade inference pipelines.

## Relationship With GUI Repository
GUI repository: https://github.com/ScatteredWood/uestc4006p-gui

Relationship:
- this repository provides model training/evaluation/inference scripts and model artifacts
- GUI repository provides the desktop application and user-facing workflow integration
- GUI can consume weights/exports/results generated from this repository

## Relationship With YOLO26 Repository
YOLO26 repository: https://github.com/ScatteredWood/uestc4006p-yolo26

Relationship:
- this repository contains the main project code and baseline/customized YOLO workflows
- YOLO26-specific model work is maintained in the dedicated YOLO26 repository
- for YOLO26-exclusive experiments, prefer scripts and docs in the YOLO26 repository

## Environment
Typical setup:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Notes:
- Python 3.10+ is recommended.
- GPU/CUDA setup depends on your PyTorch installation.

## Dataset Preparation
Datasets are not bundled in this repository. Prepare your datasets locally and provide dataset yaml paths when running scripts.

Examples:
- detection dataset yaml: `path/to/det_data.yaml`
- segmentation dataset yaml: `path/to/seg_data.yaml`

You can also set:
- `UESTC4006P_DATA_ROOT` for scripts that resolve dataset names via `config.py`.

## Script Entrypoints
Main script directory: `uestc4006p/scripts/`

Core entries:
- detection/segmentation training: `train.py`
- validation: `val.py`
- direct prediction: `predict.py`
- cascade inference (det -> ROI -> seg): `cascade_infer_detseg.py`

## Training Examples
Detection:

```bash
python uestc4006p/scripts/train.py ^
  --task detect ^
  --data path/to/det_data.yaml ^
  --model yolov8n.pt ^
  --dataset-name RDD2022 ^
  --domain China ^
  --model-alias yolov8n ^
  --imgsz 1024 --epochs 300 --batch 16 --device 0
```

Segmentation:

```bash
python uestc4006p/scripts/train.py ^
  --task segment ^
  --data path/to/seg_data.yaml ^
  --model yolov8n-seg.pt ^
  --dataset-name CRACK500 ^
  --domain ALL ^
  --model-alias yolov8nseg ^
  --imgsz 800 --epochs 200 --batch 24 --device 0
```

## Validation Examples
Detection checkpoint validation:

```bash
python uestc4006p/scripts/val.py ^
  --best uestc4006p/checkpoints/your_det_run/best.pt ^
  --data path/to/det_data.yaml ^
  --task detect ^
  --dataset RDD2022_China ^
  --model-name yolov8n.pt ^
  --device 0
```

Segmentation checkpoint validation:

```bash
python uestc4006p/scripts/val.py ^
  --best uestc4006p/checkpoints/your_seg_run/best.pt ^
  --data path/to/seg_data.yaml ^
  --task segment ^
  --dataset CRACK500 ^
  --model-name yolov8n-seg.pt ^
  --device 0
```

## Prediction Examples
Direct prediction:

```bash
python uestc4006p/scripts/predict.py ^
  --best uestc4006p/checkpoints/your_seg_run/best.pt ^
  --source path/to/images_or_video ^
  --task segment ^
  --dataset demo_input ^
  --imgsz 1280 --conf 0.25 --iou 0.6 --device 0
```

Batch prediction helpers:
- `batch_predict_det_models.py`
- `batch_predict_seg_models.py`

These scripts now support repo-relative defaults and can be overridden by environment variables such as:
- `UESTC4006P_RUNS_ROOT`
- `UESTC4006P_DET_SOURCE`
- `UESTC4006P_SEG_SOURCE`

## Cascade Inference Examples
Single cascade run:

```bash
python uestc4006p/scripts/cascade_infer_detseg.py ^
  --det uestc4006p/checkpoints/your_det_run/best.pt ^
  --seg uestc4006p/checkpoints/your_seg_run/best.pt ^
  --source path/to/images ^
  --outdir uestc4006p/runs ^
  --det-conf 0.15 --seg-conf 0.10 --seg-thr 0.30
```

Batch pair cascade helper:
- `batch_cascade_pairs_standard.py`

## Output/Result Directory Explanation
Common local artifact directories include:
- `uestc4006p/runs/`
- `uestc4006p/checkpoints/`
- `weights/`
- optional local outputs such as `logs/`, `outputs/`, `results/`, `exports/`, `wandb/`

Experimental outputs such as training logs, validation curves, prediction visualisations and model weights may be stored locally under directories such as runs/, logs/, outputs/ or checkpoints/. These files are treated as experimental artefacts and should not be removed during repository cleanup.

## Notes On Model Weights And Experimental Results
- Model weights, exported engines, and large experimental artifacts may be kept locally and are not guaranteed to be included in public commits.
- If a result/weight directory already exists locally, do not remove it during cleanup.
- Use `.gitignore` to prevent accidental commit of newly generated large artifacts.
