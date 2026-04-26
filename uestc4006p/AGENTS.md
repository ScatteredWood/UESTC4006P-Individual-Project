# Thesis project: Road crack and pothole detection / segmentation

## Project identity
This folder contains my thesis project code.
Focus on this subproject first before touching the wider Ultralytics framework.

## Project goal
This project is for road damage inspection, especially road crack and pothole analysis.

Main goals:
- detect road damage regions,
- crop ROI with padding,
- tile large ROIs into smaller patches,
- segment crack areas more accurately,
- stitch results back to the original image,
- support experiments, evaluation, and thesis figures.

## Datasets
Possible datasets used in this project include:
- RDD2022 China
- Crack500
- self-collected road images

Always inspect the actual dataset yaml, labels, and scripts before assuming class definitions.

## Important damage categories
Common target categories may include:
- D00 longitudinal crack
- D10 transverse crack
- D20 alligator crack
- D40 pothole

If current code uses different labels or merges categories, inspect the actual project files first.

## Typical pipeline
Typical workflow in this thesis project:
1. detection model finds road damage regions,
2. crop ROI with padding,
3. tile large ROI into smaller patches,
4. segmentation model predicts masks,
5. merge / stitch masks back to the original image,
6. evaluate results and export outputs for thesis analysis.

## Environment
- Windows
- VS Code
- Ultralytics YOLO
- local GPU training
- experiments organized under this subproject

## What I usually want help with
Typical tasks include:
- understanding existing scripts,
- debugging training / validation / prediction issues,
- improving preprocessing scripts,
- ROI crop / tiling / stitching,
- pseudo-label generation,
- segmentation improvement,
- experiment organization,
- evaluation output,
- thesis-ready figures or summaries.

## Default working preferences
- Read existing code first, then summarize your understanding before making large edits.
- Prefer editing files inside `uestc4006p` first.
- Do not change dataset directory structure unless necessary.
- Do not delete old experiment scripts unless explicitly asked.
- Prefer adding helper scripts over heavily rewriting old scripts.
- Keep changes minimal, reversible, and easy to test.
- If the request is ambiguous, prioritize the thesis workflow rather than framework refactoring.

## Model-structure changes
If I ask to modify model structure:
1. first identify where the model is selected or instantiated in this subproject,
2. then identify which Ultralytics source file defines the relevant module,
3. summarize the dependency path before editing,
4. keep changes minimal and reversible,
5. explain which project file and which framework file are connected.

Before changing framework files, tell me exactly:
- which framework file will be changed,
- which project file depends on it,
- what effect is expected.

## Performance and GPU evaluation
When I ask whether I need a stronger GPU, do not guess.

First inspect:
- model name / model size,
- task type (detect / seg / pose / etc.),
- image size,
- batch size,
- dataloader workers,
- training speed per epoch or per iteration,
- GPU memory usage,
- GPU utilization,
- whether out-of-memory occurs,
- whether training is bottlenecked by dataloader or storage.

Then summarize:
1. current bottleneck,
2. whether the current GPU is sufficient,
3. what settings can be optimized before upgrading hardware,
4. what kind of stronger GPU would help and why.

## Output style
- Be concrete and engineering-oriented.
- For code changes, mention exact files and expected effects.
- For experiment suggestions, prioritize practical next steps over general theory.
- For debugging, explain the most likely cause first.

## Runs/Output naming convention (default)
Use short, stable names for all new outputs:

- Directory name format:
  - `{mode}_{task}_{data}_{model}`
  - or `{mode}_{task}_{data}_{model}_{tag}`
- `mode` only allows: `train` / `val` / `pred`
- `task` supports: `det` / `seg` / `cascade`
- If `tag` is empty, omit it completely (no trailing underscore)
- No timestamp subdirectory is used anymore
- If name conflicts, append `_2`, `_3`, `_4`, ...

Checkpoints and metadata:
- `checkpoints` directory name must be the same as the corresponding training run directory name
- Every `train` / `val` / `pred` / `cascade` output directory must include `run_meta.yaml`

Compatibility and traceability:
- Keep full run configuration in `args.yaml`
- Preserve legacy naming via explicit fallback switch when needed
- For batch migration of old runs/checkpoints, do dry-run mapping first, then apply migration
