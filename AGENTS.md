# Repository-level instructions

## Scope
This repository contains the Ultralytics framework and a thesis subproject under `uestc4006p`.
When working across the repository, distinguish framework code from experiment/project code.

## General working style
- Read relevant files before making edits.
- Prefer minimal, high-confidence changes.
- Do not refactor unrelated files.
- Preserve existing code style and file organization.
- Do not rename public APIs unless explicitly requested.
- Before making large edits, summarize the relevant files and your understanding first.

## Priority
- If the request is ambiguous, focus on the thesis subproject under `uestc4006p` first.
- Prefer changing scripts/configs in `uestc4006p` instead of changing framework internals.
- Only modify the core Ultralytics framework when the task explicitly requires framework-level or model-structure changes.

## Framework edit rules
Treat the `ultralytics` package as framework code.

Before changing framework code, first identify:
1. which file defines the module,
2. which file instantiates or registers it,
3. which training/inference/export path will be affected,
4. whether the same change can be done more safely in the subproject instead.

Before editing framework code:
- summarize the dependency path,
- explain why the change is needed,
- keep the change minimal and reversible.

## Validation
After code changes, suggest lightweight validation first:
- import checks,
- shape checks,
- config checks,
- one short training/validation run,
- one inference smoke test.

Do not launch long training jobs unless explicitly requested.

## Performance / hardware awareness
When asked about training speed, memory use, or whether a stronger GPU is needed:
- inspect the actual training script, model, imgsz, batch size, dataloader settings, and augmentation settings first,
- distinguish among compute bottleneck, VRAM bottleneck, dataloader bottleneck, and I/O bottleneck,
- prefer evidence from actual runs over assumptions.

## Communication style
- Be concrete and engineering-oriented.
- Mention exact files when proposing code changes.
- When the task affects both framework code and project code, explicitly list both sides.