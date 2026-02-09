# Conformal Training

## Project
This is a conformal prediction training codebase, forked from iemppu/Conformal-training.
I am developing new methods on top of this version-0 codebase.

## Workflow
- Read and follow .claude/skills/research-coder/SKILL.md for all coding tasks
- Always work on a feature branch, never commit directly to main
- Code must work on both Google Colab and SLURM cluster
- Run pytest tests/ -v before committing

## Tech Stack
Python, PyTorch, OmegaConf, W&B, pytest

## Repo Structure
- `src/models/` — network architectures (ResNet, VGG, DenseNet, EfficientNet)
- `src/methods/` — conformal prediction methods, scoring, losses, sorting
- `src/data/` — dataset loaders (CIFAR-100)
- `src/utils/` — config parsing, metrics, training helpers
- `scripts/` — entry-point scripts (train.py)
- `configs/` — YAML experiment configs
- `tests/` — pytest tests

## Key Commands
- Train: python scripts/train.py (with args from src/utils/config.py)
- Test: pytest tests/ -v
