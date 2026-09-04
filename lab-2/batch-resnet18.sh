#!/bin/bash
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=resnet18-%j.out
#SBATCH --error=resnet18-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if (( $# > 0 )); then
    exec "$@"
fi

exec .venv/bin/python resnet18.py
