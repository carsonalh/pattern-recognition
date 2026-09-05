#!/bin/bash
# Submit the batch-size/learning-rate Cartesian product through Slurm.

set -euo pipefail

export PATH="$HOME/miniconda3/bin:/opt/slurm/bin:$PATH"

epochs=50
parallel_jobs=3
partition=comp3710
qos=
time_limit=00:30:00
batch_sizes=(32 64 128 256 512 1024)
learning_rates=(0.001 0.005)
local_mode=0

while (($# > 0)); do
    case "$1" in
        --local)
            local_mode=1
            shift 1
            ;;
        --epochs)
            epochs=$2
            shift 2
            ;;
        --parallel-jobs)
            parallel_jobs=$2
            shift 2
            ;;
        --partition)
            partition=$2
            shift 2
            ;;
        --qos)
            qos=$2
            shift 2
            ;;
        --time)
            time_limit=$2
            shift 2
            ;;
        --batch-sizes)
            IFS=',' read -r -a batch_sizes <<< "$2"
            shift 2
            ;;
        --learning-rates)
            IFS=',' read -r -a learning_rates <<< "$2"
            shift 2
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

if (( local_mode )); then
    parallel_jobs=1
fi

command -v parallel >/dev/null || {
    printf 'GNU Parallel is required but was not found\n' >&2
    exit 1
}

sbatch_options=(--partition="$partition")
if [[ -n "$qos" ]]; then
    sbatch_options+=(--qos="$qos")
fi
if [[ -n "$time_limit" ]]; then
    sbatch_options+=(--time="$time_limit")
fi

if (( local_mode )); then
    parallel --halt soon,fail=1 --line-buffer --jobs "$parallel_jobs" \
        'uv run ./resnet18.py --epochs '"$epochs"' --batch-size {1} --learning-rate {2} --num-workers 1 > "resnet18_b{1}_lr{2}.out" 2> "resnet18_b{1}_lr{2}.err"' \
        ::: "${batch_sizes[@]}" ::: "${learning_rates[@]}"
else
    parallel --halt soon,fail=1 --line-buffer --jobs "$parallel_jobs" \
        sbatch --wait "${sbatch_options[@]}" \
        --output=resnet18_b{1}_lr{2}.out \
        --error=resnet18_b{1}_lr{2}.err \
        batch-resnet18.sh \
        --epochs "$epochs" --batch-size {1} --learning-rate {2} --num-workers 1 \
        ::: "${batch_sizes[@]}" ::: "${learning_rates[@]}"
fi
