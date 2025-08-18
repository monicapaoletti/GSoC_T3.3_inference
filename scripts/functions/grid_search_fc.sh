#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --account=Sis25_piasini
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=gridsearch_jax
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mpaolett@sissa.it
#SBATCH --array=0-4        # Job array indices

# Map SLURM_ARRAY_TASK_ID to true_g values
TRUE_G_VALUES=(0.0 0.3 0.5 0.7 0.9)
TRUE_G=${TRUE_G_VALUES[$SLURM_ARRAY_TASK_ID]}

# Create output directory with date
DATE=$(date +%Y-%m-%d)
RESULTS_DIR="/leonardo/home/userexternal/mpaolett/GSoC_T3.3_inference/results/${DATE}"
OUT_DIR="${RESULTS_DIR}/out"

mkdir -p "$OUT_DIR"

# Redirect stdout and stderr, include "fc" and true_g in filenames
exec > >(tee -a "$OUT_DIR/gridsearch_fc_${TRUE_G}_cpu.out") 2> >(tee -a "$OUT_DIR/gridsearch_fc_${TRUE_G}_cpu.err" >&2)

echo "Running on $(hostname) at $(date)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Running simulation with true_g=${TRUE_G}"

# Load required modules and activate environment
module load openblas
# module load cuda/12.2

nvidia-smi

eval "$(micromamba shell hook --shell=bash)"
micromamba activate jax_conda

cd $SLURM_SUBMIT_DIR

# Run the Python script with arguments
srun --unbuffered time python3 grid_search_fc.py \
    --true_g ${TRUE_G} \
    --t_end 300000 \
    --grid 1000

