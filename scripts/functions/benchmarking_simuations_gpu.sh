#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=02:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --account=Sis25_piasini
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=benchmcpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mpaolett@sissa.it



DATE=$(date +%Y-%m-%d)
RESULTS_DIR="/leonardo/home/userexternal/mpaolett/GSoC_T3.3_inference/results/${DATE}"
OUT_DIR="${RESULTS_DIR}/out"

mkdir -p "$OUT_DIR"

# Redirect output and error logs
exec > >(tee -a "$OUT_DIR/benchmarking_gpu.out") 2> >(tee -a "$OUT_DIR/benchmarking_gpu.err" >&2)

echo "Running on $(hostname) at $(date)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load required modules and activate environment
module load openblas
# module load cuda/12.2

nvidia-smi

eval "$(micromamba shell hook --shell=bash)"
micromamba activate jax_conda

cd $SLURM_SUBMIT_DIR

srun --unbuffered time python3 benchmarking_simuations.py


