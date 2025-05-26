#!/bin/bash
#SBATCH --job-name=lime_explain
#SBATCH --output=logs/lime_%j.txt
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

module purge
module load Miniforge3/24.11.3-0

eval "$(conda shell.bash hook)"
conda activate limefaces

TRANSFER=$SLURM_SUBMIT_DIR/../data_transfer
SCRATCH=$LOCALSCRATCH
cp -r $TRANSFER/* $SCRATCH/
cd $SCRATCH

mkdir -p outputs/lime_heatmaps

for i in {0..4}; do
  python scripts/lime_explain.py \
    --data_dir data/processed/test \
    --model_path results/<MODEL_JOB_ID>/models/cnn_lime.pth \
    --image_index $i \
    --num_classes <NUM_CLASSES> \
    --output_dir outputs/lime_heatmaps
done

RESULTS=$SLURM_SUBMIT_DIR/../results/$SLURM_JOB_ID
mkdir -p $RESULTS
cp -r outputs/lime_heatmaps $RESULTS
