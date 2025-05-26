#!/bin/bash
#SBATCH --job-name=train_cnn
#SBATCH --output=logs/train_%j.txt
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

module purge
module load Miniforge3/24.11.3-0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

eval "$(conda shell.bash hook)"
conda activate limefaces

TRANSFER=$SLURM_SUBMIT_DIR/../data_transfer
SCRATCH=$LOCALSCRATCH
cp -r $TRANSFER/* $SCRATCH/
cd $SCRATCH

python scripts/train.py --data_dir data/processed --model_out cnn_lime.pth

RESULTS=$SLURM_SUBMIT_DIR/../results/$SLURM_JOB_ID
mkdir -p $RESULTS
cp -r models/ $RESULTS
