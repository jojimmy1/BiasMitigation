#!/bin/sh -l
# FILENAME: submission.sh
#SBATCH -A standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -t 04:00:00
#SBATCH --job-name process
#SBATCH --cpus-per-task=3
#SBATCH --output=report.txt
#SBATCH --mem-per-cpu=4G

# Print the hostname of the compute node on which this job is running.
/bin/hostname
# Change to the directory from which you originally submitted this job.
cd $SLURM_SUBMIT_DIR
# To use the torch-rocm environment (needs to be done every run)

module purge
module load anaconda/2024.02-py311
conda activate group