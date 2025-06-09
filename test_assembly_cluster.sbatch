#!/bin/bash
#SBATCH --job-name=assembly-test
#SBATCH --output=assembly_test_%j.out
#SBATCH --error=assembly_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=general  # Change to your cluster's partition name

# Load modules (adjust for your cluster)
module load python/3.8
module load gcc/9.3.0  # Sometimes needed for scientific Python packages

# Activate conda environment
source ~/.bashrc
conda activate assembly-opt

# Set working directory
cd $SLURM_SUBMIT_DIR

# Set parallelization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting assembly optimization testing..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32G"
echo "Start time: $(date)"

# Run the tests
echo "Running quick validation test..."
python testing.py --quick

echo "Test completed at: $(date)"
echo "Check test_results/ directory for outputs"