#!/bin/bash
#SBATCH --job-name=assembly-full-test
#SBATCH --output=assembly_full_%j.out
#SBATCH --error=assembly_full_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=general  # Change to your cluster's partition name

# Load modules (adjust for your cluster)
module load python/3.8
module load gcc/9.3.0

# Activate conda environment
source ~/.bashrc
conda activate assembly-opt

# Set working directory
cd $SLURM_SUBMIT_DIR

# Set parallelization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting FULL assembly optimization testing..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64G"
echo "Start time: $(date)"

# Run comprehensive tests
echo "Running comprehensive test suite..."
python run_tests.py

echo "Full testing completed at: $(date)"
echo "Check test_results/ directory for outputs"
echo "Main report: test_results/testing_report.html"