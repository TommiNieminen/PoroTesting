#!/bin/bash -l
#SBATCH --job-name=examplejob   # Job name
#SBATCH --output=examplejob.o%j # Name of stdout output file
#SBATCH --error=examplejob.e%j  # Name of stderr error file
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1               # Total number of nodes 
#SBATCH --ntasks-per-node=1     # 
#SBATCH --cpus-per-task=14     # 
#SBATCH --gpus-per-task=2       # Allocate one gpu per MPI rank
#SBATCH --time=03:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000223  # Project for billing
#SBATCH --mem=128G

export HF_HOME=$(realpath ./hf_cache)
module use /appl/local/csc/modulefiles
module load pytorch/1.13

srun python test.py eng.devtest wmt17.en-fi.en wmt17.en-fi.fi
