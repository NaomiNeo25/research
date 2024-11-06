#!/bin/bash

#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0  # Unlimited memory
#SBATCH --time=0-00:00  # Unlimited time (format: days-hours:minutes)
#SBATCH --job-name=IMDB
#SBATCH -o /home-mscluster/nmuzamani2/research/IMDBslurm.%N.%j.out
#SBATCH -e /home-mscluster/nmuzamani2/research/IMDBslurm.%N.%j.err


export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.0

source /home-mscluster/nmuzamani2/.bashrc
conda activate research

python3 trainIMDB.py