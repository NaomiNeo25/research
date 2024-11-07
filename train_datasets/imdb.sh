#!/bin/bash

#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
##SBATCH --gpus=1
#SBATCH --mem=0  # Unlimited memory
#SBATCH --time=2-00:00:00
#SBATCH --job-name=IMDB
#SBATCH -o /home-mscluster/nmuzamani2/research/output/IMDB.%N.%j.out
#SBATCH -e /home-mscluster/nmuzamani2/research/output/IMDB.%N.%j.err


export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.1

source /home-mscluster/nmuzamani2/anaconda3/etc/profile.d/conda.sh
conda activate research

srun python3 trainIMDB.py