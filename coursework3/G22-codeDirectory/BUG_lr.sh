#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=16000  # memory in Mb
#SBATCH -o lr_0.1_outfile  # send stdout to train_100_L2_lr
#SBATCH -e lr_0.1_errfile  # send stder to train_100_L2_lr_errfile
#SBATCH -t 8:00:00  # time requested in hour:minute:secon
export CUDA_HOME=/opt/cuda-8.0.44

export CUDNN_HOME=/opt/cuDNN-6.0_8.0

export STUDENT_ID=s1717961

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/
# Activate the relevant virtual environment:


source /home/s1717961/miniconda3/bin/activate mlp
python lr-0.1-l2.py
