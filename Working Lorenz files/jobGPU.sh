#!/bin/sh
### General options
#BSUB -q gpuv100
#BSUB -J Lorenz_Training_GPU
####BSUB -J test
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o batch_output/Lorenz_GPU_%J.out
#BSUB -e batch_output/Lorenz_GPU_%J.err

# Initialize Python environment
source ../../../EEG\ Image\ Decoder/Phillip_Code/BCI/bin/activate

#Run
#python -m jupyter nbconvert --to notebook --execute --inplace train_lorenz.ipynb --stdout > batch_output/Lorenz_${LSB_JOBID}.out
echo "Running Lorenz training script..."
python -u trainlorenz.py

# Deactivate virtual environment (optional)
deactivate