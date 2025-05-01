#!/bin/sh
### General options
#BSUB -q hpc
#BSUB -J Lorenz_Training
####BSUB -J test
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 71:00
#BSUB -o batch_output/Lorenz_CPU_%J.out
#BSUB -e batch_output/Lorenz_CPU_%J.err

# Initialize Python environment
source ../../../EEG\ Image\ Decoder/Phillip_Code/BCI/bin/activate

#Run
#python -m jupyter nbconvert --to notebook --execute --inplace train_lorenz.ipynb --stdout > batch_output/Lorenz_${LSB_JOBID}.out
echo "Running Lorenz training script..."
python -u trainlorenz.py

# Deactivate virtual environment (optional)
deactivate