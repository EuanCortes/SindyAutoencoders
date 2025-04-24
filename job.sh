#!/bin/sh
### General options
#BSUB -q c02613
#BSUB -J Flower
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:59
#BSUB -o batch_output/Flower_%J.out
#BSUB -e batch_output/Flower_%J.err

# Initialize Python environment
source venv_1/bin/activate

#Run
python -m jupyter nbconvert --to notebook --execute FedPerFlower.ipynb
# Check if the notebook executed successfully
if [ $? -eq 0 ]; then
    echo "Notebook executed successfully."
else
    echo "Notebook execution failed."
fi

# Deactivate virtual environment (optional)
deactivate