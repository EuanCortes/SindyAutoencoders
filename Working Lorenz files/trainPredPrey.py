print("Loading libraries")
import os
import datetime
import pandas as pd
import numpy as np
from example_predprey import get_pred_prey_data  # Updated to use predator-prey data
from sindy_utils import library_size
from training import train_network
import torch
import gc
import pickle
print("Libraries loaded")

# Generate training, validation, and testing data
noise_strength = 1e-6
training_data = get_pred_prey_data(1024, noise_strength=noise_strength)  # Updated to predator-prey
validation_data = get_pred_prey_data(20, noise_strength=noise_strength)  # Updated to predator-prey

# Define parameters
params = {}

params['input_dim'] = 128
params['latent_dim'] = 2  # Updated for predator-prey (2 variables: prey and predator)
params['model_order'] = 1
params['poly_order'] = 3
params['include_sine'] = False
params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)

# Sequential thresholding parameters
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.0001
params['threshold_frequency'] = 100
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
params['coefficient_initialization'] = 'constant'

# Loss function weighting
params['loss_weight_decoder'] = 1.0
params['loss_weight_sindy_z'] = 1e-4
params['loss_weight_sindy_x'] = 1e-4
params['loss_weight_sindy_regularization'] = 1e-5

params['activation'] = 'sigmoid'
params['widths'] = [64, 32]

# Training parameters
params['epoch_size'] = training_data['x'].shape[0]
params['batch_size'] = 1024
params['learning_rate'] = 1e-3

params['data_path'] = os.getcwd() + '/'
params['print_progress'] = True
params['print_frequency'] = 1

# Training time cutoffs
params['max_epochs'] = 5001
params['refinement_epochs'] = 1000

# Set to GPU if available
if torch.cuda.is_available():
    params['device'] = 'cuda'

from autoencoder_torch import full_network

date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))

# Optionally load a pre-trained model
params['load_model'] = "Results/predprey_2025_05_14_18_26_02_575880.pt"  # Set to None or specify a model path if needed

# Experiment setup
num_experiments = 1
df = pd.DataFrame()
print(f"Starting experiments, Number of epochs per experiment: {params['max_epochs']}, Number of experiments: {num_experiments}")
print(f"Threshold frequency: {params['threshold_frequency']}, Coefficient threshold: {params['coefficient_threshold']}")
print(f"Refinement epochs: {params['refinement_epochs']}, Batch size: {params['batch_size']}")
for i in range(num_experiments):
    print('EXPERIMENT %d' % i)
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    # Reset any memory (mostly useful for GPU use)
    torch.cuda.empty_cache()
    gc.collect()

    # Update experiment-specific params
    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
    params['save_name'] = 'Results/predprey_' + date  # Updated save name for predator-prey
    # Train the model
    results_dict = train_network(training_data, validation_data, params)

    # Store results
    df = pd.concat([df, pd.DataFrame([{**results_dict, **params}])], ignore_index=True)

# Save results
df.to_pickle('Results/experiment_results_' + date + '.pkl')
