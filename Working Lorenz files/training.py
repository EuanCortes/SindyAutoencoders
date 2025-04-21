import torch
import pandas as pd
import numpy as np
import datetime
import gc
import os
from autoencoder_torch import full_network
from torch import nn
import torch.nn.functional as F
import pickle

def train_network(training_data, val_data, params):
    # Convert data to PyTorch tensors
    training_data = {k: torch.from_numpy(v).float() for k, v in training_data.items()}
    val_data = {k: torch.from_numpy(v).float() for k, v in val_data.items()}
    
    # Initialize network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = full_network(params).to(device)
    
    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=params['learning_rate'])
    
    # Calculate normalization factors
    x_norm = torch.mean(val_data['x']**2).item()
    sindy_predict_norm_x = torch.mean(val_data['dx']**2).item()
    
    # Training loop
    validation_losses = []
    sindy_model_terms = [np.sum(params['coefficient_mask'])]
    
    print('TRAINING')
    for epoch in range(params['max_epochs']):
        net.train()
        
        # Mini-batch training
        for batch_start in range(0, params['epoch_size'], params['batch_size']):
            batch_end = min(batch_start + params['batch_size'], params['epoch_size'])
            batch_idx = torch.arange(batch_start, batch_end)
            
            # Get batch data
            batch_data = {
                'x': training_data['x'][batch_idx].to(device),
                'dx': training_data['dx'][batch_idx].to(device)
            }
            
            # Forward pass
            outputs = net(batch_data['x'])
            
            # Compute loss
            loss, losses = define_loss(outputs, batch_data, params)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation and printing
        if params['print_progress'] and (epoch % params['print_frequency'] == 0):
            validation_losses.append(print_progress(net, epoch, val_data, params, x_norm, sindy_predict_norm_x, device))
        
        # Sequential thresholding
        if params['sequential_thresholding'] and (epoch % params['threshold_frequency'] == 0) and (epoch > 0):
            with torch.no_grad():
                coefficient_mask = (torch.abs(net.sindy_coefficients) > params['coefficient_threshold']).float()
                net.coefficient_mask = coefficient_mask.to(device)
                print(f'THRESHOLDING: {int(torch.sum(coefficient_mask))} active coefficients')
                sindy_model_terms.append(torch.sum(coefficient_mask).item())
    
    # Refinement phase
    print('REFINEMENT')
    for epoch in range(params['refinement_epochs']):
        net.train()
        
        for batch_start in range(0, params['epoch_size'], params['batch_size']):
            batch_end = min(batch_start + params['batch_size'], params['epoch_size'])
            batch_idx = torch.arange(batch_start, batch_end)
            
            batch_data = {
                'x': training_data['x'][batch_idx].to(device),
                'dx': training_data['dx'][batch_idx].to(device)
            }
            
            outputs = net(batch_data['x'])
            loss, losses = define_loss(outputs, batch_data, params, refinement=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if params['print_progress'] and (epoch % params['print_frequency'] == 0):
            validation_losses.append(print_progress(net, epoch, val_data, params, x_norm, sindy_predict_norm_x, device, refinement=True))
    
    # Save model and results
    torch.save(net.state_dict(), params['data_path'] + params['save_name'] + '.pt')
    pickle.dump(params, open(params['data_path'] + params['save_name'] + '_params.pkl', 'wb'))
    
    # Final evaluation
    net.eval()
    with torch.no_grad():
        val_data_device = {
            'x': val_data['x'].to(device),
            'dx': val_data['dx'].to(device)
        }
        
        outputs = net(val_data_device['x'])
        final_losses = define_loss(outputs, val_data_device, params, return_losses=True)
        
        results_dict = {
            'num_epochs': epoch,
            'x_norm': x_norm,
            'sindy_predict_norm_x': sindy_predict_norm_x,
            'sindy_coefficients': net.sindy_coefficients.cpu().numpy(),
            'loss_decoder': final_losses['decoder'],
            'loss_decoder_sindy': final_losses['sindy_x'],
            'loss_sindy': final_losses['sindy_z'],
            'loss_sindy_regularization': final_losses['sindy_regularization'],
            'validation_losses': np.array(validation_losses),
            'sindy_model_terms': np.array(sindy_model_terms),
        }
    
    return results_dict

def define_loss(outputs, data, params, refinement=False, return_losses=False):
    """Compute the loss function."""
    losses = {}
    
    # Reconstruction loss
    losses['decoder'] = F.mse_loss(outputs['x_decode'], data['x'])
    
    # SINDy losses
    if params['model_order'] == 1:
        losses['sindy_z'] = F.mse_loss(outputs['dz'], outputs['sindy_predict'])
        losses['sindy_x'] = F.mse_loss(data['dx'], outputs['dx_decode'])
    else:
        losses['sindy_z'] = F.mse_loss(outputs['ddz'], outputs['sindy_predict'])
        losses['sindy_x'] = F.mse_loss(data['ddx'], outputs['ddx_decode'])
    
    # Regularization loss
    if hasattr(outputs, 'coefficient_mask'):
        sindy_coefficients = outputs['coefficient_mask'] * outputs['sindy_coefficients']
    else:
        sindy_coefficients = outputs['sindy_coefficients']
    losses['sindy_regularization'] = torch.mean(torch.abs(sindy_coefficients))
    
    if refinement:
        loss = (params['loss_weight_decoder'] * losses['decoder'] +
               params['loss_weight_sindy_z'] * losses['sindy_z'] +
               params['loss_weight_sindy_x'] * losses['sindy_x'])
    else:
        loss = (params['loss_weight_decoder'] * losses['decoder'] +
               params['loss_weight_sindy_z'] * losses['sindy_z'] +
               params['loss_weight_sindy_x'] * losses['sindy_x'] +
               params['loss_weight_sindy_regularization'] * losses['sindy_regularization'])
    
    if return_losses:
        return losses
    return loss, losses

def print_progress(net, epoch, val_data, params, x_norm, sindy_predict_norm, device, refinement=False):
    """Print training progress."""
    net.eval()
    with torch.no_grad():  # Disable gradient computation during validation
        val_data_device = {
            'x': val_data['x'].to(device),
            'dx': val_data['dx'].to(device)
        }
        if params['model_order'] == 2:
            val_data_device['ddx'] = val_data['ddx'].to(device)
        
        # Forward pass without gradient tracking
        outputs = net(val_data_device['x'])
        
        # Compute loss
        loss, losses = define_loss(outputs, val_data_device, params, refinement=refinement)
        
        training_loss = loss.item()
        validation_losses = [l.item() for l in losses.values()]
        
        print(f"Epoch {epoch}")
        print(f"   training loss {training_loss}, {validation_losses}")
        
        decoder_loss_ratio = losses['decoder'].item() / x_norm
        decoder_sindy_loss_ratio = losses['sindy_x'].item() / sindy_predict_norm
        print(f"decoder loss ratio: {decoder_loss_ratio}, decoder SINDy loss ratio: {decoder_sindy_loss_ratio}")
    
    return validation_losses