import torch
import pickle
import numpy as np
from autoencoder_torch import full_network, define_loss

def train_network(training_data, val_data, params):
    # SET UP NETWORK
    autoencoder_network = full_network(params)
    loss, losses, loss_refinement = define_loss(autoencoder_network, params)

    validation_dict = create_feed_dictionary(val_data, params, idxs=None)

    x_norm = np.mean(val_data['x']**2)
    if params['model_order'] == 1:
        sindy_predict_norm_x = np.mean(val_data['dx']**2)
    else:
        sindy_predict_norm_x = np.mean(val_data['ddx']**2)

    validation_losses = []
    sindy_model_terms = [np.sum(params['coefficient_mask'])]

    print('TRAINING')
    optimizer = torch.optim.Adam(autoencoder_network.parameters(), lr=params['learning_rate'])
    for i in range(params['max_epochs']):
        for j in range(params['epoch_size']//params['batch_size']):
            batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
            train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)

            optimizer.zero_grad()
            output = autoencoder_network(train_dict['x:0'])
            current_loss = loss(output, train_dict['x:0'])
            current_loss.backward()
            optimizer.step()

        if params['print_progress'] and (i % params['print_frequency'] == 0):
            validation_losses.append(print_progress(autoencoder_network, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm_x))

        if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 0):
            with torch.no_grad():
                sindy_coefficients = autoencoder_network.sindy_coefficients
                params['coefficient_mask'] = (torch.abs(sindy_coefficients) > params['coefficient_threshold']).float()
                print('THRESHOLDING: %d active coefficients' % np.sum(params['coefficient_mask']))
            sindy_model_terms.append(np.sum(params['coefficient_mask']))

    print('REFINEMENT')
    for i_refinement in range(params['refinement_epochs']):
        for j in range(params['epoch_size']//params['batch_size']):
            batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
            train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)

            optimizer.zero_grad()
            output = autoencoder_network(train_dict['x:0'])
            current_loss_refinement = loss_refinement(output, train_dict['x:0'])
            current_loss_refinement.backward()
            optimizer.step()

        if params['print_progress'] and (i_refinement % params['print_frequency'] == 0):
            validation_losses.append(print_progress(autoencoder_network, i_refinement, loss_refinement, losses, train_dict, validation_dict, x_norm, sindy_predict_norm_x))

    # Save model and parameters
    torch.save(autoencoder_network.state_dict(), params['data_path'] + params['save_name'])
    pickle.dump(params, open(params['data_path'] + params['save_name'] + '_params.pkl', 'wb'))
    
    # Final validation losses
    final_losses = (losses['decoder'], losses['sindy_x'], losses['sindy_z'],
                    losses['sindy_regularization'])
    with torch.no_grad():
        if params['model_order'] == 1:
            sindy_predict_norm_z = np.mean(autoencoder_network.dz**2)
        else:
            sindy_predict_norm_z = np.mean(autoencoder_network.ddz**2)
        sindy_coefficients = autoencoder_network.sindy_coefficients

    results_dict = {
        'num_epochs': i,
        'x_norm': x_norm,
        'sindy_predict_norm_x': sindy_predict_norm_x,
        'sindy_predict_norm_z': sindy_predict_norm_z,
        'sindy_coefficients': sindy_coefficients,
        'loss_decoder': final_losses[0],
        'loss_decoder_sindy': final_losses[1],
        'loss_sindy': final_losses[2],
        'loss_sindy_regularization': final_losses[3],
        'validation_losses': np.array(validation_losses),
        'sindy_model_terms': np.array(sindy_model_terms)
    }

    return results_dict

def print_progress(model, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm):
    """
    Print loss function values to keep track of the training progress.

    Arguments:
        model - the PyTorch model
        i - the training iteration
        loss - PyTorch object representing the total loss function used in training
        losses - tuple of the individual losses that make up the total loss
        train_dict - dictionary of training data
        validation_dict - dictionary of validation data
        x_norm - float, the mean square value of the input
        sindy_predict_norm - float, the mean square value of the time derivatives of the input.

    Returns:
        Tuple of losses calculated on the validation set.
    """
    model.eval()
    training_loss_vals = (loss(model(train_dict['x:0'])), *[loss_func(model(train_dict['x:0'])).item() for loss_func in losses])
    validation_loss_vals = (loss(model(validation_dict['x:0'])), *[loss_func(model(validation_dict['x:0'])).item() for loss_func in losses])
    
    print("Epoch %d" % i)
    print(f"   training loss {training_loss_vals[0]}, {training_loss_vals[1:]}")
    print(f"   validation loss {validation_loss_vals[0]}, {validation_loss_vals[1:]}")
    
    decoder_losses = [loss_func(model(validation_dict['x:0'])).item() for loss_func in [losses['decoder'], losses['sindy_x']]]
    loss_ratios = (decoder_losses[0] / x_norm, decoder_losses[1] / sindy_predict_norm)
    print(f"decoder loss ratio: {loss_ratios[0]:f}, decoder SINDy loss ratio: {loss_ratios[1]:f}")
    
    model.train()  # Switch back to training mode
    return validation_loss_vals

def create_feed_dictionary(data, params, idxs=None):
    """
    Create the feed dictionary for passing into the model.

    Arguments:
        data - Dictionary containing the data to be passed in. Must contain 'x', 'dx' (and possibly 'ddx').
        params - Dictionary containing model and training parameters.
        idxs - Optional indices that select examples from the dataset to pass in. If None, all examples are used.

    Returns:
        feed_dict - Dictionary containing relevant data for the model.
    """
    if idxs is None:
        idxs = np.arange(data['x'].shape[0])

    feed_dict = {
        'x:0': torch.tensor(data['x'][idxs], dtype=torch.float32),
        'dx:0': torch.tensor(data['dx'][idxs], dtype=torch.float32),
    }
    if params['model_order'] == 2:
        feed_dict['ddx:0'] = torch.tensor(data['ddx'][idxs], dtype=torch.float32)
    
    if params['sequential_thresholding']:
        feed_dict['coefficient_mask:0'] = torch.tensor(params['coefficient_mask'], dtype=torch.float32)
    
    feed_dict['learning_rate:0'] = torch.tensor(params['learning_rate'], dtype=torch.float32)
    
    return feed_dict
