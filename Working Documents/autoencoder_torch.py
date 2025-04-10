import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoencoderNetwork(nn.Module):
    def __init__(self, params):
        super(AutoencoderNetwork, self).__init__()

        # Store parameters
        self.input_dim = params['input_dim']
        self.latent_dim = params['latent_dim']
        self.activation = params['activation']
        self.poly_order = params['poly_order']
        self.include_sine = params.get('include_sine', False)
        self.library_dim = params['library_dim']
        self.model_order = params['model_order']
        
        # Define layers and other components
        self.encoder = None  # Placeholder, will be assigned later
        self.decoder = None  # Placeholder, will be assigned later
        self.z = None  # Placeholder, will be assigned later
        self.x_decode = None  # Placeholder, will be assigned later
        self.encoder_weights = None
        self.encoder_biases = None
        self.decoder_weights = None
        self.decoder_biases = None

        x = torch.zeros((1, self.input_dim))  # Example input, size (batch_size, input_dim)

        if self.activation == 'linear':
            self.z, self.x_decode, self.encoder_weights, self.encoder_biases, self.decoder_weights, self.decoder_biases = self.linear_autoencoder(x)
        else:
            self.z, self.x_decode, self.encoder_weights, self.encoder_biases, self.decoder_weights, self.decoder_biases = self.nonlinear_autoencoder(x, params['widths'])

        if self.model_order == 1:
            self.dz = self.z_derivative(self.z, self.encoder_weights, self.encoder_biases, self.activation)
            self.Theta = self.sindy_library_torch(self.z, self.latent_dim, self.poly_order, self.include_sine)
        else:
            self.dz, self.ddz = self.z_derivative_order2(self.z, self.encoder_weights, self.encoder_biases, self.activation)
            self.Theta = self.sindy_library_torch_order2(self.z, self.dz, self.latent_dim, self.poly_order, self.include_sine)

        # Initialize coefficients as a Parameter
        self.sindy_coefficients = nn.Parameter(torch.randn(self.library_dim, self.latent_dim))
        torch.nn.init.xavier_uniform_(self.sindy_coefficients)

        if params['sequential_thresholding']:
            self.coefficient_mask = torch.ones(self.library_dim, self.latent_dim)
            self.sindy_predict = torch.matmul(self.Theta, self.coefficient_mask * self.sindy_coefficients)
        else:
            self.sindy_predict = torch.matmul(self.Theta, self.sindy_coefficients)

        if self.model_order == 1:
            self.dx_decode = self.z_derivative(self.z, self.sindy_predict, self.decoder_weights, self.decoder_biases, self.activation)
        else:
            self.dx_decode, self.ddx_decode = self.z_derivative_order2(self.z, self.dz, self.sindy_predict, self.decoder_weights, self.decoder_biases, self.activation)

    def forward(self, x):
        """
        Forward pass for the autoencoder network.
        """
        # Define the forward pass
        # Apply the encoder and decoder based on the model order
        if self.model_order == 1:
            output = self.dx_decode
        else:
            output = self.ddx_decode
        return output

    # Define the functions for the autoencoder, derivatives, and SINDy libraries
    def linear_autoencoder(self, x):
        # Define the linear autoencoder layers
        encoder = nn.Linear(self.input_dim, self.latent_dim)
        decoder = nn.Linear(self.latent_dim, self.input_dim)
        z = encoder(x)
        x_decode = decoder(z)
        return z, x_decode, encoder.weight, encoder.bias, decoder.weight, decoder.bias

    def nonlinear_autoencoder(self, x, widths):
        # Define the nonlinear autoencoder layers
        encoder = nn.Linear(self.input_dim, widths[0])
        decoder = nn.Linear(widths[-1], self.input_dim)
        z = encoder(x)
        x_decode = decoder(z)
        return z, x_decode, encoder.weight, encoder.bias, decoder.weight, decoder.bias

    def z_derivative(self, z, encoder_weights, encoder_biases, activation):
        # Placeholder for derivative calculation
        return torch.matmul(z, encoder_weights.T) + encoder_biases

    def z_derivative_order2(self, z, dz, encoder_weights, encoder_biases, activation):
        # Placeholder for second-order derivative calculation
        return torch.matmul(dz, encoder_weights.T) + encoder_biases

    def sindy_library_torch(self, z, latent_dim, poly_order, include_sine):
        # Placeholder for SINDy library construction
        return z ** poly_order

    def sindy_library_torch_order2(self, z, dz, latent_dim, poly_order, include_sine):
        # Placeholder for SINDy library construction (2nd order)
        return z ** poly_order + dz ** poly_order

# Now the function `full_network` just returns an instance of `AutoencoderNetwork`:
def full_network(params):
    return AutoencoderNetwork(params)




def define_loss(network, params):
    """
    Create the loss functions.

    Arguments:
        network - Dictionary object containing the elements of the network architecture.
        This will be the output of the full_network() function.
    """
    x = network['x']
    x_decode = network['x_decode']
    if params['model_order'] == 1:
        dz = network['dz']
        dz_predict = network['dz_predict']
        dx = network['dx']
        dx_decode = network['dx_decode']
    else:
        ddz = network['ddz']
        ddz_predict = network['ddz_predict']
        ddx = network['ddx']
        ddx_decode = network['ddx_decode']

    # Ensure coefficient_mask is a PyTorch tensor (if it's a numpy array)
    coefficient_mask = torch.tensor(params['coefficient_mask'], dtype=network['sindy_coefficients'].dtype, device=network['sindy_coefficients'].device)

    # Now you can multiply them
    sindy_coefficients = coefficient_mask * network['sindy_coefficients']


    losses = {}
    losses['decoder'] = F.mse_loss(x, x_decode)
    if params['model_order'] == 1:
        losses['sindy_z'] = F.mse_loss(dz, dz_predict)
        losses['sindy_x'] = F.mse_loss(dx, dx_decode)
    else:
        losses['sindy_z'] = F.mse_loss(ddz, ddz_predict)
        losses['sindy_x'] = F.mse_loss(ddx, ddx_decode)

    losses['sindy_regularization'] = torch.mean(torch.abs(sindy_coefficients))

    loss = params['loss_weight_decoder'] * losses['decoder'] \
           + params['loss_weight_sindy_z'] * losses['sindy_z'] \
           + params['loss_weight_sindy_x'] * losses['sindy_x'] \
           + params['loss_weight_sindy_regularization'] * losses['sindy_regularization']

    loss_refinement = params['loss_weight_decoder'] * losses['decoder'] \
                      + params['loss_weight_sindy_z'] * losses['sindy_z'] \
                      + params['loss_weight_sindy_x'] * losses['sindy_x']

    return loss, losses, loss_refinement

def linear_autoencoder(x, input_dim, d):
    # z,encoder_weights,encoder_biases = encoder(x, input_dim, latent_dim, [], None, 'encoder')
    # x_decode,decoder_weights,decoder_biases = decoder(z, input_dim, latent_dim, [], None, 'decoder')
    z,encoder_weights,encoder_biases = build_network_layers(x, input_dim, latent_dim, [], None, 'encoder')
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, latent_dim, input_dim, [], None, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases,decoder_weights,decoder_biases

def nonlinear_autoencoder(x, input_dim, latent_dim, widths, activation='elu'):
    """
    Construct a nonlinear autoencoder.

    Arguments:

    Returns:
        z -
        x_decode -
        encoder_weights - List of tensorflow arrays containing the encoder weights
        encoder_biases - List of tensorflow arrays containing the encoder biases
        decoder_weights - List of tensorflow arrays containing the decoder weights
        decoder_biases - List of tensorflow arrays containing the decoder biases
    """
    if activation == 'relu':
        activation_function = nn.relu()
    elif activation == 'elu':
        activation_function = nn.elu()
    elif activation == 'sigmoid':
        activation_function = nn.Sigmoid()
    else:
        raise ValueError('invalid activation function')
    # z,encoder_weights,encoder_biases = encoder(x, input_dim, latent_dim, widths, activation_function, 'encoder')
    # x_decode,decoder_weights,decoder_biases = decoder(z, input_dim, latent_dim, widths[::-1], activation_function, 'decoder')
    z,encoder_weights,encoder_biases = build_network_layers(x, input_dim, latent_dim, widths, activation_function, 'encoder')
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, latent_dim, input_dim, widths[::-1], activation_function, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases


def build_network_layers(input, input_dim, output_dim, widths, activation, name):
    """
    Construct one portion of the network (either encoder or decoder).

    Arguments:
        input_dim - Integer, number of state variables in the input to the first layer
        output_dim - Integer, number of state variables to output from the final layer
        widths - List of integers representing how many units are in each network layer
        activation - Activation function to be used at each layer

    Returns:
        model - Sequential model representing the network
    """
    weights = []
    biases = []
    last_width=input_dim
    for i,n_units in enumerate(widths):
        W = nn.Parameter(torch.empty(last_width, n_units))
        nn.init.xavier_uniform_(W) 
        b = nn.Parameter(torch.zeros(n_units))
        input = torch.matmul(input, W) + b
        if activation is not None:
            input = activation(input)
        last_width = n_units
        weights.append(W)
        biases.append(b)
    W = nn.Parameter(torch.empty(last_width, output_dim))
    nn.init.xavier_uniform_(W)
    b = nn.Parameter(torch.zeros(output_dim))
    input = torch.matmul(input, W) + b
    weights.append(W)
    biases.append(b)
    return input, weights, biases


def sindy_library_torch(z, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library.

    Arguments:
        z - 2D torch tensor of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D torch tensor containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = [torch.ones(z.shape[0], device=z.device)]

    for i in range(latent_dim):
        library.append(z[:, i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(z[:, i] * z[:, j])

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(z[:, i] * z[:, j] * z[:, k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        for q in range(p, latent_dim):
                            library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q])

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:, i]))

    return torch.stack(library, dim=1)

import torch
import torch.nn.functional as F

def z_derivative(input, dx, weights, biases, activation='elu'):
    """
    Compute the first order time derivatives by propagating through the network.

    Arguments:
        input - 2D torch tensor, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        weights - List of torch tensors containing the network weights
        biases - List of torch tensors containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        dz - Torch tensor, first order time derivatives of the network output.
    """
    dz = dx
    if activation == 'elu':
        for i in range(len(weights) - 1):
            input = torch.matmul(input, weights[i]) + biases[i]
            dz = torch.mul(torch.minimum(torch.exp(input), torch.tensor(1.0, device=input.device)),
                           torch.matmul(dz, weights[i]))
            input = F.elu(input)
        dz = torch.matmul(dz, weights[-1])
    elif activation == 'relu':
        for i in range(len(weights) - 1):
            input = torch.matmul(input, weights[i]) + biases[i]
            dz = torch.mul((input > 0).float(), torch.matmul(dz, weights[i]))
            input = F.relu(input)
        dz = torch.matmul(dz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights) - 1):
            input = torch.matmul(input, weights[i]) + biases[i]
            input = torch.sigmoid(input)
            dz = torch.mul(input * (1 - input), torch.matmul(dz, weights[i]))
        dz = torch.matmul(dz, weights[-1])
    else:
        for i in range(len(weights) - 1):
            dz = torch.matmul(dz, weights[i])
        dz = torch.matmul(dz, weights[-1])
    
    return dz
