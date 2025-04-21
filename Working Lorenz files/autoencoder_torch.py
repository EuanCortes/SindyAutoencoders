import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoencoderNetwork(nn.Module):
    def __init__(self, params):
        super(AutoencoderNetwork, self).__init__()
        self.params = params
        
        # Network architecture
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # SINDy coefficients
        self.sindy_coefficients = nn.Parameter(
            torch.randn(params['library_dim'], params['latent_dim'])
        )
        nn.init.xavier_uniform_(self.sindy_coefficients)
        
        if params['sequential_thresholding']:
            self.register_buffer('coefficient_mask', 
                               torch.ones(params['library_dim'], params['latent_dim']))
    
    def _build_encoder(self):
        layers = []
        input_dim = self.params['input_dim']
        widths = self.params.get('widths', [64, 32])
        
        # Encoder layers
        for i, width in enumerate(widths):
            layers.append(nn.Linear(input_dim if i == 0 else widths[i-1], width))
            if self.params['activation'] == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif self.params['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif self.params['activation'] == 'elu':
                layers.append(nn.ELU())
        
        # Final layer to latent space
        layers.append(nn.Linear(widths[-1], self.params['latent_dim']))
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        layers = []
        widths = self.params.get('widths', [64, 32])
        
        # Decoder layers
        for i, width in enumerate(reversed(widths)):
            layers.append(nn.Linear(
                self.params['latent_dim'] if i == 0 else widths[len(widths)-i],
                width
            ))
            if self.params['activation'] == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif self.params['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif self.params['activation'] == 'elu':
                layers.append(nn.ELU())
        
        # Final layer to input space
        layers.append(nn.Linear(widths[0], self.params['input_dim']))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encode to latent space
        z = self.encoder(x)
        
        # Decode back to input space
        x_decode = self.decoder(z)
        
        # Build SINDy library
        Theta = self._sindy_library(z)
        
        # Apply coefficient mask if using sequential thresholding
        if hasattr(self, 'coefficient_mask'):
            sindy_predict = torch.matmul(Theta, self.coefficient_mask * self.sindy_coefficients)
        else:
            sindy_predict = torch.matmul(Theta, self.sindy_coefficients)
        
        # Calculate dz (latent space derivative)
        dz = sindy_predict
        
        # Calculate dx_decode (reconstructed space derivative)
        if self.training:  # Only compute gradients during training
            dx_decode = self._calculate_decoder_derivative(z, dz)
        else:
            # For validation, use a simpler approximation or disable gradient computation
            with torch.no_grad():
                dx_decode = torch.zeros_like(x_decode)
                if torch.is_grad_enabled():
                    dx_decode = self._calculate_decoder_derivative(z, dz)
        
        return {
            'x': x,
            'x_decode': x_decode,
            'z': z,
            'dz': dz,
            'dx_decode': dx_decode,
            'sindy_predict': sindy_predict,
            'sindy_coefficients': self.sindy_coefficients,
            'coefficient_mask': getattr(self, 'coefficient_mask', None),
            'Theta': Theta
        }
    
    def _calculate_decoder_derivative(self, z, dz):
        """
        Calculate the derivative through the decoder network using automatic differentiation
        """
        # Make sure z requires gradients
        z = z.detach().requires_grad_(True)
        
        # Compute decoder output
        x_decode = self.decoder(z)
        
        # Compute gradient for each output dimension
        input_dim = self.params['input_dim']
        dx_decode = torch.zeros_like(x_decode)
        
        for i in range(input_dim):
            # Compute gradient of output[:,i] w.r.t. z
            grad_output = torch.zeros_like(x_decode)
            grad_output[:, i] = 1.0
            
            # Compute gradient
            gradients = torch.autograd.grad(
                outputs=x_decode,
                inputs=z,
                grad_outputs=grad_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )[0]
            
            if gradients is not None:
                # Multiply by dz and sum over latent dimensions
                dx_decode[:, i] = torch.sum(gradients * dz, dim=1)
        
        return dx_decode
    
    def _sindy_library(self, z):
        """Build the SINDy library for first-order systems."""
        library = [torch.ones(z.shape[0], device=z.device)]
        
        for i in range(self.params['latent_dim']):
            library.append(z[:, i])
        
        if self.params['poly_order'] > 1:
            for i in range(self.params['latent_dim']):
                for j in range(i, self.params['latent_dim']):
                    library.append(z[:, i] * z[:, j])
        
        if self.params['poly_order'] > 2:
            for i in range(self.params['latent_dim']):
                for j in range(i, self.params['latent_dim']):
                    for k in range(j, self.params['latent_dim']):
                        library.append(z[:, i] * z[:, j] * z[:, k])
        
        if self.params['include_sine']:
            for i in range(self.params['latent_dim']):
                library.append(torch.sin(z[:, i]))
        
        return torch.stack(library, dim=1)

def full_network(params):
    return AutoencoderNetwork(params)