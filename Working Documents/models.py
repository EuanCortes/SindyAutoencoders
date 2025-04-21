import torch
import torch.nn as nn
import itertools

# ========== ENCODER ==========
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

# ========== DECODER ==========
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.net(z)

# ========== AUTOENCODER ==========
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim)

        # Initialize SINDy coefficients as a parameter
        self.sindy_coefficients = nn.Parameter(
            torch.zeros((1, 1)), requires_grad=False
        )  # Placeholder, will be resized during training

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def update_sindy_coefficients(self, Xi):
        """
        Update the SINDy coefficients (Xi) during training.
        """
        self.sindy_coefficients.data = Xi

# ========== SINDY LIBRARY ==========
def build_library(z, poly_order=3, include_sine=False):
    """
    Generate a SINDy library Θ(z) from input z.

    z: Tensor of shape (batch, latent_dim)
    returns: Θ(z) of shape (batch, num_features)
    """
    batch_size, latent_dim = z.shape
    theta = [torch.ones((batch_size, 1), device=z.device)]  # constant term

    # Polynomial terms
    for order in range(1, poly_order + 1):
        for combo in itertools.combinations_with_replacement(range(latent_dim), order):
            term = torch.ones((batch_size,), device=z.device)
            for i in combo:
                term *= z[:, i]
            theta.append(term.unsqueeze(1))

    # Optional sine terms
    if include_sine:
        theta.extend([torch.sin(z[:, i]).unsqueeze(1) for i in range(latent_dim)])

    return torch.cat(theta, dim=1)  # Shape: (batch, num_library_terms)
