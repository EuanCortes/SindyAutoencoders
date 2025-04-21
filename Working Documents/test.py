import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import Autoencoder, build_library
from generatePredPrey import get_pred_prey_data

# ==== CONFIG ====
input_dim = 128
latent_dim = 2         # You can try different values!
poly_order = 3
include_sine = False
batch_size = 512
num_epochs = 100
lr = 5e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== LOAD DATA ====
data = get_pred_prey_data(n_ics=50, noise_strength=0.01)
x = torch.tensor(data['x'], dtype=torch.float32).to(device)
dx = torch.tensor(data['dx'], dtype=torch.float32).to(device)

# ==== INIT MODEL ====
model = Autoencoder(input_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()

# ==== TRAINING LOOP ====
for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(x.size(0))

    epoch_loss = 0
    for i in range(0, x.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        xb = x[indices]
        dxb = dx[indices]

        xb.requires_grad_(True) 

        x_hat, z = model(xb)

        # Now z is a function of xb, and we can differentiate it
        dz_pred = torch.autograd.grad(
            z, xb, grad_outputs=torch.ones_like(z),
            create_graph=True, retain_graph=True
        )[0]


        # SINDy library and linear regression
        Theta = build_library(z, poly_order, include_sine=include_sine)

        # Solve for Xi (Θ^T Θ + λI)^(-1) Θ^T dz
        lambda_reg = 1e-6
        Xi = torch.linalg.lstsq(
            Theta.T @ Theta + lambda_reg * torch.eye(Theta.shape[1]).to(device),
            Theta.T @ dz_pred
        ).solution

        # Update the SINDy coefficients in the model
        model.update_sindy_coefficients(Xi)

        dz_fit = Theta @ Xi

        # === LOSSES ===
        recon_loss = mse_loss(x_hat, xb)
        dynamics_loss = mse_loss(dz_pred, dz_fit)
        total_loss = recon_loss + 1e-2 * dynamics_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
print("Learned SINDy Coefficients (Xi):")
print(model.sindy_coefficients.data.cpu().numpy())

# ==== OUTPUT ESTIMATED DYNAMICS ====
coefficients = model.sindy_coefficients.data.cpu().numpy()
num_latent_vars = coefficients.shape[0]

dynamics_equations = []
for i in range(num_latent_vars):
    terms = []
    for j, coeff in enumerate(coefficients[i]):
        if abs(coeff) > 1e-4:  # Ignore very small coefficients
            terms.append(f"{coeff:.4f} * Theta_{j}")
    equation = f"z{i+1}' = " + " + ".join(terms) if terms else f"z{i+1}' = 0"
    dynamics_equations.append(equation)

print("Estimated Dynamics:")
for eq in dynamics_equations:
    print(eq)

print(dynamics_loss)

