import numpy as np
from scipy.integrate import odeint
from scipy.special import legendre, binom


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l


def generate_pred_prey(z0, t, alpha=0.1, beta=0.1, delta=0.1, gamma=0.1):
    """
    Simulate the predator-prey dynamics.

    Arguments:
        z0 - Initial condition in the form of a 2-value list or array.
        t - Array of time points at which to simulate.
        alpha, beta, delta, gamma - Predator-prey parameters.

    Returns:
        z, dz, ddz - Arrays of the trajectory values and their 1st and 2nd derivatives.
    """
    def lotka_volterra(z, t):
        x, y = z
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return [dxdt, dydt]

    # Simulate the system
    z = odeint(lotka_volterra, z0, t)

    # Compute dz/dt using the system equations directly
    dz = np.array([lotka_volterra(zi, ti) for zi, ti in zip(z, t)])

    # Compute d²z/dt² numerically from dz
    ddz = np.gradient(dz, t, axis=0)

    return z, dz, ddz


def generate_pred_prey_data(ics, t, n_points, linear=True, normalization=None,
                            alpha=0.1, beta=0.1, delta=0.1, gamma=0.1):
    """
    Generate high-dimensional predator-prey data set.

    Arguments:
        ics - Nx2 array of N initial conditions.
        t - Array of time points over which to simulate.
        n_points - Size of the high-dimensional dataset created.
        linear - Boolean value. If True, high-dimensional dataset is a linear combination
                 of the predator-prey dynamics. If False, the dataset also includes quadratic modes.
        normalization - Optional 2-value array for rescaling the 2 predator-prey variables.
        alpha, beta, delta, gamma - Parameters of the predator-prey dynamics.

    Returns:
        data - Dictionary containing elements of the dataset.
    """
    n_ics = ics.shape[0]
    n_steps = t.size
    d = 2
    z = np.zeros((n_ics, n_steps, d))
    dz = np.zeros(z.shape)
    ddz = np.zeros(z.shape)

    for i in range(n_ics):
        z[i], dz[i], ddz[i] = generate_pred_prey(ics[i], t, alpha=alpha, beta=beta, delta=delta, gamma=gamma)

    if normalization is not None:
        z *= normalization
        dz *= normalization
        ddz *= normalization

    n = n_points
    L = 1
    y_spatial = np.linspace(-L, L, n)

    modes = np.zeros((2 * d, n))
    for i in range(2 * d):
        modes[i] = legendre(i)(y_spatial)

    x = np.zeros((n_ics, n_steps, n))
    dx = np.zeros(x.shape)
    ddx = np.zeros(x.shape)

    for i in range(n_ics):
        for j in range(n_steps):
            x1 = modes[0] * z[i, j, 0]
            x2 = modes[1] * z[i, j, 1]
            x3 = modes[2] * z[i, j, 0]**2
            x4 = modes[3] * z[i, j, 1]**2

            x[i, j] = x1 + x2 + x3 + x4 if not linear else x1 + x2
            dx[i, j] = modes[0] * dz[i, j, 0] + modes[1] * dz[i, j, 1]
            ddx[i, j] = modes[0] * ddz[i, j, 0] + modes[1] * ddz[i, j, 1]

    data = {
        't': t,
        'y_spatial': y_spatial,
        'modes': modes,
        'x': x,
        'dx': dx,
        'ddx': ddx,
        'z': z,
        'dz': dz,
        'ddz': ddz,
    }

    return data


def get_pred_prey_data(n_ics, noise_strength=0):
    """
    Generate a set of predator-prey training data for multiple random initial conditions.

    Arguments:
        n_ics - Integer specifying the number of initial conditions to use.
        noise_strength - Amount of noise to add to the data.

    Returns:
        data - Dictionary containing elements of the dataset.
    """
    t = np.arange(0, 50, 0.02)
    n_steps = t.size
    input_dim = 128

    ic_means = np.array([10, 5])
    ic_widths = 2 * np.array([10, 5])

    # Generate initial conditions
    ics = ic_widths * (np.random.rand(n_ics, 2) - 0.5) + ic_means
    data = generate_pred_prey_data(ics, t, input_dim, linear=False, normalization=np.array([1 / 40, 1 / 40]))
    data['x'] = data['x'].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    data['dx'] = data['dx'].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    data['ddx'] = data['ddx'].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)

    return data