import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

"""
DEFINE SPECTRAL DENSITY FUNCTION J(w)
"""


def spectral_density(omega, eta, omega_c, s):
    return eta * (omega ** s) * np.exp(-omega / omega_c) if omega >= 0 else 0


"""
DEFINE THE ORIGINAL BATH CORRELATION FUNCTION C(t)_original
"""


def coth(x):
    return 1 / np.tanh(x) if x != 0 else np.inf


def C_t_original(t, eta, omega_c, s, beta, hbar, omega_max=10):
    integrand_real = lambda omega: spectral_density(omega, eta, omega_c, s) * (
                coth(beta * hbar * omega / 2) + 1) * np.cos(omega * t) / (2 * np.pi)
    real_part, _ = quad(integrand_real, 0, omega_max)

    integrand_imag = lambda omega: spectral_density(omega, eta, omega_c, s) * (
                coth(beta * hbar * omega / 2) + 1) * np.sin(omega * t) / (2 * np.pi)
    imag_part, _ = quad(integrand_imag, 0, omega_max)

    return real_part + 1j * imag_part


# Random Initialization of Parameters
np.random.seed(42)
eta = np.random.uniform(0.01, 2)
omega_c = np.random.uniform(1e-2, 10)
s = np.random.uniform(0.5, 2)
beta = np.random.uniform(40, 1e7)
hbar = 1  # Constant set to 1

# Generate discrete values of C(t)_original
t_values = np.linspace(0, 5, 50)
C_t_original_values = np.array([C_t_original(t, eta, omega_c, s, beta, hbar) for t in t_values])

# Number of exponentials for approximation
K = 15

"""
DEFINE C(t) USING g_j, a_j, and gamma_j
"""


def C_t_sum_exponentials(t, *params):
    g = np.array(params[:K])  # Coupling strengths
    a = np.array(params[K:2 * K])  # Real part of frequencies
    gamma = np.array(params[2 * K:])  # Imaginary part of frequencies
    w = (a + 1j * gamma)  # Complex frequencies

    return np.sum(g[:, None] * np.exp(-1j * w[:, None] * t), axis=0)


"""
GRADIENT DESCENT OPTIMIZATION
"""
# Initialization
g_fine = np.random.rand(K)
a_fine = np.linspace(0.5, 2.0, K)  # Real parts of w_j
gamma_fine = np.random.rand(K) * 0.1  # Small imaginary parts

initial_learning_rate_g = 0.001
initial_learning_rate_a = 0.001
initial_learning_rate_gamma = 0.000001
decay_rate = 0.005  # Controls decay speed
num_iterations = 200
loss_history = []

for iteration in range(num_iterations):
    learning_rate_g = initial_learning_rate_g / (1 + decay_rate * iteration)
    learning_rate_a = initial_learning_rate_a / (1 + decay_rate * iteration)
    learning_rate_gamma = initial_learning_rate_gamma / (1 + decay_rate * iteration)

    grad_g = np.zeros_like(g_fine)
    grad_a = np.zeros_like(a_fine)
    grad_gamma = np.zeros_like(gamma_fine)

    grad_g = np.clip(grad_g, -1, 1)
    grad_a = np.clip(grad_a, -0.1, 0.1)
    grad_gamma = np.clip(grad_gamma, -0.01, 0.01)

    for j in range(K):
        for i, t in enumerate(t_values):
            # Compute exponentials
            e_term = np.exp(-1j * (a_fine[j] + 1j * gamma_fine[j]) * t)  # Negative exponent
            d_exp_dw = -1j * t * e_term  # Derivative w.r.t. w_j
            approx_diff = (C_t_original_values[i] - C_t_sum_exponentials(t, *np.hstack([g_fine, a_fine, gamma_fine])))

            # Compute gradients with corrected derivatives
            grad_g[j] += np.real(approx_diff * e_term).item()
            grad_a[j] += np.imag(approx_diff * d_exp_dw * g_fine[j]).item()  # Real part of frequency
            grad_gamma[j] += np.real(approx_diff * d_exp_dw * g_fine[j]).item()  # Imaginary part of frequency

            # Normalize gradients for stability
            grad_g /= (np.linalg.norm(grad_g) + 1e-8)
            grad_a /= (np.linalg.norm(grad_a) + 1e-8)
            grad_gamma /= (np.linalg.norm(grad_gamma) + 1e-8)

            # Clip updates
            grad_a[j] = np.clip(grad_a[j], -0.1, 0.1)
            grad_gamma[j] = np.clip(grad_gamma[j], -0.01, 0.01)

    # Update parameters
    g_fine -= learning_rate_g * grad_g
    a_fine -= learning_rate_a * grad_a
    gamma_fine -= learning_rate_gamma * grad_gamma
    gamma_fine = np.clip(gamma_fine, -0.5, 0.5)  # Reduce max value

    # Compute loss
    n = len(t_values)
    current_loss = np.sum(
        np.abs(C_t_original_values - C_t_sum_exponentials(t_values, *np.hstack([g_fine, a_fine, gamma_fine]))) ** 2) / n
    loss_history.append(current_loss)

    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Loss: {current_loss}")

"""
PLOT RESULTS
"""
C_t_fitted_values = np.array([C_t_sum_exponentials(t, *np.hstack([g_fine, a_fine, gamma_fine])) for t in t_values])

plt.figure(figsize=(10, 5))
plt.plot(t_values, C_t_original_values.real, 'o', label="Original Re[C(t)]")
plt.plot(t_values, C_t_fitted_values.real, '--', label="Fitted Re[C(t)]")
plt.xlabel("Time (t)")
plt.ylabel("Re[C(t)]")
plt.legend()
plt.grid()
plt.title("Gradient Descent Fit for Re[C(t)]")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_values, C_t_original_values.imag, 'o', label="Original Im[C(t)]")
plt.plot(t_values, C_t_fitted_values.imag, '--', label="Fitted Im[C(t)]")
plt.xlabel("Time (t)")
plt.ylabel("Im[C(t)]")
plt.legend()
plt.grid()
plt.title("Gradient Descent Fit for Im[C(t)]")
plt.show()
