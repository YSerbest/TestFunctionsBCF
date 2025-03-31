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



# Approximation function
def C_t_approximation(t, g_j, omega_j):
    return np.sum(g_j * np.exp(-1j * omega_j * t))


# Compute gradient of cost function
def compute_gradient(t_vals, C_vals, g_j, omega_j, learning_rate):
    K = len(omega_j)
    grad_g = np.zeros(K)  # Gradient for g_j (real)
    grad_omega_r = np.zeros(K)  # Gradient for Re(omega_j)
    grad_omega_i = np.zeros(K)  # Gradient for Im(omega_j)

    for i, t in enumerate(t_vals):
        C_approx = C_t_approximation(t, g_j, omega_j)
        error = C_approx - C_vals[i]

        for j in range(K):
            exp_term = np.exp(-1j * omega_j[j] * t)

            # Gradient w.r.t. g_j (real)
            grad_g[j] += 2 * np.real(error * np.conj(exp_term)) / len(t_vals)

            # Gradient w.r.t. Re(omega_j)
            grad_omega_r[j] += 2 * np.real(error * (-1j * g_j[j] * t * exp_term)) / len(t_vals)

            # Gradient w.r.t. Im(omega_j)
            grad_omega_i[j] += 2 * np.real(error * (g_j[j] * t * exp_term)) / len(t_vals)

    # Gradient descent updates
    g_j -= learning_rate * grad_g
    omega_j -= learning_rate * (grad_omega_r + 1j * grad_omega_i)

    return g_j, omega_j


# Gradient Descent Optimization with real g_j and complex omega_j
def optimize_C_t(t_vals, C_vals, K=30, learning_rate=0.001, max_iter=200, tol=1e-6):
    omega_j = np.linspace(0.5, 5, K) + 1j * np.random.rand(K) * 0.1  # Complex frequencies
    g_j = np.random.rand(K)  # Real coefficients

    prev_cost = np.inf
    for iteration in range(max_iter):
        g_j, omega_j = compute_gradient(t_vals, C_vals, g_j, omega_j, learning_rate)

        # Compute cost function
        C_approx_vals = np.array([C_t_approximation(t, g_j, omega_j) for t in t_vals])
        cost = np.sum(np.abs(C_vals - C_approx_vals) ** 2)

        # Check for convergence
        if np.abs(prev_cost - cost) < tol:
            break
        prev_cost = cost

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Cost = {cost}")

    return g_j, omega_j

"""
# Define time values and compute C(t)
t_vals = np.linspace(0, 5, 100)
eta = np.random.uniform(0.01, 2)
omega_c = np.random.uniform(1e-2, 10)
s = np.random.uniform(0.5, 2)
beta = np.random.uniform(0.1, 10)
hbar = 1  # Constant set to 1
"""

t_vals = np.linspace(0, 5, 500)
eta = 1.5
omega_c = 1.5
s = 1.2
beta = 4
hbar = 1


C_vals = np.array([C_t_original(t, eta, omega_c, s, beta, hbar) for t in t_vals])

# Optimize g_j (real) and complex omega_j using gradient descent
K = 30  # Number of exponentials
g_optimized, omega_optimized = optimize_C_t(t_vals, C_vals, K)

# Compute final approximation
C_approx_vals = np.array([C_t_approximation(t, g_optimized, omega_optimized) for t in t_vals])

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(t_vals, C_vals.real, label="Re[C(t)] (Original)", color='blue')
plt.plot(t_vals, C_approx_vals.real, '--', label="Re[C(t)] (Approx)", color='red')

plt.plot(t_vals, C_vals.imag, label="Im[C(t)] (Original)", color='green')
plt.plot(t_vals, C_approx_vals.imag, '--', label="Im[C(t)] (Approx)", color='orange')

plt.xlabel("t")
plt.ylabel("C(t)")
plt.title("Original vs Approximated C(t) using Gradient Descent with Complex Frequencies")
plt.legend()
plt.show()
