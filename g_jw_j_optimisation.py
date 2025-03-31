import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit

"""
DEFINE SPECTRAL DENSITY FUNCTION J(w)
"""
def spectral_density(omega, eta, omega_c, s):
    return eta * (omega**s) * np.exp(-omega / omega_c) if omega >= 0 else 0

"""
DEFINE THE ORIGINAL BATH CORRELATION FUNCTION C(t)_original
"""
def coth(x):
    return 1 / np.tanh(x) if x != 0 else np.inf

def C_t_original(t, eta, omega_c, s, beta, hbar, omega_max=10):
    integrand_real = lambda omega: spectral_density(omega, eta, omega_c, s) * (coth(beta * hbar * omega / 2) + 1) * np.cos(omega * t) / (2 * np.pi)
    real_part, _ = quad(integrand_real, 0, omega_max)

    integrand_imag = lambda omega: spectral_density(omega, eta, omega_c, s) * (coth(beta * hbar * omega / 2) + 1) * np.sin(omega * t) / (2 * np.pi)
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
K = 5

"""
DEFINE C(t) USING g_j AND w_j
"""

def C_t_sum_exponentials(t, *params):
    """
    Model function using oscillatory exponentials.
    params = [g_0, w_0, ..., g_K, w_K]
    """
    g = np.array(params[:K])  # Coupling strengths
    w = np.array(params[K:])  # Frequencies

    return np.sum(g[:, None] * np.exp(-1j * w[:, None] * t), axis=0)

"""
USE CURVE FIT TO FIND BEST PARAMETERS
"""

# Initial guesses
g_init = np.random.rand(K)  # Initial coupling strengths
w_init = np.linspace(0.5, 2.0, K)  # Initial frequencies

initial_guess = np.hstack([g_init, w_init])

# Flatten real & imaginary parts of the target function
C_t_flat = np.hstack([C_t_original_values.real, C_t_original_values.imag])

# Define wrapper function to separate real and imaginary parts
def fitting_function(t, *params):
    approx = C_t_sum_exponentials(t, *params)
    return np.hstack([approx.real, approx.imag])  # Separate real and imaginary parts

# Perform curve fitting
params_opt, _ = curve_fit(fitting_function, t_values, C_t_flat, p0=initial_guess, maxfev=100000)

# Extract optimized parameters
g_opt = params_opt[:K]
w_opt = params_opt[K:]

"""
PLOT RESULTS
"""

# Compute fitted C(t) with optimized parameters
C_t_fitted_values = np.array([C_t_sum_exponentials(t, *params_opt) for t in t_values])

# Plot Real Part
plt.figure(figsize=(10, 5))
plt.plot(t_values, C_t_original_values.real, 'o', label="Original Re[C(t)]")
plt.plot(t_values, C_t_fitted_values.real, '--', label="Fitted Re[C(t)]")
plt.xlabel("Time (t)")
plt.ylabel("Re[C(t)]")
plt.legend()
plt.grid()
plt.title("Curve Fitting for Re[C(t)] using g_j and w_j")
plt.show()

# Plot Imaginary Part
plt.figure(figsize=(10, 5))
plt.plot(t_values, C_t_original_values.imag, 'o', label="Original Im[C(t)]")
plt.plot(t_values, C_t_fitted_values.imag, '--', label="Fitted Im[C(t)]")
plt.xlabel("Time (t)")
plt.ylabel("Im[C(t)]")
plt.legend()
plt.grid()
plt.title("Curve Fitting for Im[C(t)] using g_j and w_j")
plt.show()

"""
GRADIENT DESCENT OPTIMIZATION
"""
learning_rate = 0.001
num_iterations = 200
loss_history = []

g_fine = np.copy(g_opt)
w_fine = np.copy(w_opt)

for iteration in range(num_iterations):
    grad_g = np.zeros_like(g_fine)
    grad_w = np.zeros_like(w_fine)

    for j in range(K):
        for i, t in enumerate(t_values):
            e_term = np.exp(-1j * w_fine[j] * t)
            approx_diff = (C_t_original_values[i] - C_t_sum_exponentials(t, *np.hstack([g_fine, w_fine])))
            grad_g[j] += -2 * np.real(approx_diff * e_term).item()
            grad_w[j] += -2 * np.imag(approx_diff * t * e_term * g_fine[j]).item()

    g_fine -= learning_rate * grad_g
    w_fine -= learning_rate * grad_w

    n = len(t_values)  # Number of data points
    current_loss = np.sum(np.abs(C_t_original_values - C_t_sum_exponentials(t_values, *np.hstack([g_fine, w_fine]))) ** 2) / n
    loss_history.append(current_loss)

    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Loss: {current_loss}")

"""
SAVE RESULTS
"""
np.savetxt("fitted_parameters.txt", np.column_stack([g_opt, w_opt]), header="g_j, w_j", fmt="%.6f")
np.savetxt("fine_tuned_parameters.txt", np.column_stack([g_fine, w_fine]), header="g_j, w_j", fmt="%.6f")

print("Fitted parameters saved to fitted_parameters.txt")
print("Fine-tuned parameters saved to fine_tuned_parameters.txt")


"""
SAVE RESULTS TO FILE
"""
output_data = np.column_stack([t_values, C_t_original_values.real, C_t_original_values.imag,
                               C_t_fitted_values.real, C_t_fitted_values.imag])

np.savetxt("curve_fit_results_gj_wj.txt", output_data,
           header="t, Re[C_original], Im[C_original], Re[C_fitted], Im[C_fitted]", fmt="%.6f")

print("Results saved to curve_fit_results_gj_wj.txt")
