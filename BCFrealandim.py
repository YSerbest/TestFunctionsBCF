import numpy as np
from scipy.integrate import quad

class BathCorrelationFunctionIntegral:
    """
    Class to compute the bath correlation function using the integral definition.
    """
    def __init__(self, eta, omega_c, beta, s, hbar=1.0):
        """
        Initialize the parameters for the bath correlation function.

        Parameters:
        eta (float): Coupling strength.
        omega_c (float): Cutoff frequency.
        beta (float): Inverse temperature (1 / (k_B * T)).
        s (float): Exponent for Ohmic (s=1) or super-Ohmic (s>1) spectral densities.
        hbar (float): Reduced Planck constant (default 1.0 for normalized units).
        """
        self.eta = eta
        self.omega_c = omega_c
        self.beta = beta
        self.s = s
        self.hbar = hbar

    def spectral_density(self, omega):
        """
        Compute the spectral density J(omega).

        Parameters:
        omega (float): Frequency.

        Returns:
        float: Value of J(omega).
        """
        return self.eta * (omega ** self.s) * np.exp(-omega / self.omega_c) if omega >= 0 else 0

    def coth(self, x):
        """
        Compute the hyperbolic cotangent coth(x).

        Parameters:
        x (float): Input value.

        Returns:
        float: Value of coth(x).
        """
        return 1 / np.tanh(x) if x != 0 else np.inf

    def correlation_function_integrand(self, omega, t):
        """
        Compute the integrand for the bath correlation function.

        Parameters:
        omega (float): Frequency.
        t (float): Time.

        Returns:
        complex: Value of the integrand.
        """
        J_omega = self.spectral_density(abs(omega))
        thermal_factor = self.coth(self.beta * self.hbar * omega / 2) + 1
        return (1 / (2 * np.pi)) * J_omega * thermal_factor * np.exp(-1j * omega * t)

    def bath_correlation_function(self, t):
        """
        Calculate the bath correlation function C(t) using numerical integration.

        Parameters:
        t (float): Time.

        Returns:
        complex: Value of C(t).
        """
        real_part, _ = quad(
            lambda omega: self.correlation_function_integrand(omega, t).real,
            -100 * self.omega_c, 100 * self.omega_c
        )
        im_part, _ = quad(
            lambda omega: self.correlation_function_integrand(omega, t).imag,
            -100 * self.omega_c, 100 * self.omega_c
        )
        return real_part + 1j * im_part

# Example usage
if __name__ == "__main__":
    eta = 1.0        # Coupling strength
    omega_c = 1.0    # Cutoff frequency
    beta = 1.0       # Inverse temperature
    s = 1.0          # Ohmic spectral density

    bath_corr = BathCorrelationFunctionIntegral(eta, omega_c, beta, s)

    # Calculate bath correlation function for t=1.0
    t = 1.0
    C_t = bath_corr.bath_correlation_function(t)
    print(f"Bath correlation function at t={t}: {C_t}")
