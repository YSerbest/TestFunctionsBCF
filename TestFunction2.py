import numpy as np
from scipy.special import gamma


class BathCorrelationFunction:
    """
    Class to calculate bath correlation functions based on the research article 'Analytic
    representations of bath correlation functions for ohmic and superohmic spectral densities using simple poles'
    by Gerhard Ritschel and Alexander Eisfeld.
    """

    def __init__(self, eta, omega_c, beta, s):
        """
        Initialize the parameters for the spectral density and correlation function.

        Parameters:
        eta (float): Coupling strength
        omega_c (float): Cutoff frequency
        beta (float): Inverse temperature (1 / (k_B * T)).
        s (float): Exponent for Ohmic (s=1) or superohmic (s>1) spectral densities.
        """
        self.eta = eta
        self.omega_c = omega_c
        self.beta = beta
        self.s = s

    def spectral_density(self, omega):
        """
        Calculate the spectral density J(omega).

        Parameters:
        omega (float): Frequency.

        Returns:
        float: Value of J(omega).
        """
        return self.eta * (omega ** self.s) * np.exp(-omega / self.omega_c)

    def bath_correlation_function(self, t):
        """
        Calculate the bath correlation function C(t) based on poles.

        Parameters:
        t (float): Time.

        Returns:
        complex: Value of C(t).
        """
        # Use approximation of the poles
        summand_factor= self.eta * gamma(self.s + 1) / (2 * np.pi * self.beta)
        poles = [2j * np.pi * n / self.beta for n in range(1, int(1e3))]

        correlation = 0
        for pole in poles:
            correlation += summand_factor * (1 / (pole ** 2 + self.omega_c ** 2)) * np.exp(-pole * t)

        return correlation

    def correlation_spectrum(self, omega):
        """
        Calculate the Fourier transform of the bath correlation function.

        Parameters:
        omega (float): Frequency.

        Returns:
        complex: Value of the spectrum at the given omega.
        """
        c_re = self.bath_correlation_function(0).real
        return 2 * c_re * omega / (omega ** 2 + self.omega_c ** 2)


# Example usage (parameters are set to 1.0)
if __name__ == "__main__":
    eta = 1.0  # Coupling strength
    omega_c = 1.0  # Cutoff frequency
    beta = 1.0  # Inverse temperature
    s = 1.0  # Exponent for Ohmic spectral density (Not superohmic)

    bath_corr = BathCorrelationFunction(eta, omega_c, beta, s)

    # Calculate bath correlation function for t=1.0
    t = 1.0
    C_t = bath_corr.bath_correlation_function(t)
    print(f"Bath correlation function at t={t}: {C_t}")

    # Calculate spectral density at omega=1.0
    omega = 1.0
    J_omega = bath_corr.spectral_density(omega)
    print(f"Spectral density at omega={omega}: {J_omega}")

    # Calculate Fourier transform spectrum at omega=1.0
    spectrum = bath_corr.correlation_spectrum(omega)
    print(f"Correlation spectrum at omega={omega}: {spectrum}")