import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad

class ExponentialDecompositionBathCorrelation:
    """
    Class to calculate bath correlation functions using the definitions from the research article 'High
    accuracy exponential decomposition of bath correlation functions for arbitrary and structured spectral
    densities: Emerging methodologies and new approaches' by Hideaki Takahashi, Samuel Rudge; Christoph Kaspar,
    Michael Toss; Raffaele Borrelli
    """
    def __init__(self, eta, omega_c, beta, s, num_exponentials=5):
        """
        Initialize the parameters for the bath correlation function.

        Parameters:
        eta (float): Coupling strength.
        omega_c (float): Cutoff frequency.
        beta (float): Inverse temperature (1 / (k_B * T)).
        s (float): Exponent for Ohmic (s=1) or super-Ohmic (s>1) spectral densities.
        num_exponentials (int): Number of exponential terms in the decomposition.
        """
        self.eta = eta
        self.omega_c = omega_c
        self.beta = beta
        self.s = s
        self.num_exponentials = num_exponentials
        self.exp_coefficients = self._compute_exponential_decomposition()

    def spectral_density(self, omega):
        """
        Calculate the spectral density J(omega).

        Parameters:
        omega (float): Frequency.

        Returns:
        float: Value of J(omega).
        """
        return self.eta * (omega ** self.s) * np.exp(-omega / self.omega_c)

    def _correlation_function_integrand(self, omega, t):
        """
        Integrand for the correlation function C(t).

        Parameters:
        omega (float): Frequency.
        t (float): Time.

        Returns:
        float: Value of the integrand.
        """
        return self.spectral_density(omega) * np.cos(omega * t) * (1 + 1 / (np.exp(self.beta * omega) - 1))

    def _compute_exponential_decomposition(self):
        """
        Decompose the bath correlation function into exponential terms.

        Returns:
        list of tuples: List of coefficients (amplitude, decay rate) for each exponential term.
        """
        def correlation_function_fit(t, *params):
            """
            Fit function as a sum of exponentials.

            Parameters:
            t (array): Time values.
            params (tuple): Flattened list of amplitudes and decay rates.

            Returns:
            array: Fitted correlation function values.
            """
            exp_sum = 0
            for i in range(self.num_exponentials):
                amplitude = params[i]
                decay_rate = params[self.num_exponentials + i]
                exp_sum += amplitude * np.exp(-decay_rate * t)
            return exp_sum

        # Numerical calculation of the precise correlation function for fitting
        time_values = np.linspace(0, 10 / self.omega_c, 100)
        precise_correlation = np.array([quad(self._correlation_function_integrand, 0, np.inf, args=(t,))[0] for t in time_values])

        # Initial guesses for exponential amplitudes and decay rates
        initial_amplitudes = [1.0 / self.num_exponentials] * self.num_exponentials
        initial_decay_rates = [self.omega_c * (i + 1) for i in range(self.num_exponentials)]
        initial_guess = initial_amplitudes + initial_decay_rates

        # Fit to get amplitudes and decay rates
        fitted_params, _ = curve_fit(correlation_function_fit, time_values, precise_correlation, p0=initial_guess)

        amplitudes = fitted_params[:self.num_exponentials]
        decay_rates = fitted_params[self.num_exponentials:]
        return list(zip(amplitudes, decay_rates))

    def bath_correlation_function(self, t):
        """
        Calculate the bath correlation function C(t) using the exponential decomposition.

        Parameters:
        t (float): Time.

        Returns:
        float: Value of C(t).
        """
        correlation = sum(A * np.exp(-gamma * t) for A, gamma in self.exp_coefficients)
        return correlation

    def correlation_spectrum(self, omega):
        """
        Calculate the Fourier transform of the bath correlation function.

        Parameters:
        omega (float): Frequency.

        Returns:
        float: Value of the spectrum at omega.
        """
        spectrum = sum(2 * A * gamma / (gamma ** 2 + omega ** 2) for A, gamma in self.exp_coefficients)
        return spectrum


# Example usage
if __name__ == "__main__":
    eta = 1.0        # Coupling strength
    omega_c = 1.0    # Cutoff frequency
    beta = 1.0       # Inverse temperature
    s = 1.0          # Ohmic spectral density
    num_exponentials = 5

    bath_corr = ExponentialDecompositionBathCorrelation(eta, omega_c, beta, s, num_exponentials)

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