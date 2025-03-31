import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

t_vals = np.linspace(0,10, 5)

plt.figure(figsize=(8, 6))
plt.plot(t_vals, C_vals.real, label="Re[C(t)] (Original)", color='blue')
plt.plot(t_vals, C_approx_vals.real, '--', label="Re[C(t)] (Approx)", color='red')

plt.plot(t_vals, C_vals.imag, label="Im[C(t)] (Original)", color='green')
plt.plot(t_vals, C_approx_vals.imag, '--', label="Im[C(t)] (Approx)", color='orange')

plt.xlabel("w/w_c")
plt.ylabel("C(t)")
plt.title("Original vs Approximated C(t) using Gradient Descent with L2 Regularization")
plt.legend()
plt.show()