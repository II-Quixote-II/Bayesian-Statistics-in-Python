import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('NZDUSD.csv')
x = df['daily_return_pct'].dropna().values / 100

# Prior: N(mu0, tau0^2)
mu0, tau0 = 0.0, 0.005
sigma2 = x.var()
n = len(x)

# Posterior parameters (conjugate normal)
tau0_sq = tau0**2
mu_n = (sigma2 * mu0 + n * tau0_sq * x.mean()) / (sigma2 + n * tau0_sq)
tau_n_sq = (sigma2 * tau0_sq) / (sigma2 + n * tau0_sq)

posterior = stats.norm(loc=mu_n, scale=np.sqrt(tau_n_sq))

xs = np.linspace(mu_n - 4*np.sqrt(tau_n_sq), mu_n + 4*np.sqrt(tau_n_sq), 500)
plt.figure(figsize=(8, 3))
plt.plot(xs, posterior.pdf(xs), label=f'Posterior: N({mu_n:.5f}, {tau_n_sq:.8f})')
plt.axvline(x=0, color='red', linestyle='--', label='Zero return')
plt.title('Bayesian Posterior of Mean Daily Return (Conjugate Normal)')
plt.legend()
plt.show()

print(f"Posterior mean: {mu_n:.6f} ({mu_n*100:.4f}%)")
print(f"95% Credible Interval: [{posterior.ppf(0.03)*100:.4f}%, {posterior.ppf(0.97)*100:.4f}%]")
