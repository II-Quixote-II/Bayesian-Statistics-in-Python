import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('NZDUSD.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.dropna(subset=['log_return'])
returns = df['log_return'].values

sigma2 = np.var(returns, ddof=1)          # known variance
mu0, tau0 = 0.0, 0.001                    # weak prior N(0, 0.001)

x = np.linspace(-0.001, 0.001, 500)
plt.figure(figsize=(10, 5))
plt.plot(x, stats.norm.pdf(x, mu0, np.sqrt(tau0)),
         lw=2, ls='--', color='gray', label='Prior')

for n, c in zip([50, 500, 2000, len(returns)],
                ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
    data = returns[:n]
    ybar = np.mean(data)
    post_prec = 1/tau0 + n/sigma2
    post_var  = 1/post_prec
    post_mean = (mu0/tau0 + n*ybar/sigma2) / post_prec
    plt.plot(x, stats.norm.pdf(x, post_mean, np.sqrt(post_var)),
             lw=2, color=c, label=f'Posterior (n={n})')

plt.title('Bayesian Updating of Mean Log-Return (Normal-Normal Conjugate)')
plt.xlabel('μ'); plt.ylabel('Density'); plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
