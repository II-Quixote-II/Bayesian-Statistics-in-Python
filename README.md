# Bayesian Statistics for NZD/USD Mean Daily Return

This project applies Bayesian methods to estimate the mean daily return of the NZD/USD exchange rate.  Rather than relying on
a single point estimate - in this case, a prior belief about the NZD/USD mean return is updated with observed market data to produce
a full posterior distribution, capturing both the estimate = and the uncertainty around the NZD/USD rate.

## Overview:

### Goal: 
Estimate the Mean Daily return of NZD/USD while quantifying the uncertainty around the estimate

Files:

- NZD/USD.csv | Raw Daily Exchange rate data- Sourced from Kaggle
- Bayesian_updating.py | Performs Bayesian Update and computes returns
- Posterior_mean.py | Visualises the resulting posterior distribution

  
