import numpy as np

def silverman_bandwidth(sample):
    n = np.size(sample)
    std = np.std(sample)
    return 1.06 * std * n ** (-1/5)

def robast_silverman_bandwidth(sample):
    n = np.size(sample)
    iqr = np.quantile(sample, 0.75) - np.quantile(sample, 0.25)
    std = np.std(sample)
    return 0.9 * min(std, iqr / 1.34) * n ** (-1/5)

def gaussian_kde(x_grid, sample, h):
    n = np.size(sample)
    kde = np.zeros_like(x_grid)
    for xi in sample:
        kde += 1/np.sqrt(2 * np.pi) * np.exp(-((x_grid - xi)/h) ** 2 / 2)
    
    kde /=  n * h
    return kde

