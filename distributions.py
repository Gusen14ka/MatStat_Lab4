import numpy as np

def generate_normal(n):
    rng = np.random.default_rng()
    return rng.normal(0, 1, n)

def generate_cauchy(n):
    return np.random.standard_cauchy(n)

def generate_laplace(n):
    return np.random.laplace(0, np.sqrt(1 / 2), n)

def generate_poisson(n):
    return np.random.poisson(5, n)

def generate_uniform(n):
    return np.random.uniform(-np.sqrt(3), np.sqrt(3), n)