import pandas as pd
import numpy as np

def generate_data(p, n, d, g, random_state=None):
    """
    p - bernoulli distibution pobability of class 1
    n - number of samples
    d - number of features
    g - covariance decay factor (controls how strongly features are correlated in the covariance matrix) 

    0 < p, g < 1
    n, d >= 1
    """

    if random_state is not None:
        np.random.seed(random_state)
    
    Y = np.random.binomial(n=1, p=p, size=n)
    
    mean_0 = np.zeros(d)
    mean_1 = np.array([1/i for i in range(1, d + 1)])

    S = np.fromfunction(lambda i, j: g ** np.abs(i - j), (d, d), dtype=float)
    
    X = np.array([
        np.random.multivariate_normal(mean_0 if y == 0 else mean_1, S)
        for y in Y
    ])
    
    return X, Y