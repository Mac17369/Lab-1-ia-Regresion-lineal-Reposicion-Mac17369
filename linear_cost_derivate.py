#Kevin Macario
import numpy as np


def linear_cost_derivate(X, y, theta, lambda_x):
    h = np.matmul(X, theta)
    m, _ = X.shape
    reg = (lambda_x / m) * theta.sum()
    return ((np.matmul((h - y).T, X).T) + reg) / m
