#Kevin Macario
import numpy as np


def linear_cost(X, y, theta, lambda_x):
    m, _ = X.shape
    h = np.matmul(X, theta)
    sq = (y - h) ** 2
    reg = theta ** 2
    return (sq.sum() + np.sum(lambda_x * reg))/ (2 * m) 
