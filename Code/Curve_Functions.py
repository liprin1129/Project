import numpy as np

def polynomial_curve(x, w0, w1):
    return w0 + w1*x

def powerlaw_curve(x, w0, w1, w2):
    return w0 * np.power(x, -w1) + w2 # -w0 * x**(w1) + w2

def exponential_curve(x, a, b, c):
    return a * np.exp(-b * x) + c