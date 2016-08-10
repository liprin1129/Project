import numpy as np

def polynomial_curve(x, w0, w1):
    return w0 + w1*x

def powerlaw_curve2(x, w0, w1):
    return -w0 * (x)**(-w1) # -w0 * np.power((x), w1) + w2

def powerlaw_curve3(x, w0, w1, w2):
    return -w0 * (x)**(-w1) + w2 # -w0 * np.power((x), w1) + w2

def powerlaw_curve4(x, w0, w1, w2, w3):
    return -w0 * (x+w3)**(-w1) + w2 # -w0 * np.power((x), w1) + w2

def exponential_curve2(x, a, b):
    return a * np.exp(-b * x)

def exponential_curve3(x, a, b, c):
    return a * np.exp(-b * x) + c