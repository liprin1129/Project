import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

############################################
## Polynomial Functioin
############################################

def polynomial_curve(x, w0, w1):
    return w0 + w1*x

def polynomial_least(w, x, y):
    return polynomial_curve(x, w[0], w[1]) - y


############################################
## Exponential Functioin
############################################

def exponential_curve2(x, a, b):
    return a * np.exp(-b * x)

def exponential_curve3(x, a, b, c):
    return a * np.exp(-b * x) + c

def exponential_least2(w, x, y):
    return exponential_curve2(x, w[0], w[1]) - y

def exponential_least3(w, x, y):
    return exponential_curve3(x, w[0], w[1], w[2]) - y


############################################
## Powerlaw Functioin
############################################

def powerlaw_curve2(x, w0, w1):
    return -w0 * (x)**(-w1) # -w0 * np.power((x), w1) + w2

def powerlaw_curve3(x, w0, w1, w2):
    return -w0 * (x)**(-w1) + w2 # -w0 * np.power((x), w1) + w2

def powerlaw_curve4(x, w0, w1, w2, w3):
    return -w0 * (x+w3)**(-w1) + w2 # -w0 * np.power((x), w1) + w2
    
def powerlaw_least2(w, x, y):
    return powerlaw_curve2(x, w[0], w[1]) - y
    
def powerlaw_least3(w, x, y):
    return powerlaw_curve3(x, w[0], w[1], w[2]) - y
    
def powerlaw_least4(w, x, y):
    return powerlaw_curve4(x, w[0], w[1], w[2], w[3]) - y
    
############################################
## About curve fitting
############################################
    
def cost_Function(true_y, pred_y):
    diff = np.power((true_y - pred_y), 2)
    cost = np.sum(diff)
    return cost

def curve_Fitting(func, x, y, name):
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    
    popt, pcov = curve_fit(func, x, y, maxfev = 1000000)

    upper_bound = []
    lower_bound = []
    for i in range(len(pcov)):
        upper_bound.append(popt[i] + pcov[i,i])
        lower_bound.append(popt[i] - pcov[i,i])

    x_fit = np.linspace(0, 15, 100)
    if len(popt) == 1:
        y_mean = func(x_fit, popt[0])
        y_upper = func(x_fit, upper_bound[0])
        y_lower = func(x_fit, lower_bound[0])
        cost = cost_Function(y, func(x, popt[0]))
    
    elif len(popt) == 2:
        y_mean = func(x_fit, popt[0], popt[1])
        y_upper = func(x_fit, upper_bound[0], upper_bound[1])
        y_lower = func(x_fit, lower_bound[0], lower_bound[1])
        cost = cost_Function(y, func(x, popt[0], popt[1]))
    
    elif len(popt) == 3:
        y_mean = func(x_fit, popt[0], popt[1], popt[2])
        y_upper = func(x_fit, upper_bound[0], upper_bound[1], upper_bound[2])
        y_lower = func(x_fit, lower_bound[0], lower_bound[1], lower_bound[2])
        cost = cost_Function(y, func(x, popt[0], popt[1], popt[2]))

    elif len(popt == 4):
        y_mean = func(x_fit, popt[0], popt[1], popt[2], popt[3])
        y_upper = func(x_fit, upper_bound[0], upper_bound[1], upper_bound[2], upper_bound[3])
        y_lower = func(x_fit, lower_bound[0], lower_bound[1], lower_bound[2], upper_bound[3])
        cost = cost_Function(y, func(x, popt[0], popt[1], popt[2], popt[3]))

    print(" - Curve Fitting Parameters: {0}\n".format(popt))
    print(" - Curve Fitting Cost: {0}\n".format(cost))
    print(" - Curve Fitting Covariance: \n{0}".format(pcov))
    
    ax.plot(x, y, 'rx')
    ax.plot(x_fit, y_mean, 'r-', label="curve fitting")    
    for i in range(len(x_fit)):
        if i == 0:
            ax.plot([x_fit[i], x_fit[i]], [y_lower[i], y_upper[i]], 'b-', label="variance")
        else:
            ax.plot([x_fit[i], x_fit[i]], [y_lower[i], y_upper[i]], 'b-')
    
    #ax.set_ylim([0, max(y)+0.05])
    ax.legend()
    fig.savefig('Figs/{0}'.format(name, dpi=100))
    
    return popt, pcov, cost