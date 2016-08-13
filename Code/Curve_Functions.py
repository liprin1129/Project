import numpy as np
# from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
#import DC_Pickle as dcp

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
    return a * np.exp(b * x)

def exponential_curve3(x, a, b, c):
    return a * np.exp(b * -x) + c

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

def curve_Fitting(least_func, curve_func, x, y, seed, file_path, clt_num):
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    
    # popt, pcov = curve_fit(func, x, y, maxfev = 1000000)
    '''
    upper_bound = []
    lower_bound = []
    for i in range(len(pcov)):
        upper_bound.append(popt[i] + pcov[i,i])
        lower_bound.append(popt[i] - pcov[i,i])
    '''
    
    x_fit = np.linspace(0, 15, 100)
    '''
    if seed == 1:
        lsq = least_squares(least_func, seed, args=(x, y))
        y_mean = curve_func(x_fit, lsq.x)
        cost = lsq.cost
    '''
    
    if len(seed) == 2:
        lsq = least_squares(least_func, seed, args=(x, y))
        y_mean = curve_func(x_fit, lsq.x[0], lsq.x[1])
        cost = lsq.cost
    
    elif len(seed) == 3:
        lsq = least_squares(least_func, seed, args=(x, y))
        y_mean = curve_func(x_fit, lsq.x[0], lsq.x[1], lsq.x[2])
        cost = lsq.cost

    elif len(seed) == 4:
        lsq = least_squares(least_func, seed, args=(x, y))
        y_mean = curve_func(x_fit, lsq.x[0], lsq.x[1], lsq.x[2], lsq.x[3])
        cost = lsq.cost

    print(" - Curve Fitting Parameters: {0}".format(lsq.x))    
    print(" - Curve Fitting Cost: {0}\n".format(cost))
    
    ax.plot(x, y, 'rx', label="average score")
    ax.plot(x_fit, y_mean, 'b-', label="curve fitting")    
    '''    
    for i in range(len(x_fit)):
        if i == 0:
            ax.plot([x_fit[i], x_fit[i]], [y_lower[i], y_upper[i]], 'b-', label="variance")
        else:
            ax.plot([x_fit[i], x_fit[i]], [y_lower[i], y_upper[i]], 'b-')
    
    '''
    ax.set_ylim([0, max(y)+0.2])
    ax.legend(fontsize=14)
    ax.set_title("cluster {0}: cost {1}".format(clt_num, round(cost, 2)))
    # ax.text(0.77, 0.03, "cost: {0}".format(round(cost, 2)), horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=15)
    fig.savefig(file_path, dpi=100)
    
    return lsq.x, cost