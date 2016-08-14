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
## About single curve fitting
############################################
    
def cost_Function(true_y, pred_y):
    diff = np.power((true_y - pred_y), 2)
    cost = np.sum(diff)/2
    return cost

def curve_Fitting(least_func, curve_func, x, y, seed, file_path, clt_num):
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    
    # popt, pcov = curve_fit(func, x, y, maxfev = 1000000)
    '''
    upper_bound = []
    lower_bound = []
    for i in range(len(pcov)):
        upper_bound.append(popt[i] + pcov[i,i])
        lower_bound.append(popt[i] - pcov[i,i])
    '''
    
    x_fit = np.linspace(0, 16, 100)
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
    
    
############################################
## About multipe curve fitting
############################################
    
def multi_curveFitting_2(least_func, clt, avg, seed, n_param=2):
    cost = []
    param1 = np.ones((n_param, 300))
    param2 = np.ones((n_param, 300))
    
    for n in range(300): # iteration for all data
        # print("iteration ", n)

        x1 = np.linspace(1, n, n)
        x2 = np.linspace(n+1, 300, 300-n)

        y1 = avg[:n]
        y2 = avg[n:]

        lsq1 = least_squares(least_func, seed, args=(x1, y1))
        lsq2 = least_squares(least_func, seed, args=(x2, y2))

        cost1 = lsq1.cost
        cost2 = lsq2.cost
        
        param1[:, n] = lsq1.x
        param2[:, n] = lsq2.x

        cost.append(cost1+cost2)
        
    idx = np.argmin(cost)
    return idx, cost[idx], param1[:, idx], param2[:, idx]
    
def multi_curveFitting_3(least_func, avg, seed, n_param=2):
    cost = []
    param1 = np.ones((n_param, 300))
    param2 = np.ones((n_param, 300, 300))
    param3 = np.ones((n_param, 300, 300))
    
    for n1 in range(300): # iteration for all data
        n1 = n1+1
        print("iter ", n1)
        x1 = np.linspace(1, n1, n1)
        y1 = avg[:n1]

        lsq1 = least_squares(least_func, seed, args=(x1, y1))
        cost1 = lsq1.cost
        param1[:, n1] = lsq1.x 
        
        cost_remain = []        
        for n2 in range(300-n1):
            n2 = n2+1
            x2 = np.linspace(n1+1, n1+n2, n2)
            x3 = np.linspace(n2+2, 300, 300-n1-n2)
            print("x1:{0}, x2:{1}, x3:{2}".format(x1, x2, x3))
            
            '''
            y2 = avg[n1:n1+n2]
            y3 = avg[n1+n2:]
            
            lsq2 = least_squares(least_func, seed, args=(x2, y2))
            lsq3 = least_squares(least_func, seed, args=(x3, y3))
    
            cost_remain.append(lsq2.cost+lsq3.cost)

            param2[:, n1, n2] = lsq2.x
            param3[:, n1, n2] = lsq3.x
    
        idx2 = np.argmin(cost_remain) 
        cost.append(cost1+cost_remain[idx2])
        
    idx1 = np.argmin(cost)
    return idx1, idx2, cost[idx1], param1[:, idx1], param2[:, idx1, idx2], param3[:, idx1, idx2]
            '''