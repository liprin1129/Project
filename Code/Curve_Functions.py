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
    
def multi_curveFitting_3(least_func, avg, seed, n_curve=2):
    cost = []
    break_point2 = []
    #idx_mid2 = [] # save idx2(second change)
    
    x_range = np.linspace(1, 300, 300)
    min_range = 20 # minimum x_range
    
    end1 = 0
    end2 = 0

    first_idx = []
    for n in range(int(300/min_range) - 2): # iteration for all data
        print("\n - iter{0}".format(n))
        x_idx = 0
        if x_idx == 0:
            end1 = min_range*(n+1) # caculate the first range limit
            x1 = x_range[:end1]
            y1 = avg[:end1]
            lsq1 = least_squares(least_func, seed, args=(x1, y1))
            #print('x1', x1)
            first_idx.append(end1)
        
        second_idx = []
        second_cost = []
        for j in range(int( (300-(min_range*(n+2))) / min_range) ): # iteration for 2nd and 3rd x_range
            print("iter {0}-{1}".format(n, j))
            end2 = min_range*(j+1) + end1 # caculate the second range limit
            x2 = x_range[end1:end2]
            y2 = avg[end1:end2]
            lsq2 = least_squares(least_func, seed, args=(x2, y2))
            #print('x2', x2)
            
            x3 = x_range[end2:]
            y3 = avg[end2:]
            lsq3 = least_squares(least_func, seed, args=(x3, y3))
            #print('x3', x3)

            second_idx.append(end2) # save 2nd break points
            second_cost.append(lsq1.cost + lsq2.cost + lsq3.cost) # save costs
        
        break_point2.append(second_idx[np.argmin(second_cost)]) # get index where cost of remained x_ranges is minimum
        cost.append(second_cost) # save costs
        
    point1 = np.argmin(cost) # get array index of cost is minimum
    point2 = break_point2[point1] # get index of 2nd break point
            
    return min_range*(point1+1), point2