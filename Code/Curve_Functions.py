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
    
def exponential_curve4(x, a, b, c, d):
    return a * np.exp(b * -(x+d)) + c

def exponential_least2(w, x, y):
    return exponential_curve2(x, w[0], w[1]) - y

def exponential_least3(w, x, y):
    return exponential_curve3(x, w[0], w[1], w[2]) - y
    
def exponential_least4(w, x, y):
    return exponential_curve4(x, w[0], w[1], w[2], w[3]) - y


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
    ax.set_title("Cluster {0} (Cost {1})".format(clt_num, round(cost, 2)))
    # ax.text(0.77, 0.03, "cost: {0}".format(round(cost, 2)), horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=15)
    fig.savefig(file_path, dpi=100)
    
    return lsq.x, cost
    
    
############################################
## About multipe curve fitting
############################################
    
def multi_curveFitting_2(least_func, avg, seed, min_range=5):
    cost = []
    #param1 = np.ones((n_param, 300))
    #param2 = np.ones((n_param, 300))
    
    x_range = np.linspace(1, 300, 300)
    for n in range( int(300/min_range) - 1) : # iteration for all data
        # print("iteration ", n)

        x1 = x_range[:min_range*(n+1)]
        x2 = x_range[min_range*(n+1):]

        #print('\n\n - x1:', x1)
        #print(' - x2:', x2)
        
        y1 = avg[:min_range*(n+1)]
        y2 = avg[min_range*(n+1):]

        lsq1 = least_squares(least_func, seed, args=(x1, y1))
        lsq2 = least_squares(least_func, seed, args=(x2, y2))
        
        #param1[:, n] = lsq1.x
        #param2[:, n] = lsq2.x

        cost.append(lsq1.cost+lsq2.cost)
        
    idx = np.argmin(cost)
    return min_range*(idx+1)#, param1[:, idx], param2[:, idx]

def multi_curveFitting_3(least_func, avg, seed, min_range=5):
    cost = []
    break_point2 = []
    #idx_mid2 = [] # save idx2(second change)
    
    x_range = np.linspace(1, 300, 300)
    
    end1 = 0
    end2 = 0

    first_idx = []
    for n in range(int(300/min_range) - 2): # iteration for all data
        # print("\n - iter{0}".format(n))
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
            # print("iter {0}-{1}".format(n, j))
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
    
def multi_curveFitting_4(least_func, avg, seed, min_range=5):
    cost = []
    break_point2 = []
    #idx_mid2 = [] # save idx2(second change)
    
    x_range = np.linspace(1, 300, 300)
    
    end1 = 0
    end2 = 0

    first_idx = []
    for n in range(int(300/min_range) - 2): # iteration for all data
        # print("\n - iter{0}".format(n))
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
            # print("iter {0}-{1}".format(n, j))
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

def curve_Matrix(y_data, least_func, seed=[1,1], window=10, piece=4):
    ## set initial x ranges
    x_range = np.linspace(1, 300, 300)
    y_range = y_data
    for i in range(piece):
        if i < piece-1:
            locals()["x{0}".format(i)] = x_range[i*window:window*(i+1)]
            locals()["y{0}".format(i)] = y_range[i*window:window*(i+1)]
        else:
            locals()["x{0}".format(i)] = x_range[i*window:]
            locals()["y{0}".format(i)] = y_range[i*window:]
        # print("x:[{0}, {1}]".format(eval("x{0}".format(i))[0], eval("x{0}".format(i))[-1]))
        # print("y:[{0}, {1}]".format(eval("y{0}".format(i))[0], eval("y{0}".format(i))[-1]))
    
    ## make matrix
    count = 0
    mat_iter = len(eval("x{0}".format(piece-1)))
    while(mat_iter>window):
        mat_iter = mat_iter-window
        #print(mat_iter)
        count = count+1
    
    err_matrix = np.zeros([piece, count+1])
    idx_matrix = np.zeros([piece, count+1])
    print(np.shape(err_matrix))
    
    ## change window for pieces except first pieces
    for group in range(piece-1):
        #print("\n\n - pieace ", piece-group-1)
        #print(" - x{0}:\n".format(piece-group-1), "[{0}, {1}]".format(eval("x{0}".format(piece-group-1))[0], eval("x{0}".format(piece-group-1))[-1]))        
        partition = 0
        
        lsq = least_squares(least_func, seed, args=(eval("x{0}".format(piece-group-1)), eval("y{0}".format(piece-group-1)))) # function fitting
        err_matrix[piece-group-1, partition] = lsq.cost # save cost
        idx_matrix[piece-group-1, partition] = len(eval("x{0}".format(piece-group-1))) # save x values
        #print("cost!!:", lsq.cost)
        
        while( len(eval("x{0}".format(piece-group-1))-window) > window):
            locals()["x{0}".format(piece-group-1)] = eval("x{0}".format(piece-group-1))[window:] # 마지막 piece의 첫번째를 window만큼 더한다.
            locals()["y{0}".format(piece-group-1)] = eval("y{0}".format(piece-group-1))[window:] # 마지막 piece의 첫번째를 window만큼 더한다.
            #print("[{0}, {1}]".format(eval("x{0}".format(piece-group-1))[0], eval("x{0}".format(piece-group-1))[-1]))
            partition = partition+1
            
            lsq = least_squares(least_func, seed, args=(eval("x{0}".format(piece-group-1)), eval("y{0}".format(piece-group-1)))) # function fitting
            err_matrix[piece-group-1, partition] = lsq.cost # save cost
            idx_matrix[piece-group-1, partition] = len(eval("x{0}".format(piece-group-1))) # save x values
            #print("cost!!:", lsq.cost)
            
        end = eval("x{0}".format(piece-group-1))[0]
        #print("end: ", end)
        locals()["x{0}".format(piece-group-2)] = x_range[(piece-group-2)*window:int(end-1)] #처음의 값을 window만큼 이동
        locals()["y{0}".format(piece-group-2)] = y_range[(piece-group-2)*window:int(end-1)] #처음의 값을 window만큼 이동

    ## change window for first piece
    i = 0
    #print("\n\n - piece  0")
    #print(" - x0:\n", "[{0}, {1}]".format(eval("x0")[0], eval("x0")[-1]))
    lsq = least_squares(least_func, seed, args=(eval("x{0}".format(piece-group-1)), eval("y{0}".format(piece-group-1)))) # function fitting
    err_matrix[i, 0] = lsq.cost # save cost
    idx_matrix[i, 0] = len(eval("x0"))
    while( len(eval("x0"))-window):
        end = eval("x0")[-1]
        locals()["x0"] = eval("x0")[:int(end-window)] # 마지막 piece의 첫번째를 window만큼 더한다.
        locals()["y0"] = eval("y0")[:int(end-window)] # 마지막 piece의 첫번째를 window만큼 더한다.
        #print("[{0}, {1}]".format(eval("x0")[0], eval("x0")[-1]))
        i = i+1
        
        lsq = least_squares(least_func, seed, args=(eval("x0"), eval("y0"))) # function fitting
        err_matrix[0, i] = lsq.cost # save cost
        idx_matrix[0, i] = len(eval("x0"))
        #print("cost!!:", lsq.cost)
                                               
    # print(err_matrix)
    return idx_matrix, err_matrix
    
def curve3_Fitting(idxM, errM):
    groups, mat_iter = np.shape(idxM)
    # print(groups, mat_iter)
    pre_cost = 0
    min_cost = float('nan')
    min_p = [np.nan]*groups

    ## 1st piece
    for i1 in range(mat_iter):
        # print('\n')
        p1 = idxM[0, i1]
        c1 = errM[0, i1]
        
        ## 2nd piece
        for i2 in range(mat_iter):
            p2 = idxM[1, i2]
            c2 = errM[1, i2]

            ## 3rd piece
            for i3 in range(mat_iter):
                p3 = idxM[2, i3]
                c3 = errM[2, i3]

                true_range = p1+p2+p3
                
                ## find 300 ranges
                if(true_range==300):
                    # print(p1, p2, p3, ":", true_range)
                    
                    sum_cost = c1 + c2 + c3
                    
                    ## check a minimum cost
                    if pre_cost > sum_cost:
                        min_cost = sum_cost
                        min_p = p1, p2, p3
                    
                    pre_cost = sum_cost

    print("min_cost: ", min_cost, "at {0}".format(min_p))
    return min_cost, min_p