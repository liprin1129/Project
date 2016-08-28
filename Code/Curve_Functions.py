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
    if len(seed) == 1:
        lsq = least_squares(least_func, seed, args=(x, y))
        y_mean = curve_func(x_fit, lsq.x)
        cost = lsq.cost
    
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
'''
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
'''
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
    len_matrix = np.zeros([piece, count+1])
    #print(np.shape(err_matrix))
    
    ## change window for pieces except first pieces
    for group in range(piece-1):
        #print("\n\n - pieace ", piece-group-1)
        #print(" - x{0}:\n".format(piece-group-1), "[{0}, {1}]".format(eval("x{0}".format(piece-group-1))[0], eval("x{0}".format(piece-group-1))[-1]))        
        partition = 0
        
        ## save matrix
        lsq = least_squares(least_func, seed, args=(eval("x{0}".format(piece-group-1)), eval("y{0}".format(piece-group-1)))) # function fitting
        err_matrix[piece-group-1, partition] = lsq.cost # save cost
        idx_matrix[piece-group-1, partition] = eval("x{0}".format(piece-group-1))[0]#len(eval("x{0}".format(piece-group-1))) # save x values
        len_matrix[piece-group-1, partition] = len(eval("x{0}".format(piece-group-1))) # save x length
        #print("cost!!:", lsq.cost)
        
        while( len(eval("x{0}".format(piece-group-1))-window) > window):
            locals()["x{0}".format(piece-group-1)] = eval("x{0}".format(piece-group-1))[window:] # 마지막 piece의 첫번째를 window만큼 더한다.
            locals()["y{0}".format(piece-group-1)] = eval("y{0}".format(piece-group-1))[window:] # 마지막 piece의 첫번째를 window만큼 더한다.
            #print("[{0}, {1}]".format(eval("x{0}".format(piece-group-1))[0], eval("x{0}".format(piece-group-1))[-1]))
            partition = partition+1
            
            ## save matrix
            lsq = least_squares(least_func, seed, args=(eval("x{0}".format(piece-group-1)), eval("y{0}".format(piece-group-1)))) # function fitting
            err_matrix[piece-group-1, partition] = lsq.cost # save cost
            idx_matrix[piece-group-1, partition] = eval("x{0}".format(piece-group-1))[0]# len(eval("x{0}".format(piece-group-1))) # save x values
            len_matrix[piece-group-1, partition] = len(eval("x{0}".format(piece-group-1))) # save x length
            #print("cost!!:", lsq.cost)
            
        end = eval("x{0}".format(piece-group-1))[0]
        #print("end: ", end)
        locals()["x{0}".format(piece-group-2)] = x_range[(piece-group-2)*window:int(end-1)] #처음의 값을 window만큼 이동
        locals()["y{0}".format(piece-group-2)] = y_range[(piece-group-2)*window:int(end-1)] #처음의 값을 window만큼 이동

    ## change window for first piece
    i = 0
    #print("\n\n - piece  0")
    #print(" - x0:\n", "[{0}, {1}]".format(eval("x0")[0], eval("x0")[-1]))
    ## save matrix
    lsq = least_squares(least_func, seed, args=(eval("x{0}".format(piece-group-1)), eval("y{0}".format(piece-group-1)))) # function fitting
    err_matrix[0, count] = lsq.cost # save cost
    idx_matrix[0, count] = eval("x0")[0]# len(eval("x0"))
    len_matrix[0, count] = len(eval("x0"))
    while( len(eval("x0"))-window):
        end = eval("x0")[-1]
        locals()["x0"] = eval("x0")[:int(end-window)] # 마지막 piece의 첫번째를 window만큼 더한다.
        locals()["y0"] = eval("y0")[:int(end-window)] # 마지막 piece의 첫번째를 window만큼 더한다.
        #print("[{0}, {1}]".format(eval("x0")[0], eval("x0")[-1]))
        i = i+1
        
        ## save matrix
        lsq = least_squares(least_func, seed, args=(eval("x0"), eval("y0"))) # function fitting
        err_matrix[0, count-i] = lsq.cost # save cost
        idx_matrix[0, count-i] = eval("x0")[0]
        len_matrix[0, count-i] = len(eval("x0"))
        #print("cost!!:", lsq.cost)
                                               
    # print(err_matrix)
    return idx_matrix, err_matrix, len_matrix
    
def curve2_Fitting(idxM, lenM, errM):
    groups, mat_iter = np.shape(lenM)
    pre_cost = np.nan # an argument to check successive cost
    min_cost = float('nan') # minimum cost
    min_l = [np.nan]*groups # minimum length
    min_bp = [np.nan]*groups # minimum break points
    
    count = 0

    ## 1st piece
    for i1 in range(mat_iter):
        # print('iter ', i1)
        p1 = idxM[0, i1]
        l1 = lenM[0, i1]
        c1 = errM[0, i1]
        
        ## 2nd piece
        for i2 in range(mat_iter):
            p2 = idxM[1, i2]
            l2 = lenM[1, i2]
            c2 = errM[1, i2]

            ## check whether range is 300 and index is valid
            true_range = l1+l2
            if (true_range==300) and (p1<p2):
                sum_cost = c1+c2

                ## check a minimum cost
                if  count==0 or pre_cost > sum_cost:
                    #print("indd!!!!")
                    min_cost = sum_cost
                    min_l = l1, l2
                    min_bp = p1, p2
                    count = count+1

                pre_cost = sum_cost

    print("min_cost: ", min_cost, "at {0} with {1}".format(min_bp, min_l))
    return min_cost, min_l, min_bp

def curve3_Fitting(idxM, lenM, errM):
    groups, mat_iter = np.shape(lenM)
    pre_cost = np.nan # an argument to check successive cost
    min_cost = float('nan') # minimum cost
    min_l = [np.nan]*groups # minimum length
    min_bp = [np.nan]*groups # minimum break points

    count = 0

    ## 1st piece
    for i1 in range(mat_iter):
        # print('iter ', i1)
        p1 = idxM[0, i1]
        l1 = lenM[0, i1]
        c1 = errM[0, i1]
        
        ## 2nd piece
        for i2 in range(mat_iter):
            p2 = idxM[1, i2]
            l2 = lenM[1, i2]
            c2 = errM[1, i2]

            ## 3rd piece
            for i3 in range(mat_iter):
                p3 = idxM[2, i3]
                l3 = lenM[2, i3]
                c3 = errM[2, i3]

                ## check whether range is 300 and index is valid
                true_range = l1+l2+l3
                if (true_range==300) and (p1<p2<p3):#bp_checker.all() == False):
                    #print(bp)
                    sum_cost = c1+c2+c3

                    ## check a minimum cost
                    if  count==0 or pre_cost > sum_cost:
                        #print("indd!!!!", bp)
                        min_cost = sum_cost
                        min_l = l1, l2, l3
                        min_bp = p1, p2, p3
                        count = count+1

                    pre_cost = sum_cost

    print("min_cost: ", min_cost, "at {0} with {1}".format(min_bp, min_l))
    return min_cost, min_l, min_bp
    
def curve4_Fitting(idxM, lenM, errM):
    groups, mat_iter = np.shape(lenM)
    pre_cost = np.nan # an argument to check successive cost
    min_cost = float('nan') # minimum cost
    min_l = [np.nan]*groups # minimum length
    min_bp = [np.nan]*groups # minimum break points

    count = 0

    ## 1st piece
    for i1 in range(mat_iter):
        # print('iter ', i1)
        p1 = idxM[0, i1]
        l1 = lenM[0, i1]
        c1 = errM[0, i1]
        
        ## 2nd piece
        for i2 in range(mat_iter):
            p2 = idxM[1, i2]
            l2 = lenM[1, i2]
            c2 = errM[1, i2]

            ## 3rd piece
            for i3 in range(mat_iter):
                p3 = idxM[2, i3]
                l3 = lenM[2, i3]
                c3 = errM[2, i3]
                
                ## 4th piece
                for i4 in range(mat_iter):
                    p4 = idxM[3, i4]
                    l4 = lenM[3, i4]
                    c4 = errM[3, i4]

                    ## check whether range is 300 and index is valid
                    true_range = l1+l2+l3+l4
                    if (true_range==300) and (p1<p2<p3<p4):
                        sum_cost = c1+c2+c3+c4

                        ## check a minimum cost
                        if  count==0 or pre_cost > sum_cost:
                            #print("indd!!!!")
                            min_cost = sum_cost
                            min_l = l1, l2, l3, l4
                            min_bp = p1, p2, p3, p4
                            count = count+1

                        pre_cost = sum_cost

    print("min_cost: ", min_cost, "at {0} with {1}".format(min_bp, min_l))
    return min_cost, min_l, min_bp
    
def curve5_Fitting(idxM, lenM, errM):
    groups, mat_iter = np.shape(lenM)
    pre_cost = np.nan # an argument to check successive cost
    min_cost = float('nan') # minimum cost
    min_l = [np.nan]*groups # minimum length
    min_bp = [np.nan]*groups # minimum break points

    count = 0

    ## 1st piece
    for i1 in range(mat_iter):
        # print('iter ', i1)
        p1 = idxM[0, i1]
        l1 = lenM[0, i1]
        c1 = errM[0, i1]
        
        ## 2nd piece
        for i2 in range(mat_iter):
            p2 = idxM[1, i2]
            l2 = lenM[1, i2]
            c2 = errM[1, i2]

            ## 3rd piece
            for i3 in range(mat_iter):
                p3 = idxM[2, i3]
                l3 = lenM[2, i3]
                c3 = errM[2, i3]
                
                ## 4th piece
                for i4 in range(mat_iter):
                    p4 = idxM[3, i4]
                    l4 = lenM[3, i4]
                    c4 = errM[3, i4]
                    
                    ## 4th piece
                    for i5 in range(mat_iter):
                        p5 = idxM[4, i5]
                        l5 = lenM[4, i5]
                        c5 = errM[4, i5]

                        ## check whether range is 300 and index is valid
                        true_range = l1+l2+l3+l4+l5
                        
                        if (true_range==300) and (p1<p2<p3<p4<p5):
                            sum_cost = c1+c2+c3+c4+c5

                            ## check a minimum cost
                            if  count==0 or pre_cost > sum_cost:
                                #print("indd!!!!")
                                min_cost = sum_cost
                                min_l = l1, l2, l3, l4, l5
                                min_bp = p1, p2, p3, p4, p5
                                count = count+1

                            pre_cost = sum_cost

    print("min_cost: ", min_cost, "at {0} with {1}".format(min_bp, min_l))
    return min_cost, min_l, min_bp

def curve6_Fitting(idxM, lenM, errM):
    groups, mat_iter = np.shape(lenM)
    pre_cost = np.nan # an argument to check successive cost
    min_cost = float('nan') # minimum cost
    min_l = [np.nan]*groups # minimum length
    min_bp = [np.nan]*groups # minimum break points

    count = 0

    ## 1st piece
    for i1 in range(mat_iter):
        # print('iter ', i1)
        p1 = idxM[0, i1]
        l1 = lenM[0, i1]
        c1 = errM[0, i1]
        
        ## 2nd piece
        for i2 in range(mat_iter):
            p2 = idxM[1, i2]
            l2 = lenM[1, i2]
            c2 = errM[1, i2]

            ## 3rd piece
            for i3 in range(mat_iter):
                p3 = idxM[2, i3]
                l3 = lenM[2, i3]
                c3 = errM[2, i3]
                
                ## 4th piece
                for i4 in range(mat_iter):
                    p4 = idxM[3, i4]
                    l4 = lenM[3, i4]
                    c4 = errM[3, i4]
                    
                    ## 5th piece
                    for i5 in range(mat_iter):
                        p5 = idxM[4, i5]
                        l5 = lenM[4, i5]
                        c5 = errM[4, i5]
                        
                        ## 6th piece
                        for i6 in range(mat_iter):
                            p6 = idxM[5, i6]
                            l6 = lenM[5, i6]
                            c6 = errM[5, i6]

                            ## check whether range is 300 and index is valid
                            true_range = l1+l2+l3+l4+l5+l6
                            
                            if (true_range==300) and (p1<p2<p3<p4<p5<p6):
                                sum_cost = c1+c2+c3+c4+c5+c6

                                ## check a minimum cost
                                if  count==0 or pre_cost > sum_cost:
                                    #print("indd!!!!")
                                    min_cost = sum_cost
                                    min_l = l1, l2, l3, l4, l5, l6
                                    min_bp = p1, p2, p3, p4, p5, p6
                                    count = count+1

                                pre_cost = sum_cost

    print("min_cost: ", min_cost, "at {0} with {1}".format(min_bp, min_l))
    return min_cost, min_l, min_bp

def curve7_Fitting(idxM, lenM, errM):
    groups, mat_iter = np.shape(lenM)
    pre_cost = np.nan # an argument to check successive cost
    min_cost = float('nan') # minimum cost
    min_l = [np.nan]*groups # minimum length
    min_bp = [np.nan]*groups # minimum break points

    count = 0

    ## 1st piece
    for i1 in range(mat_iter):
        # print('iter ', i1)
        p1 = idxM[0, i1]
        l1 = lenM[0, i1]
        c1 = errM[0, i1]
        
        ## 2nd piece
        for i2 in range(mat_iter):
            p2 = idxM[1, i2]
            l2 = lenM[1, i2]
            c2 = errM[1, i2]

            ## 3rd piece
            for i3 in range(mat_iter):
                p3 = idxM[2, i3]
                l3 = lenM[2, i3]
                c3 = errM[2, i3]
                
                ## 4th piece
                for i4 in range(mat_iter):
                    p4 = idxM[3, i4]
                    l4 = lenM[3, i4]
                    c4 = errM[3, i4]
                    
                    ## 5th piece
                    for i5 in range(mat_iter):
                        p5 = idxM[4, i5]
                        l5 = lenM[4, i5]
                        c5 = errM[4, i5]
                        
                        ## 6th piece
                        for i6 in range(mat_iter):
                            p6 = idxM[5, i6]
                            l6 = lenM[5, i6]
                            c6 = errM[5, i6]
                            
                            ## 7th piece
                            for i7 in range(mat_iter):
                                p7 = idxM[6, i7]
                                l7 = lenM[6, i7]
                                c7 = errM[6, i7]

                                ## check whether range is 300 and index is valid
                                true_range = l1+l2+l3+l4+l5+l6+l7

                                if (true_range==300) and (p1<p2<p3<p4<p5<p6<p7):
                                    sum_cost = c1+c2+c3+c4+c5+c6+c7

                                    ## check a minimum cost
                                    if  count==0 or pre_cost > sum_cost:
                                        #print("indd!!!!")
                                        min_cost = sum_cost
                                        min_l = l1, l2, l3, l4, l5, l6, l7
                                        min_bp = p1, p2, p3, p4, p5, p6, p7
                                        count = count+1

                                    pre_cost = sum_cost

    print("min_cost: ", min_cost, "at {0} with {1}".format(min_bp, min_l))
    return min_cost, min_l, min_bp

def curve8_Fitting(idxM, lenM, errM):
    groups, mat_iter = np.shape(lenM)
    pre_cost = np.nan # an argument to check successive cost
    min_cost = float('nan') # minimum cost
    min_l = [np.nan]*groups # minimum length
    min_bp = [np.nan]*groups # minimum break points

    count = 0

    ## 1st piece
    for i1 in range(mat_iter):
        # print('iter ', i1)
        p1 = idxM[0, i1]
        l1 = lenM[0, i1]
        c1 = errM[0, i1]
        
        ## 2nd piece
        for i2 in range(mat_iter):
            p2 = idxM[1, i2]
            l2 = lenM[1, i2]
            c2 = errM[1, i2]

            ## 3rd piece
            for i3 in range(mat_iter):
                p3 = idxM[2, i3]
                l3 = lenM[2, i3]
                c3 = errM[2, i3]
                
                ## 4th piece
                for i4 in range(mat_iter):
                    p4 = idxM[3, i4]
                    l4 = lenM[3, i4]
                    c4 = errM[3, i4]
                    
                    ## 5th piece
                    for i5 in range(mat_iter):
                        p5 = idxM[4, i5]
                        l5 = lenM[4, i5]
                        c5 = errM[4, i5]
                        
                        ## 6th piece
                        for i6 in range(mat_iter):
                            p6 = idxM[5, i6]
                            l6 = lenM[5, i6]
                            c6 = errM[5, i6]
                            
                            ## 7th piece
                            for i7 in range(mat_iter):
                                p7 = idxM[6, i7]
                                l7 = lenM[6, i7]
                                c7 = errM[6, i7]
                                
                                ## 8th piece
                                for i8 in range(mat_iter):
                                    p8 = idxM[7, i8]
                                    l8 = lenM[7, i8]
                                    c8 = errM[7, i8]

                                    ## check whether range is 300 and index is valid
                                    true_range = l1+l2+l3+l4+l5+l6+l7+l8
                                    
                                    if (true_range==300) and (p1<p2<p3<p4<p5<p6<p7<p8):
                                        sum_cost = c1+c2+c3+c4+c5+c6+c7+c8

                                        ## check a minimum cost
                                        if  count==0 or pre_cost > sum_cost:
                                            #print("indd!!!!")
                                            min_cost = sum_cost
                                            min_l = l1, l2, l3, l4, l5, l6, l7, l8
                                            min_bp = p1, p2, p3, p4, p5, p6, p7, p8
                                            count = count+1

                                        pre_cost = sum_cost

    print("min_cost: ", min_cost, "at {0} with {1}".format(min_bp, min_l))
    return min_cost, min_l, min_bp

def curve9_Fitting(idxM, lenM, errM):
    groups, mat_iter = np.shape(lenM)
    pre_cost = np.nan # an argument to check successive cost
    min_cost = float('nan') # minimum cost
    min_l = [np.nan]*groups # minimum length
    min_bp = [np.nan]*groups # minimum break points

    count = 0

    ## 1st piece
    for i1 in range(mat_iter):
        # print('iter ', i1)
        p1 = idxM[0, i1]
        l1 = lenM[0, i1]
        c1 = errM[0, i1]
        
        ## 2nd piece
        for i2 in range(mat_iter):
            p2 = idxM[1, i2]
            l2 = lenM[1, i2]
            c2 = errM[1, i2]

            ## 3rd piece
            for i3 in range(mat_iter):
                p3 = idxM[2, i3]
                l3 = lenM[2, i3]
                c3 = errM[2, i3]
                
                ## 4th piece
                for i4 in range(mat_iter):
                    p4 = idxM[3, i4]
                    l4 = lenM[3, i4]
                    c4 = errM[3, i4]
                    
                    ## 5th piece
                    for i5 in range(mat_iter):
                        p5 = idxM[4, i5]
                        l5 = lenM[4, i5]
                        c5 = errM[4, i5]
                        
                        ## 6th piece
                        for i6 in range(mat_iter):
                            p6 = idxM[5, i6]
                            l6 = lenM[5, i6]
                            c6 = errM[5, i6]
                            
                            ## 7th piece
                            for i7 in range(mat_iter):
                                p7 = idxM[6, i7]
                                l7 = lenM[6, i7]
                                c7 = errM[6, i7]
                                
                                ## 8th piece
                                for i8 in range(mat_iter):
                                    p8 = idxM[7, i8]
                                    l8 = lenM[7, i8]
                                    c8 = errM[7, i8]
                                    
                                    ## 9th piece
                                    for i9 in range(mat_iter):
                                        p9 = idxM[8, i9]
                                        l9 = lenM[8, i9]
                                        c9 = errM[8, i9]

                                        ## check whether range is 300 and index is valid
                                        true_range = l1+l2+l3+l4+l5+l6+l7+l8+l9

                                        if (true_range==300) and (p1<p2<p3<p4<p5<p6<p7<p8<p9):
                                            sum_cost = c1+c2+c3+c4+c5+c6+c7+c8+c9

                                            ## check a minimum cost
                                            if  count==0 or pre_cost > sum_cost:
                                                #print("indd!!!!")
                                                min_cost = sum_cost
                                                min_l = l1, l2, l3, l4, l5, l6, l7, l8, l9
                                                min_bp = p1, p2, p3, p4, p5, p6, p7, p8, p9
                                                count = count+1

                                            pre_cost = sum_cost

    print("min_cost: ", min_cost, "at {0} with {1}".format(min_bp, min_l))
    return min_cost, min_l, min_bp

def curve10_Fitting(idxM, lenM, errM):
    groups, mat_iter = np.shape(lenM)
    pre_cost = np.nan # an argument to check successive cost
    min_cost = float('nan') # minimum cost
    min_l = [np.nan]*groups # minimum length
    min_bp = [np.nan]*groups # minimum break points

    count = 0

    ## 1st piece
    for i1 in range(mat_iter):
        # print('iter ', i1)
        p1 = idxM[0, i1]
        l1 = lenM[0, i1]
        c1 = errM[0, i1]
        
        ## 2nd piece
        for i2 in range(mat_iter):
            p2 = idxM[1, i2]
            l2 = lenM[1, i2]
            c2 = errM[1, i2]

            ## 3rd piece
            for i3 in range(mat_iter):
                p3 = idxM[2, i3]
                l3 = lenM[2, i3]
                c3 = errM[2, i3]
                
                ## 4th piece
                for i4 in range(mat_iter):
                    p4 = idxM[3, i4]
                    l4 = lenM[3, i4]
                    c4 = errM[3, i4]
                    
                    ## 5th piece
                    for i5 in range(mat_iter):
                        p5 = idxM[4, i5]
                        l5 = lenM[4, i5]
                        c5 = errM[4, i5]
                        
                        ## 6th piece
                        for i6 in range(mat_iter):
                            p6 = idxM[5, i6]
                            l6 = lenM[5, i6]
                            c6 = errM[5, i6]
                            
                            ## 7th piece
                            for i7 in range(mat_iter):
                                p7 = idxM[6, i7]
                                l7 = lenM[6, i7]
                                c7 = errM[6, i7]
                                
                                ## 8th piece
                                for i8 in range(mat_iter):
                                    p8 = idxM[7, i8]
                                    l8 = lenM[7, i8]
                                    c8 = errM[7, i8]
                                    
                                    ## 9th piece
                                    for i9 in range(mat_iter):
                                        p9 = idxM[8, i9]
                                        l9 = lenM[8, i9]
                                        c9 = errM[8, i9]
                                        
                                        ## 9th piece
                                        for i10 in range(mat_iter):
                                            p10 = idxM[9, i10]
                                            l10 = lenM[9, i10]
                                            c10 = errM[9, i10]

                                            ## check whether range is 300 and index is valid
                                            true_range = l1+l2+l3+l4+l5+l6+l7+l8+l9+l10

                                            if (true_range==300) and (p1<p2<p3<p4<p5<p6<p7<p8<p9<p10):
                                                sum_cost = c1+c2+c3+c4+c5+c6+c7+c8+c9+c10

                                                ## check a minimum cost
                                                if  count==0 or pre_cost > sum_cost:
                                                    #print("indd!!!!")
                                                    min_cost = sum_cost
                                                    min_l = l1, l2, l3, l4, l5, l6, l7, l8, l9, l10
                                                    min_bp = p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
                                                    count = count+1

                                                pre_cost = sum_cost

    print("min_cost: ", min_cost, "at {0} with {1}".format(min_bp, min_l))
    return min_cost, min_l, min_bp

def multCurve_Fitting(y, lf, s=[1, 1], w=50, p=3): # y=data, lf=least square functions, s=seed, w=window, p=piece
    idx_matrix, err_matrix, len_matrix = curve_Matrix(y, lf, seed=s, window=w, piece=p)
    
    if p == 2:
        cost, min_length, min_indice = curve2_Fitting(idx_matrix, len_matrix, err_matrix)
        
    elif p == 3:
        cost, min_length, min_indice = curve3_Fitting(idx_matrix, len_matrix, err_matrix)
        
    elif p == 4:
        cost, min_length, min_indice = curve4_Fitting(idx_matrix, len_matrix, err_matrix)
        
    elif p == 5:
        cost, min_length, min_indice = curve5_Fitting(idx_matrix, len_matrix, err_matrix)
        
    elif p == 6:
        cost, min_length, min_indice = curve6_Fitting(idx_matrix, len_matrix, err_matrix)
        
    elif p == 7:
        cost, min_length, min_indice = curve7_Fitting(idx_matrix, len_matrix, err_matrix)
        
    elif p == 8:
        cost, min_length, min_indice = curve8_Fitting(idx_matrix, len_matrix, err_matrix)
        
    elif p == 9:
        cost, min_length, min_indice = curve9_Fitting(idx_matrix, len_matrix, err_matrix)
        
    elif p == 10:
        cost, min_length, min_indice = curve10_Fitting(idx_matrix, len_matrix, err_matrix)
        
    return cost, min_length, min_indice
    