import numpy as np
import matplotlib.pyplot as plt
from operator import add
import copy

class Point2D:
    x1 = 0
    x2 = 0
    def __init__(self, x1_, x2_):
        self.x1 = x1_
        self.x2 = x2_

def draw_data(dots):
    ## Data Plotting ===============================================
    dots_up_x1 = [x1 for [x1,x2],y in dots if y==1]
    dots_up_x2 = [x2 for [x1,x2],y in dots if y==1]
    dots_down_x1 = [x1 for [x1,x2],y in dots if y==-1]
    dots_down_x2 = [x2 for [x1,x2],y in dots if y==-1]    
    plt.plot(dots_up_x1, dots_up_x2, "b.")
    plt.plot(dots_down_x1, dots_down_x2, "rx")
    ##==============================================================     

def accuracyCalc(g,data_points):
    counter = 0
    for x,y in data_points:
        x = [1]+x
        #print g,x
        if np.inner(g,x)*y>0:counter+=1
    rate = float(counter)/len(data_points)
    #print "accuracy:","%.4f"%rate
    return rate

def accuracyCalcPrint(g,data_points):
    print "accuracy result"
    counter = 0
    for x,y in data_points:
        x = [1]+x
        #print g,x
        if np.inner(g,x)*y>0:
            counter+=1
        print g,x,y,np.inner(g,x)*y
    rate = float(counter)/len(data_points)
    #print "accuracy:","%.4f"%rate
    return rate

def Perceptron(data_points):
    #perceptron iteration
    g_w0 = 0
    g_w1 = 0
    g_w2 = 0
    
    iteration = 0
    while(True):
        iteration+=1
        for [x1,x2],y in data_points:
            x0 = 1
            if np.inner([g_w0, g_w1, g_w2],[x0,x1,x2]) * y <=0:
                #print x1, x2
                g_w0 += x0*y
                g_w1 += x1*y
                g_w2 += x2*y   
                break
        if iteration%(len(data_points)/len(data_points))==0:
            #print iteration
            rate = accuracyCalc([g_w0,g_w1,g_w2],data_points)
            if rate >= 0.999: 
                print "Interation to converge:", iteration,rate
                break
        if iteration>5000:
            print "max iteration"
            break
    
    print g_w0,g_w1,g_w2
    draw_line(g_w0,g_w1,g_w2)      
    
    return iteration

def Pocket(data_points,
           weights,
           max_iteration, 
           accuracy_threshold):
   
    
    best_accuracy = 0
    w_best = weights
    
    print "pocket algorithm begins"
    iteration = 0
    while(True):
        #print iteration
        i = 0
        for x,y in data_points:
            x = [1]+x
            wTx = np.inner(weights,x)
            if wTx * y <=0:
                xy = np.multiply(x,y)
                weights = map(add, weights, xy)
                #print "mismatch:","iteration:",iteration,"i:",i,"x,y:",x,y,"w:",[w0,w1,w2]
                break
            i+=1
            #print iteration
        rate = accuracyCalc(weights,data_points)
        if rate > best_accuracy: 
            best_accuracy=rate
            w_best=weights
            #print "update:", iteration,rate
        if iteration > max_iteration:
            print "max iteration reached",best_accuracy
            break
        elif best_accuracy >= accuracy_threshold:
            print "accuracy achieved",best_accuracy
            break
        iteration+=1
        if iteration%500==0:print "iteration:", iteration
    print w_best
    #print g_w0,g_w1,g_w2
    if len(w_best)==3: ##2D perceptron only
        draw_line(w_best[0],w_best[1],w_best[2], "g") ##2D perceptron only
    
    return w_best

def draw_line(g_w0,g_w1,g_w2,color):
    ##g function line drawing
    g_a = float(g_w1)/-g_w2
    g_b = float(g_w0)/-g_w2
    print "draw line:",[g_w0, g_w1, g_w2]
    print "g_a:", g_a, "g_b", g_b
    ##dots for g_line drawing
    g_dot_left_x1 = -1000
    g_dot_left_x2 = g_a*-1000 + g_b
    g_dot_right_x1 = 1000
    g_dot_right_x2 = g_a*+1000 + g_b
    ##g line drawing
    plt.plot([g_dot_left_x1,g_dot_right_x1],[g_dot_left_x2,g_dot_right_x2], color)            

def linear_regression(data):
    X = [[1]+x for x,y in data]
    Y = [y for x,y in data]
    
    XT = np.transpose(X)
    XTX = np.dot(XT,X)
    XTX_inv = np.linalg.inv(XTX)
    XTX_invXT = np.dot(XTX_inv,XT)
    w_lin = np.dot(XTX_invXT,Y)
    g_w0 = w_lin[0]
    g_w1 = w_lin[1]
    g_w2 = w_lin[2]
    
    ##draw_line: 2D perceptron only
    #draw_line(g_w0, g_w1, g_w2, "r")   
    print "accuracy:", accuracyCalc(w_lin,data)
    print w_lin
    return w_lin

    
    
        