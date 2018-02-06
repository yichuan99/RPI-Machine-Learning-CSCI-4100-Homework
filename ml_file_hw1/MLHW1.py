import numpy as np
import matplotlib.pyplot as plt

def data_gen(data_size, plot_size):
    #data generation
    data_points = []
    for i in range(data_size):
        x1 = np.random.uniform(0,float(plot_size))
        x2 = np.random.uniform(0,float(plot_size))
        data_points.append([x1,x2,0])   
    return data_points
    
def perceptron(plot_size, data_points, draw_g):
    
    #data_points = [[0.25,0.25,0],[0.25,0.75,0],[0.75,0.25,0],[0.75,0.75,0]]
    #target line generation
    dot1x1 = np.random.uniform(0,float(plot_size))
    dot1x2 = np.random.uniform(0,float(plot_size))
    dot2x1 = np.random.uniform(0,float(plot_size))
    dot2x2 = np.random.uniform(0,float(plot_size))
    
    ##x2 = ax1 + b
    ##b + ax1 - x2 = 0 --> w0 = b, w1 = a, w2 = -1
    
    #a = (dot2x2 - dot1x2)/(dot2x1 - dot1x1)
    #b = dot1x2 - a*dot1x1
    a = 2.5
    b = -0.33
    
    f_w0 = b
    f_w1 = a
    f_w2 = -1
    
    f_wT = np.matrix([f_w0, f_w1, f_w2])
    #dots for target line drawing
    dot_right_x2 = a*plot_size + b
    #target line drawing
    #plt.plot([0,plot_size],[b, dot_right_x2], 'g')
    
    #two-color plotting
    for i in range(len(data_points)):
        x1 = data_points[i][0]
        x2 = data_points[i][1]
        data_input = [1,x1,x2]
        if np.inner(f_wT,data_input)>0: 
            #plt.plot([x1],[x2], 'r.')
            data_points[i][2] = 1
        else: 
            #plt.plot([x1], [x2], 'b.')
            data_points[i][2] = -1
    #print data_points
    #perceptron iteration
    g_w0 = 0
    g_w1 = 0
    g_w2 = 0
    
    iteration = 0
    while(True):
        iteration+=1
        for j in range(len(data_points)):
            x0 = 1
            x1 = data_points[j][0]
            x2 = data_points[j][1]
            y = data_points[j][2]
            if np.inner([g_w0, g_w1, g_w2],[x0,x1,x2]) * y <=0:
                #print x1, x2
                g_w0 += x0*y
                g_w1 += x1*y
                g_w2 += x2*y   
                break
        if iteration%(len(data_points)/len(data_points))==0:
            #print iteration
            rate = accuracyCalc([g_w0,g_w1,g_w2],data_points)
            if rate == 1.0: 
                #print "Interation to converge:", iteration
                break
    g_w0 = 1
    g_w1 = 2
    g_w2 = 3
    ##g function line drawing
    g_a = float(g_w1)/-g_w2
    g_b = float(g_w0)/-g_w2
    
    ##dots for g_line drawing
    g_dot_right_x2 = g_a*plot_size + g_b
    ##g line drawing
    if draw_g==1: plt.plot([0,plot_size],[g_b, g_dot_right_x2], 'r')    
    
    
    #print f_w0,f_w1,f_w2
    #print g_w0,g_w1,g_w2
    plt.axis([0, 1, -1, -0.3])
    #plt.axis([0, plot_size, 0, plot_size])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.text(0.6, -0.5, r'$f(x)=-1$', fontsize=12)
    plt.text(0.2, -0.9, r'$f(x)=1$', fontsize=12)
    plt.show()
    print iteration
    return iteration
    
def accuracyCalc(g,data_points):
    counter = 0 
    for i in range(len(data_points)):
        x = [1,data_points[i][0],data_points[i][1]]
        if np.inner(g,x)*data_points[i][2]>0:counter+=1
    rate = float(counter)/len(data_points)
    #print "accuracy:","%.4f"%rate
    return rate

    
if __name__ == '__main__':
    data = data_gen(1,1)
    print perceptron(1,data,1)
    #print perceptron(1,data,0)
    
    '''
    count = 0
    for i in range(10):
        count+=perceptron(1,10)
    print "Average converge iteration ratio(10):", "%.3f"%(count/float(10*10))
    count = 0
    for i in range(10):
        count+=perceptron(1,30)
    print "Average converge iteration ratio(30):", "%.3f"%(count/float(10*30))    
    count = 0
    for i in range(10):
        count+=perceptron(1,100)
    print "Average converge iteration ratio(100):", "%.3f"%(count/float(10*100))    
    count = 0
    for i in range(10):
        count+=perceptron(1,300)
    print "Average converge iteration ratio(300):", "%.3f"%(count/float(10*300))
    count = 0
    for i in range(10):
        count+=perceptron(1,1000)
    print "Average converge iteration ratio(1000):", "%.3f"%(count/float(10*1000))   
    '''
    print "Done"