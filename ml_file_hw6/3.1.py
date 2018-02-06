import numpy as np
import matplotlib.pyplot as plt

class Point2D:
    x1 = 0
    x2 = 0
    def __init__(self, x1_, x2_):
        self.x1 = x1_
        self.x2 = x2_
        
    def InWasher(self, rad, thk):
        return (self.x1**2+self.x2**2)**0.5<=rad+thk and (self.x1**2+self.x2**2)**0.5>=rad
    
class HalfWashers():
    rad = 0
    thk = 0
    sep = 0
    num_dots = 0
    
    def __init__(self, rad_, thk_, sep_):
        self.rad = rad_
        self.thk = thk_
        self.sep = sep_        
    
    def GenerateDots(self,num_dots):
        dots = []
        while(1):
            x1 = np.random.uniform(-self.rad-self.thk,self.rad+self.thk)
            x2 = np.random.uniform(-self.rad-self.thk,self.rad+self.thk)
            pt = Point2D(x1,x2)
            if pt.InWasher(self.rad,self.thk):
                if x2<0:
                    x1+=self.rad+self.thk/2
                    x2-=self.sep
                dots.append((x1,x2))
                if(len(dots)>=num_dots):break
                
        classified_dots = []
        for dot in dots:
            if dot[1]>=0:classified_dots.append((dot,1))
            else:classified_dots.append((dot,-1))
        
        answer = []
        for (x1,x2),y in classified_dots:
            moved_dot = (x1+self.rad+self.thk,x2+self.rad+self.thk+self.sep),y
            answer.append(moved_dot)
        return answer
    
    def setPlotFrame(self):
        rad = self.rad
        thk = self.thk
        sep = self.sep
        plt.axis([0, 3*rad+2.5*thk, 0, 2*rad+2*thk+sep])
    
def Data(rad, thk, sep, num_dots):
    region = HalfWashers(rad, thk, sep)
    region.setPlotFrame()
    return region.GenerateDots(num_dots)

def draw_data(dots):
    ## Data Plotting ===============================================
    dots_up_x1 = [x1 for (x1,x2),y in dots if y==1]
    dots_up_x2 = [x2 for (x1,x2),y in dots if y==1]
    dots_down_x1 = [x1 for (x1,x2),y in dots if y==-1]
    dots_down_x2 = [x2 for (x1,x2),y in dots if y==-1]    
    plt.plot(dots_up_x1, dots_up_x2, "b.")
    plt.plot(dots_down_x1, dots_down_x2, "r.")
    ##==============================================================     

def accuracyCalc(g,data_points):
    counter = 0
    for (x1,x2),y in data_points:
        x = [1,x1,x2]
        #print g,x
        if np.inner(g,x)*y>0:counter+=1
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
        if iteration%(len(data_points)/10)==0:
            #print iteration
            rate = accuracyCalc([g_w0,g_w1,g_w2],data_points)
            if rate == 1.0: 
                print "Interation to converge:", iteration
                break
    
    print g_w0,g_w1,g_w2
    draw_line(g_w0,g_w1,g_w2)      
    
    return iteration

def draw_line(g_w0,g_w1,g_w2):
    ##g function line drawing
    g_a = float(g_w1)/-g_w2
    g_b = float(g_w0)/-g_w2
    
    ##dots for g_line drawing
    g_dot_left_x1 = -1000
    g_dot_left_x2 = g_a*-1000 + g_b
    g_dot_right_x1 = 1000
    g_dot_right_x2 = g_a*+1000 + g_b
    ##g line drawing
    plt.plot([g_dot_left_x1,g_dot_right_x1],[g_dot_left_x2,g_dot_right_x2], 'r')            

def linear_regression(data):
    X = [[1,x1,x2] for (x1,x2),y in data]
    Y = [y for (x1,x2),y in data]
    
    XT = np.transpose(X)
    XTX = np.dot(XT,X)
    XTX_inv = np.linalg.inv(XTX)
    XTX_invXT = np.dot(XTX_inv,XT)
    [lin_w0, lin_w1, lin_w2] = np.dot(XTX_invXT,Y)
    g_w0 = lin_w0
    g_w1 = lin_w1
    g_w2 = lin_w2
    
    print lin_w0,lin_w1,lin_w2
    draw_line(lin_w0, lin_w1, lin_w2)   
    
if __name__ == "__main__":
    
    data = Data(10,5,5,2000)
    iteration = Perceptron(data)
    #linear_regression(data)
    
    draw_data(data)
    ##++++++++
    plt.show()
    ##++++++++
    
    
        