import numpy as np
import matplotlib.pyplot as plt

def Learn(n):
    g_bar = [0,0]
    g_all = []
    EE_out = 0
    for i in range(n):
        x1 = np.random.uniform(-1,1)
        y1 = x1**2
        x2 = np.random.uniform(-1,1)
        y2 = x2**2
        a=x1+x2
        b=-x1*x2
        g_bar[0] += a/n
        g_bar[1] += b/n
        g_all.append((a,b))
        E_out = 0
        for i in range(n):
            x = np.random.uniform(-1,1)
            g_x = a*x+b
            sqr_err = (g_x-x**2)**2
            E_out += sqr_err/n
        EE_out+=E_out/n
        
    var = 0
    for i in range(len(g_all)):
        a = g_all[i][0]
        b = g_all[i][1]
        var_d=0
        for j in range(n):
            x = np.random.uniform(-1,1)
            g_x = a*x+b
            g_bar_x = g_bar[0]*x+g_bar[1]
            sqr_err = (g_x-g_bar_x)**2
            var_d += sqr_err/(len(g_all))
        var+=var_d/n
    
    bias = 0
    for i in range(n):
        x = np.random.uniform(-1,1)
        g_bar_x = g_bar[0]*x+g_bar[1]
        bias += (g_bar_x-x**2)**2/n
        
    print g_bar,EE_out,bias+var,var,bias
    
    x_input = np.arange(-1,1,0.001)
    g_bar_result=[]
    fx_plot=[]
    for i in range(len(x_input)):
        x = x_input[i]
        g_bar_result.append(g_bar[0]*x+g_bar[1])
        fx_plot.append(x**2)
    
    plt.plot(x_input,g_bar_result,"r-")
    plt.plot(x_input,fx_plot,"b-")
    plt.show()
    
def Plot_monotonic():
    plt.plot([-1,1],[1,-1],"r-")
    plt.text(0.3, 0.3, r'$h(x)=+1$', fontsize=12)
    plt.text(-0.3, -0.3, r'$h(x)=-1$', fontsize=12)
    plt.axis([-0.5, 0.5, -0.5, 0.5])
    plt.show()

if __name__ == "__main__":
    #Learn(5000)
    Plot_monotonic()

    
    
    