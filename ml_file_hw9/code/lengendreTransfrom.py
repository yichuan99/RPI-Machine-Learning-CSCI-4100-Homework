import numpy as np

def LegendrePolynomial(order,x):
    if order==0:
        return 1
    elif order==1:
        return x
    elif order==2:
        return (1.0/2)*(-1  + 3*x**2)
    elif order==3:
        return (1.0/2)*(-3*x+ 5*x**3)
    elif order==4:
        return (1.0/8)*( 3   -30*x**2 + 35*x**4)
    elif order==5:
        return (1.0/8)*(15*x -70*x**2 + 63*x**5)
    elif order==6:
        return (1.0/16)*(-5 + 105*x**2 - 315*x**4 + 231*x**6)
    elif order==7:
        return (1.0/16)*(-35*x+315*x**3-693*x**5+429*x**7)
    elif order==8:
        return (1.0/128)*(35-1260*x**2+6930*x**4-12012*x**6+6435*x**8)
    elif order==9:
        return (1.0/128)*(315*x-4620*x**3+18018*x**5-25740*x**7+12155*x**9)
    elif order==10:
        return (1.0/256)*(-63+3465*x**2-30030*x**4+90090*x**6-109395*x**8+46189*x**10)   

def TwoFeaturesOrderCoefficientGeneration(order):
    l = np.arange(0,order+1,1)
    order_list = [[order-i,i] for i in l]
    return order_list

def LegendreTransform(order,x1,x2):
    L = LegendrePolynomial
    OrderGen = TwoFeaturesOrderCoefficientGeneration
    i=0
    full_order_list=[]
    while i<=order:
        order_list=OrderGen(i)
        full_order_list+=order_list
        i+=1
    #print len(full_order_list)
    #print full_order_list
    
    transformed_feature_values=[]
    for [j,k] in full_order_list:
        product = L(j,x1)*L(k,x2)
        transformed_feature_values.append(product)
    return transformed_feature_values
    