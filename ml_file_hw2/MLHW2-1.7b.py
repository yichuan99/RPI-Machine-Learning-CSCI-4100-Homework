import numpy as np
import random
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os

def coin():
    count = 0
    for i in range(10):
        count+=random.randint(0,1)
    return float(count)/10

def flipCoins(times):
    c_1 = 0
    c_rand = 0
    c_min = 1.0
    rand_num = random.randint(0,times-1)
    for i in range(times):
        coin_ = coin()
        if i==rand_num:c_rand = coin_
        if i==0:c_1 = coin_
        if coin_<c_min: c_min = coin_
    return (c_1, c_rand, c_min)

def runMultipleExperiments():
    v_1    = []
    v_rand = []
    v_min  = []
    
    num_exp = 10000
    for i in range(num_exp):
        (c_1, c_rand, c_min) = flipCoins(1000)
        v_1.append(c_1)
        v_rand.append(c_rand)
        v_min.append(c_min)
    print "loop done"
    
    e_v_1    = []
    e_v_rand = []
    e_v_min  = []    
    for i in range(num_exp):
        e_v_1.append(abs(v_1[i]-0.5))
        e_v_rand.append(abs(v_rand[i]-0.5))
        e_v_min.append(abs(v_min[i]-0.5))           
    
    e=2.71828
    err = np.arange(0,1,0.01)
    hoeffding_bound = [2*np.power(e,-20*(x**2)) for x in err]    
    
    
    weights = np.ones_like(e_v_1)/float(len(e_v_1))
    plt.hist(e_v_1, normed=0, weights=weights, bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6])
    plt.plot(err,hoeffding_bound,"r-")
    plt.show()
    
    weights = np.ones_like(e_v_rand)/float(len(e_v_rand))
    plt.hist(e_v_rand, normed=0, weights=weights, bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6])
    plt.plot(err,hoeffding_bound,"r-")
    plt.show()
    
    weights = np.ones_like(e_v_min)/float(len(e_v_min))
    plt.hist(e_v_min, normed=0, weights=weights, bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6])
    plt.plot(err,hoeffding_bound,"r-")
    plt.show()      
    '''
    
    weights = np.ones_like(v_1)/float(len(v_1))
    plt.hist(v_1, normed=0, weights=weights,bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1])
    plt.show()
    weights = np.ones_like(v_rand)/float(len(v_rand))
    plt.hist(v_rand, normed=0,weights=weights, bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1])
    plt.show()
    weights = np.ones_like(v_min)/float(len(v_min))
    plt.hist(v_min, normed=0,weights=weights, bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1])
    plt.show()    
    '''

def coin6():
    count = 0
    for i in range(6):
        count+=random.randint(0,1)
    return float(count)/6

def flip2Coins6Times():
    
    data = []
    '''
    for i in range(400):
        data.append(0)
    for i in range(2100):
        data.append(1.0/6)
    for i in range(1344):
        data.append(2.0/6) 
    for i in range(228):
        data.append(0.5)   
    '''
    
    '''
    for i in range(10000):
        a = abs(coin6()-0.5)
        b = abs(coin6()-0.5)
        data.append(max(a,b))
     '''   
    e=2.71828
    err = np.arange(0,0.6,0.01)
    hoeffding_bound = [4*np.power(e,-12*(x**2)) for x in err] 
    
    weights = np.ones_like(data)/float(len(data))
    plt.hist(data,normed=0,weights=weights,bins=np.arange(0,0.6,0.01))
    plt.plot(err,hoeffding_bound,"r-")
    plt.show()
    
            
if __name__ == '__main__':
    runMultipleExperiments()
    #flip2Coins6Times()
    print "Done"