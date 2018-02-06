import numpy as np
import matplotlib.pyplot as plt
import classify as cls
from heapq import nsmallest
import time
import P_3_1 as p31
import copy

def KNearestNeighbor(k, data, test_point, use_sort):
    #print len(data)
    dist_list = [(EuclideanDistance(test_point,x),y) for x,y in data]    
    
    if use_sort: ##sort and get k smallest
        dist_list = sorted(dist_list, key=lambda tup: tup[0])
        k_neighbors_val = [y for dist, y in dist_list[:k]]
    else:  ##select k smallest
        k_neighbors = nsmallest(k, dist_list, key=lambda tup: tup[0])
        k_neighbors_val = [y for dist, y in k_neighbors]
    
    #print sign(sum(k_neighbors_val))
    return sign(sum(k_neighbors_val))
    
def sign(n):
    if n>0:return 1
    else:return -1

def EuclideanDistance(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

def DrawNNContour(k, data, trans, xmin, xmax, ymin, ymax, use_sort):
    start_time = time.time()
    
    xlist = np.linspace(xmin, xmax, 250)
    ylist = np.linspace(ymin, ymax, 250)
    X, Y = np.meshgrid(xlist, ylist)
    Z=[]
    for x_row, y_row in zip(X,Y):
        z_row=zip(x_row,y_row)
        Z.append(z_row)    
    
    new_Z=[]
    if trans:
        trans_data = [(Transform(x),y) for x,y in data]
        for z_row in Z:
            row=[KNearestNeighbor(k, trans_data, Transform(test_point), use_sort) for test_point in z_row]
            new_Z.append(row)
    else:
        for z_row in Z:
            row=[KNearestNeighbor(k, data, test_point, use_sort) for test_point in z_row]
            new_Z.append(row)    
            
    #plt.figure()
    cp = plt.contourf(X, Y, new_Z)
    plt.colorbar(cp)
    plt.title('NN Plot')
    plt.xlabel('X1')
    plt.ylabel('X2')
    cls.draw_data(data)
    print("--- %s seconds ---" % (time.time() - start_time))   
    plt.show()    
    
## dirty solution...
def DrawRBFContour(centers, w_lin, data, xmin, xmax, ymin, ymax):
    start_time = time.time()
    
    xlist = np.linspace(xmin, xmax, 250)
    ylist = np.linspace(ymin, ymax, 250)
    X, Y = np.meshgrid(xlist, ylist)
    Z=[]
    for x_row, y_row in zip(X,Y):
        z_row=zip(x_row,y_row)
        Z.append(z_row)    
    
    new_Z=[]
    
    k = len(centers)
    r = 2/(np.sqrt(k))
    for z_row in Z:
        row=[RBF_Classify(w_lin, centers, test_point, r) for test_point in z_row]
        new_Z.append(row)

    #plt.figure()
    cp = plt.contourf(X, Y, new_Z)
    plt.colorbar(cp)
    plt.title('RBF Plot')
    plt.xlabel('X1')
    plt.ylabel('X2')
    cls.draw_data(data)
    print("--- %s seconds ---" % (time.time() - start_time))   
    plt.show()    
    


def Transform(x):
    [x1,x2]=x 
    return [np.sqrt(x1**2+x2**2), np.arctan(float(x2)/(x1+0.0000000001))]

def P_6_1():
    d1 = [1,0],-1
    d2 = [0,1],-1
    d3 = [0,-1],-1
    d4 = [-1,0],-1
    d5 = [0,2],1
    d6 = [0,-2],1
    d7 = [-2,0],1
    
    D = [d1,d2,d3,d4,d5,d6,d7]    
    
    DrawNNContour(1, D, False, -3, 3, -3, 3, True)
    DrawNNContour(3, D, False, -3, 3, -3, 3, True)
    DrawNNContour(1, D, True, -3, 3, -3, 3, True)
    DrawNNContour(3, D, True, -3, 3, -3, 3, True)

def P_6_4():
    data = p31.CircleData(10,5,5,2000)
    #DrawNNContour(1, data, False, -5, 50, -5, 40, False)
    DrawNNContour(3, data, False, -5, 50, -5, 40, False)

##requires: space square size and location (given in form of xy bounds)
##          number of data points
##modifies: nothing
##throws:   nothing
##effects:  nothing
##returns:  a list of data points within space square: D[[x1,x2]]
def GenerateData(xmin, xmax, ymin, ymax, n):
    D=[]
    for i in range(n):
        x1=np.random.uniform(xmin, xmax)
        x2=np.random.uniform(ymin, ymax)
        D.append(([x1,x2],1))
    return D

##requires: list of data point: D[[x1,x2]]
##          number of clusters: n
##modifies: nothing
##throws: nothing
##effects: nothing
##returns: first generation of center locations: [[x1,x2],y]
def GenerateCenters(data, n):
    ##randomly selected first center
    first_center_location=data[np.random.randint(0,len(data))][0]
    r=0
    centers=[first_center_location]
    
    ##pick other n-1 centers
    for i in range(0,n-1):
        next_center=[0,0]
        max_dist=0
        for x,y in data:
            dist_all_centers=[EuclideanDistance(x, center) for center in centers]
            dist_from_center_set = min(dist_all_centers)
            if dist_from_center_set>max_dist:
                next_center=x
                max_dist=dist_from_center_set                
        centers.append(next_center)
    return centers    

##requires: list of data point: D[[x1,x2],y]
##          number of clusters(centers): n
##modifies: nothing
##throws: nothing
##effects: nothing
##returns: a list of clustered data according to centers
##note: Loyd's algorithm
def ClusterData(data, n):
    
    centers = GenerateCenters(data, n)
    iterations=3
    clustered_data=[[center,0,[(center,1)]] for center in centers]
    for i in range(iterations):
        ##recaululate centers (average)
        centers=[]
        for [center,r,cluster] in clustered_data:
            x1_list = [x1 for [x1,x2],y in cluster]
            x2_list = [x2 for [x1,x2],y in cluster]
            center_x1=sum(x1_list)/len(x1_list)
            center_x2=sum(x2_list)/len(x2_list)
            centers.append([center_x1, center_x2])
            
            ##clearn the cluster for data point assignment
            cluster=[]

        ##assign points to centers
        ##iterate over all data points and assign each point to its closest center
        for data_point in data:
            x,y = data_point
            ##list of distance of this data point to all centers
            dist_to_centers=[EuclideanDistance(x, center) for center in centers]
            ##get the closest center and assign this point to that center
            closest_center_index = np.argmin(dist_to_centers)
            clustered_data[closest_center_index][2].append(data_point)
            ##if this data point has larger distance than center radius
            closest_center = clustered_data[closest_center_index][0]
            center_r = clustered_data[closest_center_index][1]
            if EuclideanDistance(x,closest_center)>r:
                clustered_data[closest_center_index][1]=EuclideanDistance(x,closest_center)            
    
    return centers, clustered_data

##helper: draw points with desired color
def draw_cluster(dots, color):
    ## Data Plotting ===============================================
    dots_x1 = [x1 for [x1,x2],y in dots]
    dots_x2 = [x2 for [x1,x2],y in dots]

    plt.plot(dots_x1, dots_x2, color)
    ##==============================================================  

##requires: number of nearest neighbors -> k
##          data to run validation -> data
##modifies: nothing
##throws: nothing
##effects: nothing
##returns: crossvalidation error
def CrossValidationNN(k, data):
    error_count=0
    for validation_point in data:
        val_train_data = copy.deepcopy(data)
        val_train_data.remove(validation_point)        
        x,y = validation_point
        if y!=KNearestNeighbor(k, val_train_data, x, True):
            error_count+=1
    return float(error_count)/len(data)

##requires: center -> [x1,x2]
##          data point -> [x1,x2]
##modifies: nothing
##throws: nothing
##effects: nothing
##returns: evaluation based on gaussian kernel
def RBF_Gaussian_Eval(center, data_point, r):
    e = np.e
    dist = EuclideanDistance(center, data_point)
    return e**(-0.5*(dist/r)**2)

def RBF_Classify(w_lin, centers, x, r):
    #print "x:", x
    #print "centers:", centers
    trans_x = [1]+[RBF_Gaussian_Eval(center,x,r) for center in centers]
    wtz = np.dot(w_lin, trans_x)
    return sign(wtz)

def RBFAccuracyCalc(w_lin, centers, r, data):
    count=0
    for x,y in data:
        if RBF_Classify(w_lin, centers, x, r)*y<0:
            count+=1
    return float(count)/len(data)

def KNNAccuracyCalc(k, data, train_data):
    count=0
    i = 0
    for x,y in data:
        #if i%200==0:print i
        if KNearestNeighbor(k, train_data, x, False)*y<0:
            count+=1
        i+=1
    return float(count)/len(data)


##requires: data -> [([x1,x2],y)]
##          number of centers -> k
##modifies: nothing
##throws: nothing
##effects: nothing
##returns: centers with appropriate weights 
def RBF_Learn(data, k):
    print "RBF learn with", k, "centers"
    ## generate k centers
    centers, clustered_Data=ClusterData(data, k)
    
    ## learn weights for centers (use linear regression)
    
    ## specify radius
    r = 2/(np.sqrt(k))
    ## transform data    
    trans_data=[]
    for x,y in data:
        trans_x = [1]+[RBF_Gaussian_Eval(center,x,r) for center in centers]
        trans_data.append((trans_x, y))
    w_lin = cls.linear_regression(trans_data, 0.0001)
    
    return centers, w_lin
       
def CrossValidRBF(k, data):
    #print "RBF Cross validation..."
    ## generate k centers
    centers, clustered_Data=ClusterData(data, k)
    
    ## learn weights for centers (use linear regression)
    
    ## specify radius
    r = 2/(np.sqrt(k))
    ## transform data    
    trans_data=[]
    for x,y in data:
        trans_x = [1]+[RBF_Gaussian_Eval(center,x, r) for center in centers]
        trans_data.append((trans_x, y))
    #print trans_data[0]
    e_cv = cls.crossValidationRegression(trans_data, 0.0001)
    return e_cv

##Problem 6.16, not done
def P_6_16():
    data=GenerateData(0,1,0,1,10000)
    num=1
    colors = ["r.", "b.", "g.", "k.", "m.", "y.","rx", "bx", "gx", "kx"]*num
    centers, clustered_Data=ClusterData(data, num*10)
    i = 0
    for [center, r, cluster] in clustered_Data:
        draw_cluster(cluster, colors[i])
        i+=1
        
    
    

if __name__=="__main__":
    P_6_16()
    plt.show()