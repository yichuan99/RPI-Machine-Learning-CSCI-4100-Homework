import numpy as np
import matplotlib.pyplot as plt
import classify as cls
from heapq import nsmallest
import time
import P_3_1 as p31

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
        D.append([x1,x2])
    return D

##requires: list of data point: D[[x1,x2]]
##          number of clusters: n
##modifies: nothing
##throws: nothing
##effects: nothing
##returns: first generation of center locations: [[x1,x2]]
def GenerateCenters(data, n):
    ##randomly selected first center
    first_center_location=data[np.random.randint(0,len(data))]
    r=0
    centers=[(first_center_location,r)]
    
    ##pick other n-1 centers
    for i in range(0,n-1):
        next_center=[0,0]
        max_dist=0
        for point in data:
            dist_all_centers=[EuclideanDistance(point, center) for center in centers]
            dist_from_center_set = min(dist_all_centers)
            if dist_from_center_set>max_dist:
                next_center=point
                max_dist=dist_from_center_set                
        centers.append(next_center)
    return centers    

##requires: list of data point: D[[x1,x2]]
##          number of clusters: n
##modifies: nothing
##throws: nothing
##effects: nothing
##returns: a list of clustered data according to centers
def ClusterData(data, n):
    centers = GenerateCenters(data, n)
    iterations=3
    clustered_data=[[center,0,[]] for center in centers]
    for i in range(iterations):
        ##assign points to centers
        for point in data:
            dist_to_centers=[EuclideanDistance(point, center) for center in centers]
            clustered_data[np.argmin(dist_to_centers)][1].append(point)
        ##recaululate centers (average)
        centers=[]
        for [center,cluster] in clustered_data:
            x1_list = [x1 for [x1,x2] in cluster]
            x2_list = [x2 for [x1,x2] in cluster]
            center_x1=sum(x1_list)/len(x1_list)
            center_x2=sum(x2_list)/len(x2_list)
            centers.append([center_x1, center_x2])
    
    return clustered_data
##Problem 6.16
def P_6_16():
    data=GenerateData(0,1,0,1,10000)
    clustered_Data=ClusterData(data, 10)
    

if __name__=="__main__":
    ## Parameters====================================
    #P_6_1()
    P_6_4()
