import numpy as np
import matplotlib.pyplot as plt
import classify as cls
import image_features as features
from   digits_extract import extractDigits

def drawImage(image):
    image_array = []
    k=0
    for i in range(16):
        row = []
        for j in range(16):
            gray_intensity=(image[k]+1)/2
            #print gray_intensity
            row.append([gray_intensity,gray_intensity,gray_intensity])
            k+=1
        image_array.append(row)
    image_array = np.array(image_array)
    print image_array
    plt.imshow(image_array)
    plt.show()
    
def labelImageData(data, digit1, digit2, feature1, feature2):
    labeled_data = []
    for item in data[digit1]: ##each item is an image
        x1 = feature1(item)
        x2 = feature2(item)
        labeled_data.append(([x1,x2],1))
    for item in data[digit2]: ##each item is an image
        x1 = feature1(item)
        x2 = feature2(item)
        labeled_data.append(([x1,x2],-1))
    return labeled_data

def regressionAndPocket(data, required_accuracy, max_iteration):
    w_lin = cls.linear_regression(data)
    g=cls.Pocket(data, w_lin, max_iteration, required_accuracy)    
    return g  

def thirdOrderTransfrom(raw_data): ## assume the raw data has two features x1,x2
    transformed_data=[]
    for x,y in raw_data:
        x1=x[0]
        x2=x[1]
        tran_x1 = x1
        tran_x2 = x2
        tran_x3 = x1**2
        tran_x4 = x2**2
        tran_x5 = x1*x2
        tran_x6 = x1**3
        tran_x7 = x2**3
        tran_x8 = (x1**2)*x2
        tran_x9 = x1*(x2**2)
        X = [tran_x1,
             tran_x2,
             tran_x3,
             tran_x4,
             tran_x5,
             tran_x6,
             tran_x7,
             tran_x8,
             tran_x9]
        
        data_point = (X,y)
        transformed_data.append(data_point)
    return transformed_data

if __name__=="__main__":
    ## Parameters====================================
    d1 = 1
    d2 = 5
    tran = True
    see_train = True

    max_iteration = 10000
    accuracy_threshold = 0.999
    ##================================================
    
    ## Data preprocessing=====================================
    training_data = extractDigits("ZipDigits.train") 
    test_data = extractDigits("ZipDigits.test")
    
    raw_train_labeled_data = labelImageData(training_data,d1,d2,
                                   features.horiAsymmetry,
                                   features.vertAsymmetry)
    raw_test_labeled_data = labelImageData(test_data, d1, d2, 
                                       features.horiAsymmetry,
                                       features.vertAsymmetry)
    
    if see_train: cls.draw_data(raw_train_labeled_data)
    else: cls.draw_data(raw_test_labeled_data)
    
    if not tran: 
        trans_train_data = raw_train_labeled_data
        trans_test_data = raw_test_labeled_data
    else: 
        trans_train_data = thirdOrderTransfrom(raw_train_labeled_data)
        trans_test_data = thirdOrderTransfrom(raw_test_labeled_data)
    
    ##===========================================================
    
    ##Learning=========================================================
    g = regressionAndPocket(trans_train_data, accuracy_threshold, max_iteration)
    ##===========================================================
    
    
    ##Draw contour==============================
    x1_list = np.arange(0,0.6,0.001)
    x2_list = np.arange(0,0.6,0.001)
    X1,X2 = np.meshgrid(x1_list,x2_list)    
    
    if not tran: Z = g[0] + g[1]*X1 + g[2]*X2 
    else: Z = g[0] + g[1]*X1 + g[2]*X2 + g[3]*(X1**2) + g[4]*(X2**2) + g[5]*X1*X2 + g[6]*(X1**3) + g[7]*(X2**3) + g[8]*X2*(X1**2) + g[8]*X1*(X2**2)
    ##=============================================

    print "test result:", cls.accuracyCalc(g,trans_test_data)
    plt.contour(X1, X2, Z, [0])
    plt.axis([0, 0.6, 0, 0.6])
    plt.show()
