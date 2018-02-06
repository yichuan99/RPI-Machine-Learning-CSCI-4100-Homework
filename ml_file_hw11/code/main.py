import numpy as np
import matplotlib.pyplot as plt
import classify as cls
import image_features as features
from   digits_extract import extractDigits
import lengendreTransfrom as lt
import copy
import nn

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
    
def labelImageData(data, digit, other_digits, feature1, feature2):
    labeled_data = []
    for item in data[digit]: ##each item is an image
        x1 = feature1(item)
        x2 = feature2(item)
        labeled_data.append(([x1,x2],1))
    for digit in other_digits:
        for item in data[digit]: ##each item is an image
            x1 = feature1(item)
            x2 = feature2(item)
            labeled_data.append(([x1,x2],-1))
    print len(labeled_data), "data points"
    return labeled_data

def DataSelection():
    print "Data Selection=========================="
    print "Extracting Data..."
    ## Data preprocessing=====================================
    test = extractDigits("ZipDigits.train") 
    train = extractDigits("ZipDigits.test")
    ##combine train and test
    data={}
    for i in range(10):
        data[i]=train[i]+test[i]
        
    d1 = 1
    d2 = [2,3,4,5,6,7,8,9,0]    
    raw_labeled_data = labelImageData(data,d1,d2,
                                   features.horiAsymmetry,
                                   features.vertAsymmetry)
    norm_data = features.featureNormalize(raw_labeled_data)
    
    print "Selecting Data..."
    num_list = []
    train_data = []
    while len(train_data)<300:
        num = np.random.randint(len(norm_data))
        train_data.append(norm_data[num])
        norm_data.remove(norm_data[num])
    test_data = norm_data
    ##===========================================================
    
    print "Data selection Done"
    return train_data, test_data

def modelSelection(reg_range, order, trans_train, trans_test, do_print):
    ##Model Selection==================================================
    if do_print: print "MODEL SELECTION:"
    e_cv_list=[]
    e_test_list=[]
    total_range=reg_range
    step=(float(total_range)/200)
    reg_list=np.arange(step,total_range,step)
    e_cv_min=1.0
    reg_min = 0
    horizontal_axis=[]
    for reg in reg_list:
        if do_print: print "MODEL: reg = ", reg
        ##Learning=========================================================
        g = cls.linear_regression(trans_train, reg)   
        e_cv=cls.crossValidationRegression(trans_train, reg)
        e_test = 1-cls.accuracyCalc(g, trans_test)
        if e_cv<e_cv_min: 
            e_cv_min=e_cv
            reg_min=reg
        if do_print: print "E_CV:", e_cv, "E_test:", e_test
        ##Cross Validation=================================================
        e_cv_list.append(e_cv)
        e_test_list.append(e_test)
        horizontal_axis.append(0)
        if do_print: print "================================================"  
   
    #plt.plot(reg_list, e_cv_list, "b.")
    #plt.plot(reg_list, horizontal_axis, "r-")
    #plt.show()
    
    if True:
        ##lambda star============================
        g = cls.linear_regression(trans_train, reg_min) 
        lt.DrawLegendreContour(order, g, train_data, test_data)
        e_cv=cls.crossValidationRegression(trans_train, reg_min)
        e_test = 1-cls.accuracyCalc(g, trans_test)    
        
        print "lambda:", reg_min, "E_CV:", e_cv, "E_test:", e_test
        ##=============================================    
        
        
def P_1(train_data, test_data):
    print "KNN======================="
    
    k_list = range(1,50)
    knn_e_cv_list=[]
    min_e_cv=1
    min_k=1
    print "Selecting K..."
    
    for k in k_list:
        #if k%5==0:print k
        e_cv=nn.CrossValidationNN(k, train_data)
        knn_e_cv_list.append(e_cv)
        if min_e_cv>e_cv:
            min_e_cv=e_cv
            min_k=k
    print "best k and cv:", min_k, min_e_cv
    
    print "e_test:", nn.KNNAccuracyCalc(min_k, test_data, train_data)
    plt.plot(k_list, knn_e_cv_list, "b.")
    plt.show()    
    print "Drawing Contour..."
    nn.DrawNNContour(min_k, train_data, False, -1, 1, -1, 1, False)
    
def P_2(train_data, test_data):
    print "RBF========================="
    min_k = 0
    min_e_cv = 1
    e_cv_list = []
    k_list = range(1,60)
    print "Selecting K..."
    for k in k_list:
        #if k%10==0:print k
        e_cv = nn.CrossValidRBF(k, train_data)
        e_cv_list.append(e_cv)
        if e_cv < min_e_cv:
            min_e_cv = e_cv
            min_k = k
            #print "update k:", k, "e_cv:", e_cv
            
    centers, w_lin = nn.RBF_Learn(train_data, min_k)   
    print "best k and cv:", min_k, min_e_cv
    r=2/(np.sqrt(min_k))
    print "etest:", nn.RBFAccuracyCalc(w_lin, centers, r, test_data)
    plt.plot(k_list, e_cv_list, "b.")
    plt.show()
    nn.DrawRBFContour(centers, w_lin, train_data, -1, 1, -1, 1)    
    
def LinearModel(train_data, test_data):
    print "Linear Model=================================="
    trans_train = lt.L_transform(8, train_data)
    trans_test = lt.L_transform(8, test_data)
    ##===========================================================
    
    modelSelection(2, 8, trans_train, trans_test, False)

   
    

if __name__=="__main__":
    train_data, test_data = DataSelection()
    P_1(train_data, test_data)
    P_2(train_data, test_data)
    LinearModel(train_data, test_data)
    a = raw_input()


    

    

   
    