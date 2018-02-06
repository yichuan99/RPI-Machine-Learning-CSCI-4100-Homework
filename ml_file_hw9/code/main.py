import numpy as np
import matplotlib.pyplot as plt
import classify as cls
import image_features as features
from   digits_extract import extractDigits
import lengendreTransfrom as lt
import copy

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

def featureNormalize(data):
    X1 = [x1 for [x1,x2],y in data]
    x1_max = max(X1)
    x1_min = min(X1)
    x1_scale = float(2)/(x1_max-x1_min)
    x1_shift = float(x1_max+x1_min)/2

    X2 = [x2 for [x1,x2],y in data]
    x2_max = max(X2)
    x2_min = min(X2)
    x2_scale = float(2)/(x2_max-x2_min)
    x2_shift = float(x2_max+x2_min)/2    
    
    normalized_data = []
    for [x1,x2],y in data:
        norm_x1 = (x1-x1_shift)*x1_scale
        norm_x2 = (x2-x2_shift)*x2_scale
        norm_data_point = ([norm_x1,norm_x2],y)
        normalized_data.append(norm_data_point)
        
    return normalized_data

def crossValidationOld(train_data, reg):
    #print "cross validation: reg =", reg
    counter = 0
    i=0
    for validation_point in train_data:
        if i%10==0:print i
        i+=1
        val_train_data = copy.deepcopy(train_data)
        val_train_data.remove(validation_point)
        g_ = cls.linear_regression(val_train_data, reg)
        if cls.accuracyCalc(g_, [validation_point])==1.0:
            counter+=1
    accuracy = float(counter)/len(train_data)
    return 1-accuracy

def crossValidation(data, reg):
    Z = [x for x,y in data]
    Y = [y for x,y in data]
    d = len(Z[0])
    #print "d:", d, Z[0]
    
    
    I=np.identity(d)
    ZT = np.transpose(Z)
    ZTZ = np.matmul(ZT,Z)
    #print np.linalg.cond(ZTZ)
    ZTZ_REG = ZTZ+I*reg 
    ZTZ_REG_inv = np.linalg.inv(ZTZ_REG)
    ZTZ_REG_inv_ZT = np.matmul(ZTZ_REG_inv,ZT)
    H = np.matmul(Z, ZTZ_REG_inv_ZT)
    Y_bar = np.matmul(H,Y)

    E_CV=0
    for n in range(len(data)):
        E_CV+=float(((Y_bar[n]-Y[n])/(1-H[n][n]))**2)/len(data)
    return E_CV

def DrawLegendreContour(order, g):
    ##Draw contour==============================
    print "Drawing Contour..."
    x1_list = np.arange(-1.2,1.2,0.01)
    x2_list = np.arange(-1.2,1.2,0.01)
    X1,X2 = np.meshgrid(x1_list,x2_list)    
    
    Z=[]
    transformed=[lt.LegendreTransform(order, x1, x2) for x1,x2 in zip(X1,X2)]
    for x in transformed:
        weighted = sum([coef*feature for coef,feature in zip(g,x)])
        Z.append(weighted)
    ##=============================================
    cls.draw_data(train_data)
    plt.contour(X1, X2, Z, [0])
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.show()
    
    cls.draw_data(test_data)
    plt.contour(X1, X2, Z, [0])
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.show()

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
        e_cv=crossValidation(trans_train, reg)
        e_cv=0
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
   
    plt.plot(reg_list, e_test_list, "b.")
    plt.plot(reg_list, horizontal_axis, "r-")
    plt.show()
    
    if False:
        ##lambda star============================
        g = cls.linear_regression(trans_train, reg_min) 
        DrawLegendreContour(order, g)  
        e_cv=crossValidation(trans_train, reg)
        e_test = 1-cls.accuracyCalc(g, trans_test)    
        
        if do_print: print "lambda:", reg_min, "E_CV:", e_cv, "E_test:", e_test
        ##=============================================    

def power(base, exponent):
    if exponent<0:
        return ((float(1)/base))**(-exponent)
    else: 
        return base**exponent

if __name__=="__main__":
    ## Parameters====================================
    d1 = 1
    d2 = [2,3,4,5,6,7,8,9,0]
    #d2 = [5]
    order = 10
    reg = 2.0
        
    print "Extracting Data..."
    ## Data preprocessing=====================================
    test = extractDigits("ZipDigits.train") 
    train = extractDigits("ZipDigits.test")
    ##combine train and test
    data={}
    for i in range(10):
        data[i]=train[i]+test[i]
    
    raw_labeled_data = labelImageData(data,d1,d2,
                                   features.horiAsymmetry,
                                   features.vertAsymmetry)
    norm_data = featureNormalize(raw_labeled_data)
    
    print "Selecting Data..."
    num_list = []
    train_data = []
    while len(train_data)<300:
        num = np.random.randint(len(norm_data))
        train_data.append(norm_data[num])
        norm_data.remove(norm_data[num])
    test_data = norm_data
    trans_train = Z_transform(order, train_data)
    trans_test = Z_transform(order, test_data)
    ##===========================================================
    
    ##Model Selection
    if False:
        power_list = np.arange(-6,8,1.0)
        range_list = [power(10,x) for x in power_list]
        #print range_list
        for l_range in range_list:
            print "lambda range:", l_range
            modelSelection(l_range, order, trans_train, trans_test, False)
   
    
    g = cls.linear_regression(trans_train, 1000000000) 
    DrawLegendreContour(order, g)     
    
    ##Q2
    if False:
        g = cls.linear_regression(trans_train, 0) 
        DrawLegendreContour(order, g)
    
    ##Q3
    if False:
        g = cls.linear_regression(trans_train, 2) 
        DrawLegendreContour(order, g)
      
   
    