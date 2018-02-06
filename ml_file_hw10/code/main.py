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


def P_6_1():
    d1 = [1,0],-1
    d2 = [0,1],-1
    d3 = [0,-1],-1
    d4 = [-1,0],-1
    d5 = [0,2],1
    d6 = [0,-2],1
    d7 = [-2,0],1
    D = [d1,d2,d3,d4,d5,d6,d7]    
    

if __name__=="__main__":
    ## Parameters====================================
    P_6_1

    

    

   
    