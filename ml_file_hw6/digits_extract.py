import numpy as np
import matplotlib.pyplot as plt

def extractDigits(filename):
    f = open(filename)
    i=0
    digits_data={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
    while(True):
        line = f.readline() 
        if line=="":break
        i+=1
        image_str = str.split(line)
        digit = int(float(image_str[0]))
        image = [float(pixel) for pixel in image_str[1:]]
        digits_data[digit].append(image)
        #print digit
    f.close()
    print "data done"
    #print len(digits_data[1]),len(digits_data[5])  
    #print i
    return digits_data
    
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
    
def hori_asymmetry(image):
    image_matrix=[]
    k=0
    for i in range(16):
        row = []
        for j in range(16):
            row.append(image[k])
            k+=1
        image_matrix.append(row)
        
    total=0
    for i in range(16):
        for j in range(8):
            total+=abs(image_matrix[i][j]-image_matrix[i][15-j])

    return total
    
    
def vert_asymmetry(image):
    image_matrix=[]
    k=0
    for i in range(16):
        row = []
        for j in range(16):
            row.append(image[k])
            k+=1
        image_matrix.append(row)
        
    total=0
          
    for i in range(8):
        for j in range(16):
            total+=abs(image_matrix[i][j]-image_matrix[15-i][j])
    return total    

def intensity(image):
    return sum(image)
    
def featureSep(digits_data, d1, d2):    
    intensity_list = []
    hori_asymmetry_list = []
    vert_asymmetry_list = []
    
    for image in digits_data[d1]:
        intensity_list.append(vert_asymmetry(image))
        hori_asymmetry_list.append(hori_asymmetry(image))
        vert_asymmetry_list.append(vert_asymmetry(image))
    plt.plot(vert_asymmetry_list, hori_asymmetry_list, "bo")
    
    intensity_list = []
    hori_asymmetry_list = []
    vert_asymmetry_list = []
    for i in [2,3,4,5,6,7,8,9,0]:
        for image in digits_data[i]:
            intensity_list.append(vert_asymmetry(image))
            hori_asymmetry_list.append(hori_asymmetry(image))
            vert_asymmetry_list.append(vert_asymmetry(image))
    plt.plot(vert_asymmetry_list, hori_asymmetry_list, "rx")
    plt.show()

if __name__=="__main__":
    digits_data = extractDigits("ZipDigits.test") 
    featureSep(digits_data,1,5)
    #drawImage(digits_data[1][0])
    #drawImage(digits_data[5][0])
