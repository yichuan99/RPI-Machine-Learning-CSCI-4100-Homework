import numpy as np

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
    print "Extract data from: ", filename
    #print len(digits_data[1]),len(digits_data[5])  
    #print i
    return digits_data
    
