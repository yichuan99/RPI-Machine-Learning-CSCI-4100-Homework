def horiAsymmetry(image):
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

    return round(total/256, 5)
    
    
def vertAsymmetry(image):
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
    return round(total/256, 5)

def avgIntensity(image):
    return sum(image)/256

def FrobIntensity(image):
    for i in image:
        i=float(i+1)/2
    f_norm = np.sqrt(sum([x**2 for x in image]))
    return f_norm

##Only for 2D yet...
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