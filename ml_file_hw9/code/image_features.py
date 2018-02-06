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