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

    return total/256
    
    
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
    return total/256

def avgIntensity(image):
    return sum(image)