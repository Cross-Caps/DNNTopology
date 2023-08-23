'''
# Code has in  https://github.com/shizuo-kaji/CubicalRipser_3dim
# pip install https://github.com/shizuo-kaji/CubicalRipser_3dim

import cripser
import ripser
import persim

def get_persist(grayscale,max_dim=1):
    pd = cripser.computePH(grayscale,maxdim=max_dim,location="birth")
    pds = [pd[pd[:,0] == i] for i in range(3)]
    persistence_diag = [p[:,1:3] for p in pds]
    return(persistence_diag)

# betti number values as per the book computational topology by Tamal Dey and Yusu wang
def betti(persistence_diag,threshold_val):
    betti_val=[]
    for i in persistence_diag:
        temp=0
        for j in i:
            if j[0] <= threshold_val and j[1] >= threshold_val:
                temp = temp+1
        betti_val.append(temp)
    return(betti_val)

    
# Reshaping data 
def set_data_shape(data,value=32):
    #shape_data = data.shape
    #print(shape_data)
    vol = np.dstack([i.reshape(value,value) for i in data])
    return(vol)
'''
