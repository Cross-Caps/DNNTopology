'''
# Code has in  https://github.com/shizuo-kaji/CubicalRipser_3dim
# pip install https://github.com/shizuo-kaji/CubicalRipser_3dim
import cripser
import ripser
import persim
import gudhi
import gudhi.wasserstein
def get_persist(grayscale,max_dim=1):
    pd = cripser.computePH(grayscale,maxdim=max_dim,location="birth")
    pds = [pd[pd[:,0] == i] for i in range(3)]
    persistance_val = [p[:,1:3] for p in pds]
    return(persistance_val)

#Code has described in Book Computational Topology by Tamal Dey and Yusu Wang

def betti(persistance_val,threshold_val):
    betti_num=[]
    for i in perstistance_val:
        temp = 0
        for j in i:
            if j[0]<=threshold_val and j[1]>threshold_val:
                temp = temp+1
        betti_num.append(temp)
    return(betti_num)

def set_data_shape(data,value=32):
    #shape_data = data.shape
    vol = np.dstack([i.reshape(value,value) for i in data])
    return(vol)
'''