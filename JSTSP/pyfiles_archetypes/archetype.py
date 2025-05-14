import spams as sp
import numpy as np
import matplotlib.pyplot as plt
import persim
from ripser import Rips
from persim import PersistenceImager
from sklearn.decomposition import PCA
class return_archetype():
  def __init__(self):
    self.K = 200 # learns a dictionary with 64 elements
    self.robust = False # use robust archetypal analysis or not, default parameter(True)
    self.epsilon = 1e-3 # width in Huber loss, default parameter(1e-3)
    self.computeXtX = True # memorize the product XtX or not default parameter(True)
    self.stepsFISTA = 0 # 3 alternations by FISTA, default parameter(3)
    # a for loop in FISTA is used, we stop at 50 iterations
    # remember that we are not guarantee to descent in FISTA step if 50 is too small
    self.stepsAS = 10 # 7 alternations by activeSet, default parameter(50)
    self.randominit = True # random initilazation, default parameter(True)
    self.numThreads = -1
    self.returnAB= True
  def get_function(self,data):
    I = np.asfortranarray(data,dtype = np.float64)
    (Z,A,B) = sp.archetypalAnalysis(
    np.asfortranarray(I), returnAB= self.returnAB, p = self.K, \
    robust = self.robust, epsilon = self.epsilon,
    computeXtX = self.computeXtX,  stepsFISTA = self.stepsFISTA ,
    stepsAS = self.stepsAS, numThreads = self.numThreads)
    return(Z)
def matdata_get_archetype(AA,data,class_labels,classes_ = True,reducer =None):
  if type(reducer) !=type(None) :
     data = reducer.fit_transform(data)
  archetypes_of_data = []
  uni_que = np.unique(class_labels)
  if classes_ == True :
   for i in range(len(uni_que)):
     img = data[class_labels==uni_que[i]]
     print(img.shape)
     temp = AA.get_function(img.T).T
     archetypes_of_data.append(temp)
     print("Number of unique : ", np.unique(temp,axis =0).shape)
   return(np.asarray(archetypes_of_data))
  else:
    archetypes_of_data = AA.get_function(data.T).T
    archetypes_of_data = np.asarray(archetypes_of_data)
    print(" SHAPE : ",archetypes_of_data.shape)
    return(archetypes_of_data)
def get_arche(K,x_cap,class_labels,classes_ = False,reducer = None):
  AA = return_archetype();AA.K = K
  a_d = matdata_get_archetype(AA,x_cap,class_labels,
                      classes_ = classes_,reducer = reducer)
  return a_d
def get_data_class(data,d = [[20,0,0]]):
  n = data.shape
  c = np.ones(n[0])
  y = np.zeros(n[0])
  new_data = data
  k = 1
  for i in d:
      new_data = np.vstack((new_data,data+(np.ones(n)*np.asarray(i))))
      y = np.hstack((y,c*k))
      k = k + 1
  return((new_data,y))
class setup_rip():
  def __init__(self,data,y,K):
    self.data = data
    self.rand_data = None
    self.y = y
    self.un_class = np.unique(y)
    self.K = K
    self.persist_val = {}
    self.wdata_arche = None
    self.rand_cl = None
    self.c_aa = None
    self.near_pt = []
    self.rips = Rips(maxdim = 1)
  def get_data_persist(self,rips_type = "gv_rip",maxdim = 1):
    print("Data: ",self.data.shape)
    sec = tell_time()
    self.persist_val["data"] = return_persistence(self.data,rips_type,maxdim)
    print("Archetype: ")
    sec = tell_time(sec)
    K = self.K*len(self.un_class)
    self.wdata_arche = get_arche(K,self.data,self.y)
    self.persist_val["w_AA"] = return_persistence(self.wdata_arche,rips_type,maxdim)
    print("Random: ")
    sec = tell_time(sec)
    self.rand_data = self.data[np.random.choice(self.data.shape[0], K, replace=False), :]
    self.persist_val["w_R"] = return_persistence(self.rand_data,rips_type,maxdim)
    sec = tell_time(sec)
  def get_class_persist(self,rips_type = "gv_rip",maxdim = 1, dist_func = "euclidean"):
    print("Class archetype persistence : ")
    sec = tell_time()
    self.c_aa = get_arche(self.K,self.data,self.y,classes_ = True,reducer = None)
    print("Number of unique : ", np.unique(self.c_aa,axis =0).shape)
    self.persist_val["c_AA"] = return_persistence(stack_arr(self.c_aa),rips_type,maxdim)
    sec = tell_time(sec)
    print("Random class persistence : ")
    class_data = list(map(lambda i: self.data[np.where(self.y==i)[0],:],self.un_class))
    self.rand_cl = list(map(lambda value: value[np.random.choice(value.shape[0],self.K,False),:] ,class_data ))
    print("Random data shape of a class : ",self.rand_cl[0].shape)
    self.persist_val["c_R"] = return_persistence(stack_arr(self.rand_cl),rips_type,maxdim)
    sec = tell_time(sec)
    for i in tqdm(range(len(self.un_class)),desc = "Get values from class : "):
      vdata = self.data[np.where(np.asarray(self.y)==self.un_class[i])[0],:]
      self.near_pt.append(close_by_ele(self.c_aa[i], vdata, funct = dist_func))
    self.persist_val["n_AA"] = return_persistence(stack_arr(self.near_pt),rips_type,maxdim)
  def get_rips_plot(self,reducer = PCA(n_components = 2)):
    red_data = self.data.copy()
    if self.data.shape[1]>2:
      red_data = reducer.fit_transform(self.data)
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.scatter(red_data[:,0], red_data[:,1], s=4)
    plt.title("Scatter plot ")
    plt.subplot(122)
    self.rips.plot(self.dgms, legend=False, show=False)
    plt.title("Persistence diagram of $H_0$ and $H_1$")
    plt.show()


