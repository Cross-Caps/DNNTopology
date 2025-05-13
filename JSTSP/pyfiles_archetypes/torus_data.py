import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
def torus_dt(n = 40,r=1,R=2):
  angle = np.linspace(0, 2 * np.pi, n)
  theta, phi = np.meshgrid(angle, angle)
  X = (R + r * np.cos(phi))* np.cos(theta)
  Y = (R + r * np.cos(phi))* np.sin(theta)
  Z = r * np.sin(phi)*3
  visual_mesh(X,Y,Z)
  return([X,Y,Z])
def add_class(l,n,labels = None):#l = the number of instances in data ,  n =  the number of classes needed
  if type(labels) == type(None) :
    class_labels = [];k = -1;n = l//n
    for i in range(l):
      if i%n==0:
        k = k + 1
      class_labels.append(k)
  else :
    class_labels = labels
  return np.asarray(class_labels)
def funct_return(x,y):
  print(x.shape)
  R = (x-2)*x*(x-1)**2
  print(R.shape)
  R = R+(y**2)
  R = R**2
  Z = np.sqrt(40 - R)
  return(Z)
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
def visual_mesh(X,Y,Z,class_labels = None):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"},constrained_layout=True)
  if type(class_labels)==type(None):
    ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)
  else:
    for i in np.unique(class_labels):
      ax.plot_surface(X[class_labels==i], Y[class_labels==i], Z[class_labels==i], vmin=Z.min() * 2)
  ax.set_xlabel("x - axis")
  ax.set_ylabel("y - axis")
  ax.set_zlabel("z - axis")
  ax.zaxis.labelpad = -4
  plt.show()
def visual_2d_mesh(x0,y0,title,class_labels = None):
  x0 = np.asarray(x0).flatten()
  y0 = np.asarray(y0).flatten()
  fig, axes = plt.subplots(1, 2, figsize=(10,3))
  if type(class_labels) == type(None):
    axes[0].plot(x0,y0,color = 'r')
    axes[1].scatter(x0,y0,color = 'r')
  else :
    uni_que = np.unique(class_labels)
    k = 0
    for i in uni_que:
      axes[0].plot(x0[class_labels==i],y0[class_labels==i],color = "C"+str(k),label = "class-"+str(i))
      axes[1].scatter(x0[class_labels==i],y0[class_labels==i],color = "C"+str(k),label = "class-"+str(i))
      k = k+1
  plt.title(title.upper())
  plt.legend(loc="best")
  plt.show()
def plot_dict(dict_var,sx = 7,sy = 7):
  keys_ = list(dict_var.keys())
  values =  [dict_var[i] for i in keys_]
  fig = plt.figure(figsize=(sx,sy))
  ax = fig.add_subplot(111)
  plt.plot(keys_,values,marker='*', mfc='red')
  for i,j in zip(keys_,values):
    ax.annotate(str(round(j,2)),xy=(i,j),horizontalalignment='right')
def stack_arr(arr):
  vdata  = arr[0]
  for i in arr[1:]:vdata  = np.vstack((vdata,i))
  print("Shape of Data : ",vdata.shape)
  return vdata
def rand_sample(data,sample_size):
  rand_index = np.random.choice(list(range(data.shape[0])),sample_size,replace=False)
  print("The shape of random data : ",data[rand_index,:].shape)
  return(data[rand_index,:])
def get_data(data,sample_size,noise_pert,mu=0,sigma=0.1):
  rand_data = rand_sample(data,sample_size)
  x_1, x_2 = train_test_split(rand_data,test_size = noise_pert )
  print(x_1.shape,x_2.shape)
  noise = np.random.normal(mu, sigma, x_2.shape)
  data0 = np.vstack((x_1,x_2 + noise))
  print("Data with noise shape : ",data0.shape)
  return(data0)
