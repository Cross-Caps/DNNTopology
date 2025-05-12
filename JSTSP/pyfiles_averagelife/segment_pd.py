import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
class segment_stat(): # gets the statistics from the PD
  def __init__(self):
    self.values = {"sum_length":[],"mean_length":[],"H_max":[],
                   "variance_length":[], "num_of":[]} 
    self.key_val = None
    self.seg = {}
    self.seg_mat = []
  def get_seg(self,dict_val):
    self.key_val = list(dict_val.keys())
    for i in self.key_val:
      self.seg[i] = []
      for j in dict_val[i]:
        temp = abs(j[:,0]-j[:,1])
        temp = np.asarray([i if i<10**6 else 0 for i in temp])
        if len(temp)>0:
          self.seg[i].append(temp)
        else:
          self.seg[i].append(np.zeros((1)))
    for j in range(len(self.seg[self.key_val[0]])):
      self.seg_mat.append([])
      for i in range(len(self.key_val)):
        self.seg_mat[j].append(self.seg[self.key_val[i]][j])
    self.seg_mat = np.asarray(self.seg_mat,dtype=object)
  def get_stats(self):
    for i in self.key_val:
      self.values["sum_length"].append(list(map(lambda k:reduce(lambda a,b:a+b,k),self.seg[i])))
      self.values["mean_length"].append(list(map(lambda k:sum(k)/len(k),self.seg[i])))
      self.values["num_of"].append(list(map(lambda k:len(k),self.seg[i])))
      self.values["H_max"].append(list(map(lambda k:max(k),self.seg[i])))
      self.values["variance_length"].append(list(map(lambda k:np.var(k),self.seg[i])))
  def plot(self,name,data_name,only_dim=None):
    values = self.values[name]
    if type(only_dim)==type(None):
      for i in range(len(values[0])):
        plt.plot(self.key_val, [j[i] for j in values],label="H"+str(i))
    else:
       plt.plot(self.key_val, [j[only_dim] for j in values],label="H"+str(only_dim))
    plt.legend(loc="best")
    plt.title(" Homology "+data_name+" Stats "+name)
    plt.show()
def stats_for_all(layers_dgm): #gets the stats for all the layers 
  values = []
  for n in range(len(layers_dgm)):
    dict_val = {}
    for i in range(1,len(layers_dgm[n])):
      dict_val[i] = layers_dgm[n][i][0]
    S = segment_stat()
    S.get_seg(dict_val = dict_val)
    S.get_stats()
    values.append(S)
  return values

def return_average_life(list_segment_stats,listx_label,list_names,type_avg = "whole",n_color=3,name="",plot_=False):#compute for listed index
  from cycler import cycler
  average_life = []
  colors = plt.cm.hot(np.linspace(0,1,n_color))
  f, ax = plt.subplots()
  cy = cycler('color', colors)
  ax.set_prop_cycle(cy)
  if len(list_segment_stats)!=len(list_names): return("The list should have same length")
  if type_avg=="by_homology": return(list_segment_stats[0].values["mean_length"])
  for i in range(len(list_segment_stats)):
    counter = np.asarray(list_segment_stats[i].values['sum_length']).sum(axis=1)
    num_of = np.asarray(list_segment_stats[i].values['num_of']).sum(axis=1)
    average_life.append(counter/num_of)
    x_label = np.asarray(list(range(len(average_life[i]))))
    x_label = (x_label-np.min(x_label))/(np.max(x_label)-np.min(x_label))
    if plot_==True:ax.plot(x_label,average_life[i],label = list_names[i])
  if plot_==False:return(average_life)
  #plt.xticks([0,0.2,0.4,0.6,0.8,1],fontsize=20)
  #plt.yticks([0,0.75,1.5],fontsize=20)
  #plt.yticks([0.5,1.5,2.5,3.5],fontsize=20) # stl10 40
  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)
  plt.xlabel("Interval",fontsize =20)
  plt.ylabel(r"$\lambda$",fontsize =20)
  plt.legend()
  plt.savefig(name+"mean_length.pdf",format="pdf",bbox_inches='tight')
  plt.show()
  return(average_life)

def return_average_life_layers(list_segment_stats,labels,n_color=10,name="",r=1,plot_=False):#compute the average life all layers  using the statisticsnin pd
  from cycler import cycler
  average_life = []
  colors = plt.cm.plasma(np.linspace(0,1,n_color))
  f, ax = plt.subplots()
  cy = cycler('color', colors)
  ax.set_prop_cycle(cy)
  for i in range(len(list_segment_stats)):
    counter = np.asarray(list_segment_stats[i].values['sum_length']).sum(axis=1)
    num_of = np.asarray(list_segment_stats[i].values['num_of']).sum(axis=1)
    average_life.append([counter/num_of])
  average_life = np.asarray(average_life)
  average_life = average_life.reshape(average_life.shape[0],average_life.shape[2])
  new_average = [] # ploting after normalising 
  average_life = average_life.T
  print("The shape of average life",average_life.shape)
  if plot_==False:return(average_life)
  x_label = []
  for i in range(len(average_life)):
    new_average.append([])
    for j in range(len(average_life[i])):
      if j%r==0:
        new_average[i].append(average_life[i,j])
        if i==0: x_label.append(str(j))
  new_average = np.asarray(new_average)
  print(new_average.shape)
  for i in range(average_life.shape[0]):
     #ax.plot(average_life[i,:],label = i)
     ax.plot(new_average[i,:],label = labels[i])
  plt.xticks([11,22,33,44,55,66],fontsize=20)
  #plt.yticks([0,2,4,6,8,10,12],fontsize=20)
  #ax.set_xticklabels(x_label)
  #plt.xticks(fontsize=20)
  #plt.yticks([0.5,1.5,2.5,3.5],fontsize=20)
  #plt.yticks([0,4.5,9,13.5,18],fontsize=20)
  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)
  plt.xlabel("Layer",fontsize =20)
  plt.ylabel(r"$\lambda$",fontsize =20)
  plt.legend(fontsize =16)
  plt.savefig(name+"mean_length.pdf",format="pdf",bbox_inches='tight')
  plt.show()
  return(average_life)
