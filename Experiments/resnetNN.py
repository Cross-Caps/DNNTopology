# Get the betti number min and max
import models
import Topology_functions as tf
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
# Get the betti number min and max
def red_(val):
  for i in range(len(val)):
    if val[i].size==0:
      val.pop(i)
  x=np.asarray(list(map(lambda i:np.asarray(min(i.flatten()),max(i.flatten())),val)))
  return([np.min(x.flatten()),np.max(x.flatten())])

#Get layer by layer feature
def feat_lay_net(model,image,layer_t):
  if layer_t == 4:
      out = []
      output = nn.Sequential(*list(model.children())[:4])(image)
      for i, mod in enumerate(list(model.children())):
        if isinstance(mod, nn.Sequential):
          for bn in mod:
            output = bn(output)
            n = output.shape
            out_nump = output.detach().numpy().reshape(n[1],n[2],n[3])
            out_nump = out_nump.T
            #print(n,end=",")
            out.append(tf.get_persist(out_nump,max_dim=2))
      return(out)
  n=image.shape
  out=nn.Sequential(*list(model.children())[:layer_t])(image)
  out_nump=out.detach().numpy()
  out_nump=out_nump.T
  n=out_nump.shape
  out_nump=out_nump.reshape(n[0],n[1],n[2])
  return(tf.get_persist(out_nump,max_dim=2))

#Get the betti numbers
    
def matrix_betti(model,image,layer_=[1,2,3,4,5],thres=-5):
  result_1=list(map(lambda x:feat_lay_net(model,image,x),layer_[:3]))
  result_1=result_1+feat_lay_net(model,image,4)
  bettis_=[]
  if type(thres)!=int and type(thres)!=float:
    #print("entered into listed section")
    for i_cap in thres:
      betti0_=[]
      for i in result_1:
        betti0_.append(list(map(lambda x:tf.betti(i,i_cap)[x],list(range(3)))))
      bettis_.append(betti0_)
    #print("Note the matrix here contains different threshold values ",np.asarray(bettis_).shape)
  else:
    for i in result_1:
      bettis_.append(list(map(lambda x:tf.betti(i,thres)[x],list(range(3)))))
  return(np.asarray(bettis_))

#Getting the betti curve 

def thresholdall_betti(model,image,data_name,layer_=[1,2,3,4,5],nth_betti=0,
                       ra_labx=0,ra_laby=10,bet_thres=0.125,ra_min=5,ra_max=5,save_to_path=None):
  result_1=list(map(lambda x:feat_lay_net(model,image,x),layer_[:3]))
  result_1=result_1+feat_lay_net(model,image,4)
  layer_=list(range(1,len(result_1)+1))
  ra_min_max=red_(result_1[0])
  for j in result_1[1:]:
    k=red_(j)
    ra_min_max=[min(ra_min_max[0],k[0]),max(ra_min_max[1],k[1])]
  ra_min_max=[int(ra_min_max[0])+ra_min,int(ra_min_max[1])+ra_max]
  betti0=[]
  #print("betti")
  ind=1
  for i in result_1:
    if len(i)==2 and nth_betti==2:
      print(ind," : warning")
      ind=ind+1
      continue
    betti0.append(list(map(lambda x:tf.betti(i,x)[nth_betti],np.arange(ra_min_max[0],ra_min_max[1],bet_thres))))
    ind=ind+1
  print(len(betti0))
  k=ra_labx+1
  ra_laby=min(ra_laby,len(betti0))
  plt.figure(figsize=(5,5))
  a=0;
  k0=[0,0]
  for i in betti0[ra_labx:ra_laby]:
    lay_=np.arange(ra_min_max[0],ra_min_max[1],bet_thres)
    #lay_=lay_/np.linalg.norm(lay_)
    plt.plot(lay_,
             np.asarray(i),label=str(k)+"-Blocks")
    a=max(a,np.amax(np.asarray(i)))
    k0[0]=min(k0[0],np.amin(np.asarray(lay_)))
    k0[1]=max(k0[1],np.amax(np.asarray(lay_)))
    plt.legend(loc="best",fontsize=10)
    k=k+1
  x_tick_val=[round(p,2) for p in np.arange(k0[0],k0[1],step=(k0[1]-k0[0])/4)]
  print(x_tick_val,k0,round((k0[1]-k0[0])/4))
  plt.xlabel("$\eta$",fontsize=16)
  plt.ylabel(r"$\beta$"+str(nth_betti),fontsize=16)
  k=k+1
  plt.xticks(x_tick_val+[round(k0[1],2)],fontsize=16)
  a=int(a)+1000
  y_ax=list(range(0,a,100))
  plt.yticks(xticks_(y_ax,4),yticks_(y_ax,4),fontsize=16)
  #   plt.legend(loc="best",fontsize=10)
  #   k=k+1
  # plt.xlabel("$\eta$",fontsize=20)
  # plt.ylabel(r"$\beta$"+str(nth_betti),fontsize=20)
  # plt.title(data_name,fontsize=16)
  if save_to_path !=None:
    plt.savefig(save_to_path, format='eps', bbox_inches='tight')
  plt.show()

  
class betti_matrix_get():
    def __init__(self,thres):
        self.Betti_matrix=[]
        self.thres=thres
        self.loader=None
    def get_betti(self,model,layer_=False):
        if layer_==False:
            mat_bet=list(map(lambda x:matrix_betti(model,x[0],thres=self.thres),self.loader))
        else:
            mat_bet=list(map(lambda x:matrix_betti(model,x[0],layer_,thres=self.thres),self.loader))
        self.Betti_matrix.append(mat_bet)
    def get_loader(self,sub_dat):
        self.loader=DataLoader(sub_dat, batch_size=1,shuffle=False,num_workers=2, pin_memory=True)
