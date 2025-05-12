import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
def get_tau(average_life): #computing tau from average life 
  n = len(average_life)
  norm_val = []
  for i in range(n):
    norm_val.append(np.linalg.norm(average_life[i],ord = 1)/len(average_life[i]))
  norm_val = 1/np.asarray(norm_val)
  return norm_val
class class_plot_figure():
  def __init__(self,x_values,y_values,labels ):
    self.x_values = x_values
    self.y_values = y_values
    self.marker = ["v","^","<",">","P","X","*","h","H"]
    self.colors = ["darkblue","blue","teal","cyan","hotpink","green","darkred","red","orange"]
    self.model_names=["ResNet18","ResNet50","ResNet101","ResNet152","MobileNetV2","MnasNet1_0","DenseNet-121","DenseNet-169","DenseNet-201"]
    self.labels = labels
    self.data_name =  ["CIFAR10","STL10","Aircraft","CIFAR100"]
  def plot_fig(self,del_mem=[-1,-3],save_to_path=None,fit_line =True):
      fig, ax = plt.subplots(figsize=(4, 4))
      new_acc = []
      new_x_values = []
      for k in range(len(self.y_values)):
        if k in del_mem :
                continue
        new_x_values.append(self.x_values[k])
        new_acc.append(self.y_values[k])
        plt.plot(self.x_values[k],self.y_values[k],self.marker[k],markersize=15,label=self.labels[k],color = self.colors[k])
      #plt.legend(loc = 'best',ncol=len(self.labels),bbox_to_anchor=(0, -0.1))
      #plt.legend(loc='upper left',ncol=len(new_acc),bbox_to_anchor=(0, 1.12))
      plt.legend(loc='upper left',ncol=5,bbox_to_anchor=(-0.5, 1.3),fontsize=16)
       #print(val_,acc_mod)
      if fit_line == True:
        y0 = get_curve(new_x_values,new_acc)
        plt.plot(new_x_values,y0,color="grey",ls="--")
      ax.set_ylabel(r"$\tau$", fontsize=25)
      ax.set_xlabel("Accuracy",fontsize=16)
      ax.tick_params(labelsize=16)
      #plt.xticks([0,0.5,1])
      if type(save_to_path) !=type(None):
         plt.savefig(save_to_path+"_accuracy.pdf",format="pdf",bbox_inches='tight')
      plt.show()
      print("Correlation : ",np.corrcoef(new_x_values,new_acc))
      res = stats.pearsonr(new_x_values, new_acc)
      print("Pearson: Correlation : ",res.statistic,": p -value : ",res.pvalue)
      res = stats.spearmanr(new_x_values, new_acc)
      print("Spearman :Correlation : ",res.statistic,": p -value : ",res.pvalue)
      res = stats.kendalltau(new_x_values, new_acc)
      print("Kendall :Correlation : ",res.statistic,": p -value : ",res.pvalue)
  def plot_diff_data(self,list_score,list_acc,del_mem=[-1,-3],save_to_path=None,fit_line =True,del_add_cor=[]):
      fig, ax = plt.subplots(figsize=(15, 4),nrows=1, ncols=len(self.data_name))
      name_order = ""
      for d in range(len(self.data_name)):
        self.x_values = list_acc[d]
        self.y_values = list_score[d]
        new_acc = []
        new_x_values = []
        for k in range(len(self.y_values)):
          if k in del_mem :
                continue
          new_x_values.append(self.x_values[k])
          new_acc.append(self.y_values[k])
          ax[d].plot(self.x_values[k],self.y_values[k],self.marker[k],markersize=15,label=self.labels[k],color = self.colors[k])
          #ax[d].text(0.1,0.5,str(round(np.corrcoef(new_x_values,new_acc)[0,1],2)))
        if fit_line == True:
          y0 = get_curve(new_x_values,new_acc)
          ax[d].plot(new_x_values,y0,color="black")
        print("Data Name",self.data_name[d])
        res = stats.pearsonr(new_x_values, new_acc)
        print("==========================================================")
        print("Method 1: Correlation : ",res.statistic,": p -value : ",res.pvalue)
        res = stats.spearmanr(new_x_values, new_acc)
        print("Method 2:Correlation : ",res.statistic,": p -value : ",res.pvalue)
        print("===========================================================")
        res = stats.kendalltau(new_x_values, new_acc)
        print("Method 3:Correlation : ",res.statistic,": p -value : ",res.pvalue)
        print("===========================================================")
        ax[d].set_xlabel("Accuracy",fontsize=16)
        ax[d].tick_params(labelsize=16)
        name_order = name_order+" "+self.data_name[d]
      #plt.legend(loc = 'best',ncol=len(self.labels),bbox_to_anchor=(0, -0.1))
      plt.legend(loc='upper left',ncol=len(new_acc),bbox_to_anchor=(-3, 1.12))
      ax[0].set_ylabel(r"$\Lambda$", fontsize =20)
      plt.title()
      #ax.set_xlabel("Accuracy",fontsize=16)
      #ax.tick_params(labelsize=16)
      if type(save_to_path) !=type(None):
         plt.savefig(save_to_path+"_accuracy.pdf",format="pdf",bbox_inches='tight')
      plt.show()
      print("Correlation : ",np.corrcoef(new_x_values,new_acc))
def get_curve(X_true,y): #plotting line in the graph 
  #X = X_true/np.linalg.norm(X_true)
  X = np.asarray(X_true)
  X = X.reshape(-1, 1)
  reg = LinearRegression().fit(X, y)
  y0 = reg.predict(np.asarray(X_true).reshape(-1, 1))
  return(y0)
def plot_by_data(average_life_cifar10,average_life_stl10,average_life_aircraft,average_life_cifar100,model_names,index=0):
  from cycler import cycler # same model different data 
  colors = plt.cm.cool(np.linspace(0,1,5))
  f, ax = plt.subplots()
  cy = cycler('color', colors)
  ax.set_prop_cycle(cy)
  ax.plot(average_life_cifar10[index],label = "CIFAR10")
  ax.plot(average_life_stl10[index],label = "STL10")
  ax.plot(average_life_cifar100[index],label = "CIFAR100")
  ax.plot(average_life_aircraft[index],label = "Aircraft")
  val_max =  round(max([max(average_life_cifar10[index]),max(average_life_stl10[index]),max(average_life_aircraft[index])]),1)
  d = round(val_max/3,1)
  #ax.plot(new_average[i,:],label = labels[i])
  #plt.xticks([10,20,30,40,50,60,70],fontsize=20)
  #plt.yticks([0,2,4,6,8,10,12],fontsize=20)
  #ax.set_xticklabels(x_label)
  #plt.xticks([0,3,6,9,12,15,18],fontsize=20)
  plt.xticks(fontsize=20)
  #print(d,val_max)
  y_label = np.arange(0,val_max,d)
  if len(y_label)==4:y_label[3]= val_max
  else : y_label = np.append(y_label,[val_max])
  plt.yticks(y_label,fontsize=20)
  #plt.yticks([0.5,1.5,2.5,3.5],fontsize=20)
  plt.xlabel("Blocks",fontsize =20)
  plt.ylabel(r"$\lambda$",fontsize =20)
  #plt.legend(fontsize =17)
  plt.savefig(model_names[index]+"80_mean_length.pdf",format="pdf",bbox_inches='tight')
  plt.show()
