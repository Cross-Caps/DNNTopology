import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
def plot_violin_resize(path,name_data,resize_size=[28,32,40,64,80,90,150]):
  plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':True, 'top':True})
  colors = cm.RdPu(np.linspace(0, 1, 9))
  data_resize_all = np.load(path,allow_pickle=True)
  val = {"$E$":[],"dataset":[]}
  for i in range(len(resize_size)):
      make_list = list(data_resize_all[i])
      val["$E$"] = val["$E$"]+make_list
      val["dataset"]=val["dataset"]+[resize_size[i]]*len(make_list)
  df =  pd.DataFrame(val)
  Means = df.groupby('dataset')['$E$'].mean()
  fig=plt.figure(figsize =(9, 6))
  sns.violinplot(data=df,y="$E$",x="dataset",hue="dataset",density_norm="count",color = "green",palette=colors)
  plt.xlabel("")
  plt.legend(ncol=1+(len(resize_size)//2),fontsize=18)
  plt.ylabel(r"$\it{E}$",fontsize=30,weight="bold")
  plt.yticks([0,5,10],fontsize=30)
  plt.xticks(fontsize=18)
  plt.savefig(name_data+"_resize_violin.pdf",bbox_inches = "tight")
  plt.show()
