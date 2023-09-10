'''
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np
class TTP():
  def __init__(self,degree = 3,vector = [0,1,0.5,0.5]):
    self.TTP_score = []
    self.degree = degree
    self.differ_coef = np.asarray(vector).reshape(len(vector),1)
  def TTPscore(self,coef_values):
    if self.degree+1!= self.differ_coef.shape[0]:raise ValueError("The differential coeffient and coeffients of polynomial are not coinciding")
    self.TTP_score.append(coef_values.dot(self.differ_coef)[0])
  def polynomial_fit(self,betti_value,layer_value,
                     interaction_only = True,
                     funct = Ridge(alpha=1e-3),plot= False):
    betti_value  = betti_value
    n = betti_value.shape
    if len(n)!=4: raise ValueError("Enter 4D data (nclass,nsamples,nlayer,nbetti)")
    betti_value = np.sum(betti_value,axis = 3)
    betti_value = np.mean(betti_value,axis = 1)
    betti_value = betti_value.mean(axis = 0)
    x  = np.asarray(list(range(1,n[2]+1))).reshape(-1, 1)
    #x  = x/np.linalg.norm(x)
    y = betti_value/np.linalg.norm(betti_value)
    polynomial_model = make_pipeline(PolynomialFeatures(degree = self.degree,
                          interaction_only=interaction_only),funct)
    polynomial_model.fit(x,y)
    y_pred = polynomial_model.predict(x)
    if plot == True:
      plt.figure(figsize = (5,5))
      plt.plot(x , y_pred)
    return(polynomial_model.steps[1][1].coef_)
  def get_score(self,Betti_values,layer_value,interaction_only = False,funct = Ridge(alpha=1e-3),plot= False):
    n = len(Betti_values)
    self.TTP_score = []
    for i in range(n):
      coef_values = self.polynomial_fit(Betti_values[i],layer_value,
          interaction_only = interaction_only,funct = funct,plot= plot)
      self.TTPscore(coef_values)
  def marker_plot(self,accuracy,model_names,save_to_path=None):
      fig, ax = plt.subplots(figsize=(4, 4))
      if len(self.TTP_score)!=len(accuracy) or len(accuracy)!=len(model_names):raise ValueError("Number of values in accuracy doesn't match with other parameters")
      names = list(model_names.keys())
      for k in range(len(accuracy)):
        plt.plot(self.TTP_score[k],accuracy[k],model_names[names[k]],markersize=15,label=names[k])
      ax.set_xlabel(r'$\theta$', fontsize=16)
      ax.set_ylabel("Accuracy",fontsize=16)
      ax.tick_params(labelsize=16)
      plt.legend(loc="upper right",ncol=5,bbox_to_anchor=(2, 1.25),fontsize=14)
      if save_to_path !=None:
        plt.savefig(save_to_path, format='eps', bbox_inches='tight')
'''
