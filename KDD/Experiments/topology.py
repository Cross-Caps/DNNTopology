import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve,PersistenceEntropy,PairwiseDistance
from scipy.stats import chisquare
from scipy import stats
from collections import Counter
from gtda.plotting import plot_diagram
import plotly.io as pio
import plotly.graph_objects as go
def generate_plotly_params(dict_values,add_val= None,del_val=None):
  values_map  =  {"title": "Plot Title", "xaxis": "X Axis Title","yaxis": "Y Axis Title","autosize":False,"width":500,"height":500,
                  "marginl":50,"marginr":50,"marginb": 10,"margint":10,"marginpad":0,"fontsize":24,"paper_bgcolor":'grey'}
  for i in list(dict_values.keys()):
    values_map[i] = dict_values[i]
  plotly_params = {"layout" : {
                             "font":dict(size=values_map["fontsize"],color='black'),
                             "title": dict(text=values_map["title"] ),
                             "xaxis": dict(title=dict(text=values_map["xaxis"],font = dict(size=values_map["fontsize"]))),
                             "yaxis": dict(title=dict(text=values_map["yaxis"],font = dict(size=values_map["fontsize"]))),
                             "autosize":values_map["autosize"],"width":values_map["width"],"height":values_map["height"],
                             "margin":dict(l=values_map["marginl"],r=values_map["marginr"],b=values_map["marginb"],t=values_map["margint"],pad=values_map["marginpad"]),
                             "paper_bgcolor":values_map["paper_bgcolor"]
                             }}
  if type(add_val)!=type(None):
    for key_val in list(add_val.keys()):
      plotly_params["layout"][key_val] =  add_val[key_val]
  if type(del_val)!=type(None):
    for key_val in del_val: plotly_params["layout"].pop(key_val)
  return plotly_params
values_map  =  {"title": "", "xaxis": "$\Large{\eta}$","yaxis": r"$\Large{\beta(\eta)}$","autosize":False,"width":400,"height":400,
                  "marginl":20,"marginr":20,"marginb": 10,"margint":5,"marginpad":0,"fontsize":18,"paper_bgcolor":'white'}
plotly_params = generate_plotly_params(values_map)
# fig =  img1_homology.BC.plot(img1_homology.betti_curves,plotly_params=plotly_params)
# #fig.update_traces(marker=dict(color="orange"),selector=dict(name="H2"))
# fig.show()
#some required plotly graphs
values_map  =  {"title": "", "xaxis": "$\Large{\eta}$","yaxis": r"$\Large{\beta(\eta)}$","autosize":False,"width":400,"height":400,
                  "marginl":20,"marginr":20,"marginb": 10,"margint":5,"marginpad":0,"fontsize":18,"paper_bgcolor":'white'}
plotly_params = generate_plotly_params(values_map)
d_meas = ['landscape','wasserstein']
values_map  =  {"title": "", "xaxis": "$\Large{\eta}$","yaxis": r"$\Large{\beta(\eta)}$","autosize":False,"width":500,"height":500,
                  "marginl":10,"marginr":10,"marginb": 10,"margint":10,"marginpad":10,"fontsize":10,"paper_bgcolor":'white'}
plotly_params_summary = generate_plotly_params(values_map,del_val=["xaxis","yaxis"])
values_map =  {"title": "", "xaxis": "$\Large{\eta}$","yaxis": r"$\Large{\omega(\eta)}$","autosize":False,"width":400,"height":400,
                  "marginl":20,"marginr":20,"marginb": 10,"margint":5,"marginpad":0,"fontsize":18,"paper_bgcolor":'white'}
plotly_params_sum = generate_plotly_params(values_map,{"template" : "simple_white"})
new_label_sub = ["I","I-NI(gn)","I-DNI(gn)","I-NI(s&p)","I-DNI(s&p)","I-NI(pn)","I-DNI(pn)","I-NI(sk)","I-DNI(sk)"]
values_map  =  {"title": "", "xaxis": "Data","yaxis": "PE","autosize":False,"width":500,"height":500,
                  "marginl":0,"marginr":0,"marginb": 0,"margint":10,"marginpad":0,"fontsize":20,"paper_bgcolor":'white'}
plotly_params_pe = generate_plotly_params(values_map)
def plot_diagm_pd(pdx,diag_num=0,homology_get=(0,1),save_fig = False,name="data", norm_tick = False):
  values_map  =  {"title": "", "xaxis": "$\Large{b(\eta)}$","yaxis": "$\Large{d(\eta)}$","autosize":False,"width":500,"height":500,
                  "marginl":0,"marginr":0,"marginb": 0,"margint":10,"marginpad":0,"fontsize":24,"paper_bgcolor":'white'}
  plotly_params_diagm = generate_plotly_params(values_map)
  fig =plot_diagram(pdx[diag_num],plotly_params=plotly_params_diagm)
  fig.update_traces(marker=dict(color="#19D3F3"),name ="$\Large{H_0}$",selector=dict(name="H0"))
  fig.update_traces(marker=dict(color="orange"),name ="$\Large{H_1}$",selector=dict(name="H1"))
  if len(homology_get)==3:
        fig.update_traces(marker=dict(color="#4af04a"),name ="$\Large{H_2}$",selector=dict(name="H2"))
  fig.update_layout(legend=dict(
    yanchor="bottom",y=0.,
    xanchor="right",x=1))
  if norm_tick==True:
    #max_val  = np.max(pdx[diag_num])
    #min_val  = np.min(pdx[diag_num])
    val = [0,127,250]
    #val = [-500,0,500,1000]
    #val_text = ["$\Large{0}$","$\Large{127}$","$\Large{250}$"]
    val_text = ["$\Large{"+str(i)+"}$" for i in val ]
    fig.update_layout(
        yaxis = dict(tickvals = val, ticktext = val_text,tickwidth = 4,tickfont=dict( size=24)),
        xaxis = dict(tickvals = val, ticktext = val_text,tickwidth = 4,tickfont=dict( size=24))
#    xaxis = dict(tickvals = [min_val,(min_val+max_val)/2,max_val],tickwidth = 2)
    )
    fig.update_xaxes( linewidth=2)
    fig.update_yaxes( linewidth=2)
    fig.update_layout(showlegend=False)
  pio.show(fig)
  if save_fig==True:fig.write_image(name+"pdiag.pdf")
def norm_arr(arr):
  arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
  return(arr)
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
class topo_measures():
  def __init__(self,data,data_name="image",to_grey=True,homology_get = (0,1),infinity_values = 256,reduced_homology=False,diag_num = 0,save_fig=False,rips=False):
    self.cp = CubicalPersistence(homology_get,n_jobs=-1,infinity_values=infinity_values,reduced_homology=reduced_homology)
    if rips == True :
      print("changing to vietoris rips")
      self.cp = VietorisRipsPersistence(homology_dimensions= homology_get)
    self.BC = BettiCurve()
    self.betti_curves = None
    self.pdx = None
    self.org_pdx = None
    self.sum_betti = None
    self.summaries ={"PE":None,'landscape':None,'wasserstein':None,"betti":None,'nbetti':None}
    self.other_summaries = {"mse":[],"ssim":[],"fid":[],"ins":[],"psnr":[],"npsnr":[]}
    self.data_name="image"
    self.data_numpy = data
    self.data = torch.from_numpy(data)
    #self.data_numpy = data.cpu().data.numpy()
    shape_val = self.data_numpy.shape
    if len(shape_val)<3:
        self.data_numpy =   self.data_numpy.reshape(1,shape_val[0],shape_val[1])
    if to_grey==True:
      if len(shape_val)>3:
        self.data_numpy = self.data_numpy.mean(axis=3)
        print(self.data_numpy.shape)
    self.pdx = self.cp.fit_transform(self.data_numpy)
    self.save_fig=save_fig
  def norm_pdx(self,diag_num = 0,save_fig=False,set_min=None):
    self.org_pdx = self.pdx.copy()
    for i in range(len(self.org_pdx)):
      arr = self.pdx[i][:,:2]
      print(np.max(arr),np.min(arr))
      if type(set_min)!=type(None):
        arr = (arr-set_min)/(np.max(arr)-set_min)
      else:
        arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
      self.pdx[i][:,:2] = arr
    plot_diagm_pd(pdx=self.pdx,diag_num=diag_num,homology_get=(0,1),save_fig = save_fig,name="norm_"+self.data_name)
  def topo_metric(self,pdx,p=2,m_name=['landscape'],diag_num=0):
    if self.save_fig==True:
      plot_diagm_pd(cp=self.cp,pdx=pdx,diag_num=diag_num,homology_get=(0,1),save_fig = False,name=self.data_name)
    self.sum_betti = []
    self.betti_curves = self.BC.fit_transform(pdx)
    for i in range(len(self.betti_curves)):
      values = {}
      for dim in list(self.BC.samplings_.keys()):
        dict_val = dict(zip(self.BC.samplings_[dim], self.betti_curves[i][dim]))
        values=Counter(dict_val)+Counter(values)
      self.sum_betti.append(list(values.values()))
    self.sum_betti = np.asarray(self.sum_betti,dtype="object")
    self.summaries['betti'] = []
    for i in range(len(self.sum_betti)):
      #self.summaries['betti'].append(chisquare(self.sum_betti[i])[0])
      self.summaries['betti'].append(np.asarray(self.sum_betti[i]).mean())
    self.summaries['betti'] = np.asarray(self.summaries['betti'])
    if 'landscape' in m_name :
      PD = PairwiseDistance(metric='landscape',metric_params={'p': p,'n_layers': 5, 'n_bins': 1000},order=None)
      self.summaries['landscape'] = PD.fit_transform(pdx)
    if 'wasserstein' in m_name:
      PD = PairwiseDistance(metric='wasserstein',metric_params={'p': p, 'delta': 0.1},order=None)
      self.summaries['wasserstein'] = PD.fit_transform(pdx)
    self.summaries["PE"]= PersistenceEntropy().fit_transform(pdx)
    indx = np.where(self.summaries["PE"]<0)
    self.summaries["PE"][indx] = 0
  def val_acrossPD(self,across_val=0):
    #compute the E as sum EK
    self.summaries["E"] = self.summaries["PE"].sum(axis=1)
    self.summaries["delta"] = np.abs(self.summaries["E"]-self.summaries["E"][across_val])
    self.summaries["ndelta"] = norm_arr(self.summaries["delta"])
    self.summaries['nbetti'] = norm_arr(self.summaries["betti"])
  def ssim(self,data):
    length_image = len(data)
    for i in range(length_image ):
      self.other_summaries["ssim"].append([])
      self.other_summaries["mse"].append([])
      self.other_summaries["psnr"].append([])
      for j in range(length_image ):
        min_val = np.min([np.min(data[i]),np.min(data[j])])
        max_val = np.max([np.max(data[i]),np.max(data[j])])
        self.other_summaries["ssim"][i].append(ssim(data[i],data[j],data_range=max_val - min_val))
        self.other_summaries["mse"][i].append(mean_squared_error(data[i],data[j]))
        self.other_summaries["psnr"][i].append(PSNR(data[i],data[j]))
      self.other_summaries["npsnr"].append(norm_arr(self.other_summaries["psnr"][i]))
    self.other_summaries["ssim"]=np.asarray(self.other_summaries["ssim"])
    self.other_summaries["mse"]=np.asarray(self.other_summaries["mse"])
    self.other_summaries["psnr"]= np.asarray(self.other_summaries["psnr"])
    self.other_summaries["npsnr"]= np.asarray(self.other_summaries["npsnr"])
