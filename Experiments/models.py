'''
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import datasets
from skimage import io
from PIL import Image
import torchvision.models as models
torch.manual_seed(17)
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Subset
def index_(data_set,cl_):
  idx=[]
  for i in range(len(data_set)):
    x=data_set[i]
    if x[1]==cl_:idx.append(i)
  return(idx)

  def red_(val):
  for i in range(len(val)):
    if val[i].size==0:
      val.pop(i)
    x=np.asarray(list(map(lambda i:np.asarray(min(i.flatten()),max(i.flatten())),val)))
  return([np.min(x.flatten()),np.max(x.flatten())])

  def grey_(img,dim=32,ax="off"):
  gray_image = transforms.Grayscale(num_output_channels= 1)
  img=gray_image(img)
  plt.imshow(img.reshape(dim,dim))
  if ax=="off":
    plt.axis(ax)
  plt.show()

  class set_up():
  def __init__(self,pre_trained=True):
    self.model=None
    self.dataset=None
    self.index_data=None
    self.pre_trained=pre_trained
  def get_model(self,model_name):
    if model_name=="vgg16":
      self.model=models.vgg16(pretrained=self.pre_trained)
    elif model_name=="resnet18":
      self.model=models.resnet18(pretrained=self.pre_trained)
    elif model_name=="resnet50":
      self.model=models.resnet50(pretrained=self.pre_trained)
    elif model_name=="resnet101":
      self.model=models.resnet101(pretrained=self.pre_trained)
    elif model_name=="resnet152":
      self.model=models.resnet152(pretrained=self.pre_trained)
    elif model_name=="mobilenetv2":
      self.model=models.mobilenet_v2(pretrained=self.pre_trained)
    elif model_name=="mnasnet1_0":
      self.model=models.mnasnet1_0(pretrained=self.pre_trained)
    elif model_name=="densenet121":
      self.model=models.densenet121(pretrained=self.pre_trained)
    elif model_name=="densenet169":
      self.model=models.densenet169(pretrained=self.pre_trained)
    elif model_name=="densenet201":
      self.model=models.densenet201(pretrained=self.pre_trained)
    else:
      print("Warning model not found")
  def get_dataset(self,dataset_name,train_=False,test=True,split="test",resize_=256,centre_crop=224,trans_val=None):
    com_trans=None
    if  dataset_name=="cifar10" or dataset_name=="cifar100" or dataset_name=="stl10":
      if trans_val==None:
        com_trans=transforms.Compose([transforms.Resize(resize_),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),])
        print("3-D")
      elif trans_val=="norm":
        com_trans=transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor(),transforms.Normalize((0,),(1,))])
        print("Grayscale Normal")
    elif trans_val==None or trans_val=="norm":
      if trans_val==None:
        print("resize and centrecrop_done")
        com_trans=transforms.Compose([transforms.Resize(resize_),transforms.CenterCrop(centre_crop),transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),])
      elif trans_val=="norm":
        print("grayscale")
        com_trans=transforms.Compose([transforms.Resize(resize_),transforms.CenterCrop(centre_crop),
                                      transforms.Grayscale(num_output_channels=1),transforms.ToTensor(),
                                      transforms.Normalize((0,),(1,))])
    else:
      com_trans=trans_val
    if dataset_name=="cifar10":
      self.dataset= datasets.CIFAR10("cifar10",train=train_, download=True,transform=com_trans)
    elif dataset_name=="cifar100":
      self.dataset=datasets.CIFAR100("cifar100",train=train_,download=True,transform=com_trans)
    elif dataset_name=="stl10":
      self.dataset=datasets.STL10("stl10",split=split,download=True,transform=com_trans)
    elif dataset_name=="aircraft":
      self.dataset=datasets.FGVCAircraft("FGVCAircraft",split=split,download=True,transform=com_trans)
    elif dataset_name=="dtd":
      self.dataset=datasets.DTD("dtd",split=split,download=True,transfrom=com_trans)
    elif dataset_name=="cars":
      self.dataset=datasets.StandfordCars("standfordcars",split=split,download=True,transform=com_trans)
    elif dataset_name=="sun":
      self.dataset=datasets.SUN397("SUN397",download=True,transform=com_trans)
    elif dataset_name=="caltech101":
      self.dataset=datasets.Caltech101("caltech101",download=True,transform=com_trans)
    else:
      print("Warning dataset is empty ")
  def dataset_index(self,indices=False,n0_indices=10,sub_set=False,path_archetype=None,n_comp=625):
    if indices==True and sub_set=="random":
      self.index_data=torch.randperm(len(self.dataset))[:n0_indices]
    elif indices == True and sub_set == "class":
      self.index_data=list(map(lambda ele_:index_(self.dataset,ele_),range((len(self.dataset.classes)))))
    else:
      print("Warning no indices ")
'''
