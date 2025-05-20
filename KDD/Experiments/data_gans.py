import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
def return_data(data_name="mnist"):
  data_dictionary = {"mnist":datasets.MNIST,"fmnist":datasets.FashionMNIST,"cifar10": datasets.CIFAR10,"stl10":datasets.STL10, "aircraft":datasets.FGVCAircraft, "flower":datasets.Flowers102 }
  return(data_dictionary[data_name])
class get_balanced_data():
  def __init__(self,data_name="mnist",transform = None,transform_type="grey",sum_grey=False):
    self.transform = transform
    if transform_type=="grey" and data_name =="mnist":
      self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])
      print("Initialising the transform with the normalisation for greyscale with mean 0.5 and std 0.5.")
    elif transform_type=="grey" and data_name =="cifar10":
      self.transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Grayscale(num_output_channels=1),transforms.Normalize(mean=[0.5],std=[0.5])])
      print("Initialising the transform with the normalisation for greyscale with mean 0.5 and std 0.5.")
    elif transform_type == "3d" and data_name =="stl10"or data_name=="aircraft"or data_name=="flower"or data_name=="cifar10":
      IMAGE_DIM = (64,64)
      if data_name=="cifar10":IMAGE_DIM = (32,32)
      self.transform = transforms.Compose([transforms.Resize((IMAGE_DIM[0],IMAGE_DIM[1])),transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))])
    dataset = None
    if data_name == "mnist" or data_name=="cifar10":
      dataset = return_data(data_name)(root='../data/', train=True, transform=self.transform, download=True)
    elif data_name=="stl10" or data_name=="flower" or data_name=="aircraft":
      IMAGE_DIM=(64,64)
      dataset = return_data(data_name)("data",split ="train",transform =self.transform,download=True)
    else: ValueError('data name only support mnist and cifar10')
    try :
         self.class_names = dataset.classes
         print("class names accessed ")
    except :
        self.class_names = "Not accessed by the method"
        print(self.class_names)
    self.indicies = []
    self.data_array = []
    shape_data = dataset[0][0].cpu().data.numpy().shape
    print("===================================================")
    print("The shape of the instance in dataset :  ", shape_data)
    print("===================================================")
    for i in range(len(dataset)):
      self.indicies.append(dataset[i][1])
      if shape_data[0]==1:self.data_array.append(dataset[i][0].cpu().data.numpy().reshape(shape_data[1],shape_data[2]))
      if sum_grey==True:
        image = dataset[i][0].cpu().data.numpy().T
        self.data_array.append(image.mean(axis=2))
      else :self.data_array.append(dataset[i][0].cpu().data.numpy().T)
    self.data_array = np.asarray(self.data_array)
    self.indicies = np.asarray(self.indicies)
    print("===================================================")
    print("The shape of the X array :  ", self.data_array.shape)
    print("The shape of the index array : ",self.indicies.shape)
    print("===================================================")
  def single_class_sample(self,un_class_label,sample_size = None):
    temp  =  np.where(self.indicies==un_class_label)[0]
    print("sample size : ",sample_size," : size of class  : ",len(temp))
    if sample_size>len(temp): ValueError('data name sample size greter than the class size ')
    single_sub_index = np.random.choice(temp,sample_size,replace=False)
    return(single_sub_index)
  def return_sample_set(self,type_sub = "balanced", sample_size = None,mclass = 2):
    #suppose I need a 200 sample
    sub_index = np.asarray([])
    unique_label =  np.unique(self.indicies)
    print("===================================================")
    if type_sub == "balanced":
      new_sample_size = int(sample_size/len(unique_label))
      for un in unique_label:
        sub_index = np.int_(np.hstack((sub_index,self.single_class_sample(un_class_label=un,sample_size = new_sample_size))))
    elif type_sub == "singleclass":
      un = int(np.random.choice(unique_label,1)[0])
      print("The class value is : ",un)
      sub_index = self.single_class_sample(un_class_label=un,sample_size = sample_size)
    elif type_sub == "multiclass":
      print("The number of classes in multi class is :  ",mclass)
      new_sample_size = int(sample_size/mclass)
      unique_label = np.random.choice(unique_label,mclass,replace=False);temp = [];
      print("The unique class choosen are  :  ",unique_label)
      for un in unique_label:
        sub_index = np.int_(np.hstack((sub_index,self.single_class_sample(un_class_label=un,sample_size = new_sample_size))))
    else:
      sub_index = np.random.choice(len(self.indicies),sample_size,replace=False)
    np.random.shuffle(sub_index)
    sub_index = np.int_(sub_index)
    print("The length of the sub index :  ", len(sub_index))
    print("Check if the sample has unique index : ", np.unique(sub_index).shape)
    for un in np.unique(self.indicies[sub_index]):
        print("class : ",un," inbalance  : ",np.where(self.indicies[sub_index]==un)[0].shape)
    print("The shape of the suset of data returned : ",self.data_array[sub_index].shape)
    print("===================================================")
    return([self.data_array[sub_index],self.indicies[sub_index]])
