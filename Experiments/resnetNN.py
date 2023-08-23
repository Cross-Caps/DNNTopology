'''

# Get the betti number min and max
import models
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
            out.append(get_persist(out_nump,max_dim=2))
      return(out)
  n=image.shape
  out=nn.Sequential(*list(model.children())[:layer_t])(image)
  out_nump=out.detach().numpy()
  out_nump=out_nump.T
  n=out_nump.shape
  out_nump=out_nump.reshape(n[0],n[1],n[2])
  return(get_persist(out_nump,max_dim=2))

#Get the betti numbers
    
def matrix_betti(model,image,layer_=[1,2,3,4,5],thres=-5):
  result_1=list(map(lambda x:feat_lay_net(model,image,x),layer_[:3]))
  result_1=result_1+feat_lay_net(model,image,4)
  bettis_=[]
  for i in result_1:
    bettis_.append(list(map(lambda x:betti(i,thres)[x],list(range(3)))))
  return(np.asarray(bettis_))

'''
