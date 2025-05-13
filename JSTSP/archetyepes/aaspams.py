import spams as sp
import numpy as np
class return_archetype():
  def __init__(self):
    self.K = 200 # learns a dictionary with 64 elements
    self.robust = False # use robust archetypal analysis or not, default parameter(True)
    self.epsilon = 1e-3 # width in Huber loss, default parameter(1e-3)
    self.computeXtX = True # memorize the product XtX or not default parameter(True)
    self.stepsFISTA = 0 # 3 alternations by FISTA, default parameter(3)
    # a for loop in FISTA is used, we stop at 50 iterations
    # remember that we are not guarantee to descent in FISTA step if 50 is too small
    self.stepsAS = 10 # 7 alternations by activeSet, default parameter(50)
    self.randominit = True # random initilazation, default parameter(True)
    self.numThreads = -1
    self.returnAB= True
  def get_function(self,data):
    I = np.asfortranarray(data,dtype = np.float64)
    (Z,A,B) = sp.archetypalAnalysis(
    np.asfortranarray(I), returnAB= self.returnAB, p = self.K, \
    robust = self.robust, epsilon = self.epsilon,
    computeXtX = self.computeXtX,  stepsFISTA = self.stepsFISTA ,
    stepsAS = self.stepsAS, numThreads = self.numThreads)
    return(Z)