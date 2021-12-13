"""
Package to build the LeNet architecture for a CNN, 
This is just a basic example of a LeNet.
"""

#import the important PyTorch NN feaatures
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch import flatten

scaler = StandardScaler()

class ECNN10(torch.nn.Module):
    def __init__(self, channels, classes):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ECNN10, self).__init__()
        #self.convolutional_1 = nn.Conv2d(1, 1, kernel_size = 2, padding=1)
        #self.ReLU_1 = nn.ReLU()
        #self.convolutional_2 = nn.Conv2d(1, 1, kernel_size = 2, padding =1)
        #self.ReLU_2 = nn.ReLU()
        self.fully_connected_in1 = nn.Linear(in_features = 144, out_features = 500)
        self.fully_connected_in = nn.Linear(in_features = 500, out_features = 300)
        self.ReLU_3 = nn.ReLU()
        self.fully_connected_out = nn.Linear(in_features= 300, out_features=classes)
        self.softmax = nn.Softmax(dim = 0)
    
    def forward(self, x): 
        
        x = (x-np.mean(x))/np.std(x)
        x = torch.from_numpy(x).float() # convert maze numpy nd array into tensor
        x = flatten(x)
        #x = self.convolutional_1(x[None,None,...])
        #x = self.ReLU_1(x)
       # x = self.convolutional_2(x)
        #x = self.ReLU_2(x)
        #x = flatten(x,1)
        x = self.fully_connected_in1(x)
        x = self.ReLU_3(x)
        x = self.fully_connected_in(x)
        x = self.ReLU_3(x)
        x = self.fully_connected_out(x)
        x = self.softmax(x)
        
        return x
    
"""
 TO DO LIST:
     A few things need adding, this is a very basic CNN architecture,
     I want to make it able to change to new maze shapes, to do this I need
     to create a function that calculates the size of the output from a given
     conv layer, this will be easy, the equation is simply:
         size_i = (in_size_i - kernal_size_i + 2*padding_size_i)/(stride_i) +1
    I just need to implement this in between the conv2d layers and the fc layers
    figure out why no params present
    
"""