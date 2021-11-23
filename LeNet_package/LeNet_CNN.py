"""
Package to build the LeNet architecture for a CNN, 
This is just a basic example of a LeNet.
"""

#import the important PyTorch NN feaatures

from torch.nn import Module # base class of neural nets
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten


class LeNetCNN(Module):
    def __innit__(self, input_channels = 3, classes = 4):
        super(LeNetCNN, self).__innit__()
        self.convolutional_1 = Conv2d(in_channels = input_channels, out_channels = 20, kernal_size = (5,5))
        self.max_pool_1 = MaxPool2d(kernal_size = (2,2), stride = (2,2))
        self.ReLU_1 = ReLU()
        self.convolutional_2 = Conv2d(in_channels = 20, out_channels = 50, kernal_size = (5,5))
        self.ReLU_2 = ReLU()
        self.max_pool_2 = MaxPool2d(kernal_size = (2,2), stride = (2,2))
        self.fully_connected_in = Linear(in_features = 800, out_features = 500)
        self.ReLU_3 = ReLU()
        self.fully_connected_out = Linear(in_features= 500, out_features=classes)
        self.log_softmax = LogSoftmax(dim = 1)
    
    def forward(self, x):
        
        x = self.convolutional_1(x)
        x = self.max_pool_1(x)
        x = self.ReLU_1(x)
        x = self.convolutional_2(x)
        x = self.ReLU_2(x)
        x = self.max_pool_2(x)
        x = flatten(x,1)
        x = self.fully_connected_in(x)
        x = self.ReLU_3(x)
        x = self.fully_connected_out(x)
        x = self.log_softmax(x)
        
        return x
    
"""
 TO DO LIST:
     A few things need adding, this is a very basic CNN architecture,
     I want to make it able to change to new maze shapes, to do this I need
     to create a function that calculates the size of the output from a given
     conv layer, this will be easy, the equation is simply:
         size_i = (in_size_i - kernal_size_i + 2*padding_size_i)/(stride_i) +1
    I just need to implement this in between the conv2d layers and the fc layers
"""