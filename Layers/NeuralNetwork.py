#import sys;
#import os;
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.dirname(SCRIPT_DIR))

from FullyConnected import FullyConnected
from Base import BaseLayer
from ReLU import ReLU
from SoftMax import SoftMax
from Optimization.Optimizers import Sgd
from Base import BaseLayer
import copy

class NeuralNetwork():

    def __init__(self,optimizer):

        self.optimizer = optimizer
        self.loss=[]
        self.layers=[]
        self.data_layer=None
        self.loss_layer=None
        self.input_tensor= None
        self.label_tensor = None#.next()

        pass

    def forward(self,):
        data  = self.data_layer.next()
        self.input_tensor=data[0]
        self.label_tensor=data[1]
        o = self.input_tensor
        for l in self.layers:
            o = l.forward(o)

        o = self.loss_layer.forward(o,self.label_tensor)

        return o
        pass

    def backward(self,):
        o = self.loss_layer.backward(self.label_tensor)
        
        for i in range(len(self.layers)-1,-1,-1):
            o = self.layers[i].backward(o)
        
        return o

        pass

    def append_layer(self,layer:BaseLayer):
        if layer.trainable==True:
            op = copy.deepcopy(self.optimizer)

            layer.optimizer = op

            pass

        self.layers.append(layer)
        pass

    def train(self,iterations):
        
        for i in range(iterations):
            o = self.forward()
            self.loss.append(o)

            self.backward()
            


        pass
    def test(self,input_tensor):
       
        o = input_tensor
        for l in self.layers:
            o = l.forward(o)
        return o
        

    pass

