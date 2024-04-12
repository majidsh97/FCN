import sys
sys.path.append('Layers')
from Base import BaseLayer
import numpy as np
from ReLU import ReLU


class FullyConnected(BaseLayer):
    def __init__(self,input_size, output_size) -> None:
        super().__init__()
        self.trainable=True
        self._optimizer=None
        self.weights=np.random.uniform(0,1,(input_size+1,output_size )) 
        #rint(self.weights.shape)
        self.gradient_weights=None
        self.__forward_out=None



    def forward(self, input_tensor):
        super().forward(input_tensor)
        
        inp =np.concatenate([input_tensor,np.ones((input_tensor.shape[0],1))],1) #wx +b , w*x
        self._input_tensor = inp
        self.__forward_out = np.matmul( inp , self.weights )  # (m x inp) X (inp x out ) -> (m x out) + (m *x out)
        #print(o.shape)
        return  self.__forward_out

    def backward(self,error_tensor): # (m x out) (mxinp) -> (inp x out)
        super().backward(error_tensor)
        xt=np.transpose( self._input_tensor)


        new_error = np.matmul( xt,error_tensor)
        self.gradient_weights=new_error


        if self._optimizer is not None:
            self.weights =  self._optimizer.calculate_update(self.weights,new_error)

        
        return np.matmul( error_tensor,self.weights[:-1,:].T) # m x out out x inp -> m * inp

    @property #-> self.optimizer=2 raise error
    def optimizer(self,):
        return self._optimizer

    @optimizer.setter # self.optimizer = 2 no error
    def optimizer(self,value):
        self._optimizer=value





