
from abc import ABC,abstractmethod
import numpy as np
class BaseLayer(ABC):
    pass
class BaseLayer(ABC):
    def __init__(self) -> None:
        self.trainable=False    
        self.weights=None
        self._input_tensor=None

        pass
    def __call__(self,l:BaseLayer)->BaseLayer:
        pass
    @abstractmethod
    def forward(self,input_tensor):
        self._input_tensor=np.array(input_tensor)
        pass
    
    @abstractmethod
    def backward(self,error_tensor): 
        if  self._input_tensor is None :
            raise Exception('No forward called!')
        pass
    
 
        
