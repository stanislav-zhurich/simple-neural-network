import numpy as np
from abc import abstractclassmethod, ABC


class Activation(ABC):
    @abstractclassmethod
    def calculate_forward(Z: np.array):
        raise NotImplementedError
    
    @abstractclassmethod
    def calculate_backward(Z: np.array):
        raise NotImplementedError

class ReLU(Activation):
    def __init__(self):
        pass
    def calculate_forward(Z: np.array):
        A = np.maximum(0,Z)
        return 
    
    def calculate_backward(Z: np.array):
        s = Z >= 0
        return s.astype(int)
    
class Sigmoid(Activation):
    def __init__(self):
        pass
    def calculate_forward(Z: np.array):
        s = 1/(1 + np.exp(-Z))
        s = np.minimum(s, 0.99999) 
        s = np.maximum(s, 0.00001)
        return s
    def calculate_backward(Z: np.array):
        s = 1/(1 + np.exp(-Z))
        s = np.minimum(s, 0.99999) 
        s = np.maximum(s, 0.00001)
        return s * (1 - s)
    
def lookup_activation(activation: str) -> Activation:
    match activation:
        case "relu": return ReLU()
        case "sigmoid": return Sigmoid()
        case _: raise Exception("activation is not found")