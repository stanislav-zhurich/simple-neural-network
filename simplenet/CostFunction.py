import numpy as np
from abc import abstractmethod, ABC
from simplenet.Layer import *

class CostFunction(ABC):
    
    def __init__(self):
        pass

   
    def calculate(self, Y: np.array, Y_hat: np.array, layers, parameters: dict, configuration: dict) -> float:
        cost = self._calculate_cost(Y, Y_hat)
        lambd = configuration["lambda"]
        m = Y.shape[1]
        if lambd is not None:
            weight_sum = 0
            for i in range(1, len(layers)):
                W = parameters["W" + str(1)]
                weight_sum += np.sum(np.square(W))
            cost = cost + 2 * lambd/m * weight_sum
        return cost
        
    
    @abstractmethod
    def _calculate_cost(self, Y: np.array, Y_hat: np.array) -> float:
        raise NotImplementedError
    
    @staticmethod
    def lookup_cost_function(str_cost_function: str):
        match str_cost_function:
            case "crossentropy": return CrossEntropyCostFunction()
            case _: raise Exception("cost function is not found")
    
class CrossEntropyCostFunction(CostFunction):

    def _calculate_cost(self,  Y: np.array, Y_hat: np.array) -> np.array:
        m = Y.shape[1]
        cost = (1./m) * (- np.dot(Y,np.log(Y_hat).T) - np.dot(1 - Y, np.log(1 - Y_hat).T))
        return cost
    
    

