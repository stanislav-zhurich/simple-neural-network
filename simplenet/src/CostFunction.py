import numpy as np
from abc import abstractclassmethod, ABC

class CostFunction(ABC):
    
    def __init__(self):
        pass

    @abstractclassmethod
    def calculate(self, Y: np.array, Y_hat: np.array) -> np.array:
        raise NotImplementedError
    
class CrossEntropyCostFunction(CostFunction):

    def calculate(self,  Y: np.array, Y_hat: np.array) -> np.array:
        m = Y.shape[1]
        cost = (1./m) * (- np.dot(Y,np.log(Y_hat).T) - np.dot(1 - Y, np.log(1 - Y_hat).T))
        return cost
    
def lookup_cost_function(str_cost_function: str) -> CostFunction:
    match str:
        case "crossentropy": return CrossEntropyCostFunction()
        case _: raise Exception("cost function is not found")
    

