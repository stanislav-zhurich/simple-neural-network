from Activation import Activation, lookup_activation
from CostFunction import CostFunction, lookup_cost_function
import numpy as np

class Layer:
    def __init__(self, node_num, activation: Activation):
        self.node_num = node_num
        self.activation = lookup_activation(activation)

class SimpleNet:
    def __init__(self, layers: list(Layer)) -> None:
        self.layers = layers

    def train(self, X_test: np.array, Y_test: np.array):
        np.random.seed(3)
        keep_prob = 0.9
        lambd = None
        learning_rate = 0.01
        iterations = 2500
        layers = layers.copy()
        layers[:0] = [Layer(X.shape[0])]

        params = self.__initialize_parameters(layers)
        params["A0"] = X
        for i in range(0, iterations):
            AL, params = self.__forward_propagation(params, keep_prob)
            #print(params)
            gradients = self.__back_propagation(Y, params, lambd = lambd, keep_prob = keep_prob)
            
            params = self.__update_params(params, gradients, learning_rate)
            cost_function = lookup_cost_function()
            cost = cost_function(Y_test, AL, layers, params, lambd = lambd)
            if i % 100 == 0:
                print(i, cost)
            #print(AL)
        return params
    
    def __initialize_parameters(self):
        parameters = {}
        for i in range(1, len(self.layers)):
            parameters["W" + str(i)] = np.random.randn(self.layers[i].node_num, self.layers[i - 1].node_num) * np.sqrt(2/self.layers[i-1].node_num) 
            parameters["b" + str(i)] = np.zeros((self.layers[i].node_num, 1)) 
        return parameters
    
    def __update_params(self, params, gradients, learning_rate):
        updated_params = params.copy()
        for i in range(1, len(self.layers)):
            updated_params["W" + str(i)] = updated_params["W" + str(i)] - learning_rate * gradients["dW" + str(i)]
            updated_params["b" + str(i)] = updated_params["b" + str(i)] - learning_rate * gradients["db" + str(i)]
        return updated_params
    
    def __back_propagation(self, Y, params, lambd = None, keep_prob = 1.0):
        gradients = {}
        m = Y.shape[1]
        l = len(self.layers)
        AL = params["A" + str(l - 1)]
        dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        for i in reversed(range(1, len(self.layers))):
            Z = params["Z" + str(i)]
            W = params["W" + str(i)]
            A_prev = params["A" + str(i - 1)]

            dZ = dA *  self.layers[i].calculate_backward(Z)

            dW = 1/m * np.dot(dZ,  A_prev.T)
            
            if lambd is not None:
                dW = dW + lambd/m * W
            db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
            dA = np.dot(W.T, dZ)
            #not for X
            if i > 1:
                D = params["D" + str(i - 1)]
                dA = np.multiply(dA, D) / keep_prob
            gradients["dW" + str(i)] = dW
            gradients["db" + str(i)] = db
        
        return gradients
    
    def __forward_propagation(self, params, keep_prob = 1.0):
        updated_params = params.copy()
        A_prev = updated_params["A0"]
        dim_len = len(self.layers)
        for i in range(1, dim_len):
            W = params["W" + (str(i))]
            b = params["b" + (str(i))]
            Z = np.dot(W, A_prev) + b

            A = self.layers[i].calculate_forward(Z)
                
            #Add dropout for all but output layers
            if i < (dim_len - 1):
                dropout_matrix = np.random.rand(A.shape[0], A.shape[1])
                dropout_matrix = (dropout_matrix < keep_prob).astype(int)
                A = np.multiply(A, dropout_matrix) 
                A = A / keep_prob
                updated_params["D" + str(i)] = dropout_matrix
            A_prev = A.copy()
            updated_params["A" + str(i)] = A
            updated_params["Z" + str(i)] = Z
            
        return (A_prev, updated_params)
