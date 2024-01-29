from simplenet.CostFunction import *
from simplenet.Layer import *
import numpy as np
import copy

class SimpleNet:

    __default_configuration = {
        "keep_prob": 1.0,
        "learning_rate": 0.01,
        "cost_function": "crossentropy",
        "iteration_num": 2500,
        "random_seed": 3,
        "lambda": None,
        "metric": "accuracy"
    }

    def __init__(self, layers, user_configuration: dict) -> None:
        self.layers = layers
        self.configuration = self.__default_configuration.copy()
        self.configuration.update(user_configuration)
        np.random.seed(3)

    def train(self, X_test: np.array, Y_test: np.array):
        iterations = self.configuration["iteration_num"]
        cost_function_str = self.configuration["cost_function"]
        _layers = self.layers.copy()
        _layers[:0] = [Layer(X_test.shape[0])]

        cache = self.__initialize_cache(_layers)
        cache["A0"] = X_test
        for i in range(0, iterations):
            AL, cache = self.__forward_propagation(_layers, cache, self.configuration)
            #print(params)
            gradients = self.__back_propagation(Y_test, _layers, cache, self.configuration)
            
            cache = self.__update_params(cache, _layers, gradients, self.configuration)
            cost_function = CostFunction.lookup_cost_function(cost_function_str)
            cost = cost_function.calculate(Y_test, AL, _layers, cache, self.configuration)
            if i % 100 == 0:
                print(i, cost)
            #print(AL)
        self.cache = cache
        return cache
    
    def predict(self, X):
        _layers = copy.deepcopy(self.layers)
        _layers[:0] = [Layer(X.shape[0])]
        _cache = copy.deepcopy(self.cache)
        A_prev = X
        for i in range(1, len(_layers)):
            W = _cache["W" + str(i)]
            b = _cache["b" + str(i)]
            Z = np.dot(W, A_prev) + b
            A = _layers[i].activation.calculate_forward(Z)
            probs = A > 0.5
            A_prev = A
        return probs.astype(int)
    
    def __initialize_cache(self, layers):
        cache = {}
        for i in range(1, len(layers)):
            cache["W" + str(i)] = np.random.randn(layers[i].node_num, layers[i - 1].node_num) * np.sqrt(2/layers[i-1].node_num) 
            cache["b" + str(i)] = np.zeros((layers[i].node_num, 1)) 
        return cache
    
    def __update_params(self, cache, layers, gradients, configuration):
        learning_rate = configuration["learning_rate"]
        for i in range(1, len(layers)):
            cache["W" + str(i)] = cache["W" + str(i)] - learning_rate * gradients["dW" + str(i)]
            cache["b" + str(i)] = cache["b" + str(i)] - learning_rate * gradients["db" + str(i)]
        return cache
    
    def __back_propagation(self, Y, layers, cache, configuration):

        keep_prob = configuration["keep_prob"]
        lambd = configuration["lambda"]

        gradients = {}
        m = Y.shape[1]
        l = len(layers)
        AL = cache["A" + str(l - 1)]
        dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        for i in reversed(range(1, len(layers))):
            Z = cache["Z" + str(i)]
            W = cache["W" + str(i)]
            A_prev = cache["A" + str(i - 1)]

            dZ = dA * layers[i].activation.calculate_backward(Z)

            dW = 1/m * np.dot(dZ,  A_prev.T)
            
            if lambd is not None:
                dW = dW + lambd/m * W
            db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
            dA = np.dot(W.T, dZ)
            #not for X
            if i > 1:
                D = cache["D" + str(i - 1)]
                dA = np.multiply(dA, D) / keep_prob
            gradients["dW" + str(i)] = dW
            gradients["db" + str(i)] = db
        
        return gradients
    
    def __forward_propagation(self, layers, cache, configuration):
        keep_prob = configuration["keep_prob"]
        A_prev = cache["A0"]
        dim_len = len(layers)
        for i in range(1, dim_len):
            W = cache["W" + (str(i))]
            b = cache["b" + (str(i))]
            Z = np.dot(W, A_prev) + b

            A = layers[i].activation.calculate_forward(Z)
                
            #Add dropout for all but output layers
            if i < (dim_len - 1):
                dropout_matrix = np.random.rand(A.shape[0], A.shape[1])
                dropout_matrix = (dropout_matrix < keep_prob).astype(int)
                A = np.multiply(A, dropout_matrix) 
                A = A / keep_prob
                cache["D" + str(i)] = dropout_matrix
            A_prev = A.copy()
            cache["A" + str(i)] = A
            cache["Z" + str(i)] = Z
            
        return (A_prev, cache)
