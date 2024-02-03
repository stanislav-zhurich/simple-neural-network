from simplenet.CostFunction import *
from simplenet.Layer import *
import numpy as np
import copy

class SimpleNet:

    __default_configuration = {
        "keep_prob": 1.0,
        "learning_rate": 0.01,
        "cost_function": "crossentropy",
        "epoch": 2500,
        "batch_size": 1,
        "random_seed": 3,
        "lambda": None,
        "metric": "accuracy"
    }

    def __init__(self, layers, user_configuration: dict) -> None:
        self.layers = copy.deepcopy(layers)
        self.configuration = self.__default_configuration.copy()
        self.configuration.update(user_configuration)
        np.random.seed(3)


    def train_sgd(self, X_test: np.array, Y_test: np.array):
        epoch = self.configuration["epoch"]
        cost_function_str = self.configuration["cost_function"]
        self.layers[:0] = [Layer(X_test.shape[0])]

        parameters = self.__initialize_parameters()
        for i in range(0, epoch):
            cost_sum = 0
            for j in range(X_test.shape[1]):
                X_j = np.expand_dims(X_test[:, j], axis = 1)
                Y_j = np.expand_dims(Y_test[:, j], axis = 1)
                AL, caches = self._forward_propagation(X_j, parameters)
                gradients = self.__back_propagation(Y_j, AL, caches)
                parameters = self.__update_params(parameters, gradients)
                cost_function = CostFunction.lookup_cost_function(cost_function_str)
                cost_sum = cost_function.calculate(Y_j, AL, self.layers, parameters, self.configuration)
            cost = cost_sum/X_test.shape[1]
            if i % 100 == 0:
                print(i, cost)
        self.parameters = parameters
        return parameters

    def train(self, X_test: np.array, Y_test: np.array):
        epoch = self.configuration["epoch"]
        cost_function_str = self.configuration["cost_function"]
        self.layers[:0] = [Layer(X_test.shape[0])]

        parameters = self.__initialize_parameters()
        for i in range(0, epoch):
            AL, caches = self._forward_propagation(X_test, parameters)
            gradients = self.__back_propagation(Y_test, AL, caches)
            parameters = self.__update_params(parameters, gradients)
            cost_function = CostFunction.lookup_cost_function(cost_function_str)
            cost = cost_function.calculate(Y_test, AL, self.layers, parameters, self.configuration)
            if i % 100 == 0:
                print(i, cost)
            #print(AL)
        self.parameters = parameters
        return parameters
    
    def predict(self, X):
        layers = self.layers
        A_prev = X
        for i in range(1, len(layers)):
            W = self.parameters["W" + str(i)]
            b = self.parameters["b" + str(i)]
            Z = np.dot(W, A_prev) + b
            A = layers[i].activation.calculate_forward(Z)
            probs = A > 0.5
            A_prev = A
        return probs.astype(int)
    
    def __initialize_parameters(self):
        parameters = {}
        layers = self.layers
        for i in range(1, len(layers)):
            parameters["W" + str(i)] = np.random.randn(layers[i].node_num, layers[i - 1].node_num) * np.sqrt(2/layers[i-1].node_num) 
            parameters["b" + str(i)] = np.zeros((layers[i].node_num, 1)) 
        return parameters
    
    def __update_params(self, parameters, gradients):
        learning_rate = self.configuration["learning_rate"]
        for i in range(1, len(self.layers)):
            parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * gradients["dW" + str(i)]
            parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * gradients["db" + str(i)]
        return parameters
    
    def __back_propagation(self, Y, AL, caches):

        layers = self.layers
        keep_prob = self.configuration["keep_prob"]
        lambd = self.configuration["lambda"]

        gradients = {}
        m = Y.shape[1]

        dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        for i in reversed(range(1, len(layers))):
            (A_prev, W, b, Z, D) = caches[i - 1]

            if i < len(layers) - 1:
                dA = np.multiply(dA, D) / keep_prob

            activation = layers[i].activation

            if activation is not None:
                dZ = dA * activation.calculate_backward(Z)
                dW = 1/m * np.dot(dZ,  A_prev.T)
        
                if lambd is not None:
                    dW = dW + lambd/m * W
                db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
                dA = np.dot(W.T, dZ)

                gradients["dW" + str(i)] = dW
                gradients["db" + str(i)] = db
          
        return gradients
    
    
    def _forward_propagation(self, X_test, parameters):
        keep_prob = self.configuration["keep_prob"]
        layers = self.layers
        A_prev = X_test
        dim_len = len(layers)
        caches = []
        for i in range(1, dim_len):
            W = parameters["W" + (str(i))]
            b = parameters["b" + (str(i))]
            Z = np.dot(W, A_prev) + b

            activation = layers[i].activation

            if activation is not None:
                A = activation.calculate_forward(Z)
                dropout_matrix = None
                if i < (dim_len - 1):
                    dropout_matrix = np.random.rand(A.shape[0], A.shape[1])
                    dropout_matrix = (dropout_matrix < keep_prob).astype(int)
                    A = np.multiply(A, dropout_matrix) 
                    A = A / keep_prob
                caches.append((A_prev, W, b, Z, dropout_matrix))
            A_prev = A.copy()
        return A_prev, caches
