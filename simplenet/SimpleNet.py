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
                AL, caches = self.__forward_propagation(X_j, parameters)
                gradients = self.__back_propagation(Y_j, AL, caches)
                parameters = self.__update_params(parameters, gradients)
                cost_function = CostFunction.lookup_cost_function(cost_function_str)
                cost_sum = cost_function.calculate(Y_j, AL, self.layers, parameters, self.configuration)
            cost = cost_sum/X_test.shape[1]
            if i % 100 == 0:
                print(i, cost)
        self.parameters = parameters
        return parameters
    
    def __train_mini_batch(self, batched_X, batched_Y, parameters, layers):
        cost_function_str = self.configuration["cost_function"]
        AL, caches = self.__forward_propagation(batched_X, parameters, layers)
        gradients = self.__back_propagation(batched_Y, AL, caches, layers)
        parameters = self.__update_params(parameters, gradients, layers)
        cost_function = CostFunction.lookup_cost_function(cost_function_str)
        cost = cost_function.calculate(batched_Y, AL, self.layers, parameters, self.configuration)
        return parameters, cost
    
    def train(self, X_test: np.array, Y_test: np.array):
        epoch = self.configuration["epoch"]
        batch_size = self.configuration["batch_size"]
        batch_iterations = int(X_test.shape[1]/batch_size)
        remains = X_test.shape[1] - batch_size * batch_iterations
        layers = copy.deepcopy(self.layers)
        layers[:0] = [Layer(X_test.shape[0])]

        parameters = self.__initialize_parameters(layers)
        costs = []

        for i in range(0, epoch):
            cost_sum = 0
            shuffled_X, shuffled_Y = self.__shuffle_dataset(X_test, Y_test)
            for j in range(batch_iterations):
                batched_X = shuffled_X[:, batch_size * j: batch_size * j + batch_size ]
                batched_Y = shuffled_Y[:, batch_size * j: batch_size * j + batch_size ]
                parameters, cost = self.__train_mini_batch(batched_X, batched_Y, parameters, layers)
                cost_sum += cost
            if remains > 0:
                batched_X = shuffled_X[:, -remains: ]
                batched_Y = shuffled_Y[:, -remains: ]
                parameters, cost = self.__train_mini_batch(batched_X, batched_Y, parameters, layers)
                cost_sum += cost
            cost = cost_sum/X_test.shape[1]
            costs.append(np.squeeze(cost).item())
            if i % 100 == 0:
                print(i, cost)
            self.parameters = parameters
        return costs

    
    def predict(self, X):
        layers = self.layers
        layers = copy.deepcopy(self.layers)
        layers[:0] = [Layer(X.shape[0])]
        A_prev = X
        for i in range(1, len(layers)):
            W = self.parameters["W" + str(i)]
            b = self.parameters["b" + str(i)]
            Z = np.dot(W, A_prev) + b
            A = layers[i].activation.calculate_forward(Z)
            probs = A > 0.5
            A_prev = A
        return probs.astype(int)
    
    def __initialize_parameters(self, layers):
        parameters = {}
        for i in range(1, len(layers)):
            parameters["W" + str(i)] = np.random.randn(layers[i].node_num, layers[i - 1].node_num) * np.sqrt(2/layers[i-1].node_num) 
            parameters["b" + str(i)] = np.zeros((layers[i].node_num, 1)) 
        return parameters
    
    def __update_params(self, parameters, gradients, layers):
        learning_rate = self.configuration["learning_rate"]
        for i in range(1, len(layers)):
            parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * gradients["dW" + str(i)]
            parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * gradients["db" + str(i)]
        return parameters
    
    def __back_propagation(self, Y, AL, caches, layers):

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
    
    
    def __forward_propagation(self, X_test, parameters, layers):
        keep_prob = self.configuration["keep_prob"]
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
    
    def __shuffle_dataset(self, X, Y):
        permutation = list(np.random.permutation(X.shape[1]))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        return shuffled_X, shuffled_Y
