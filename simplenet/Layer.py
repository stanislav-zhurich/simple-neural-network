from simplenet.Activation import *

class Layer:

    def __init__(self, node_num: int, activation: str = None) -> None:
        self.node_num = node_num
        self.activation = Activation.lookup_activation(activation)