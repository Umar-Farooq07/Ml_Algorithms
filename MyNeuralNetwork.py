import numpy as np


class Layer:

    def __init__(self,input_size: int, output_size: int ):

        self.parameters = np.random.rand(input_size,output_size)
        self.bias = np.random.rand(output_size)

        

    def forward(self, x):
        ...

    def backward(self, grad):
        ...


layer = Layer(5,6)
print(layer.parameters)