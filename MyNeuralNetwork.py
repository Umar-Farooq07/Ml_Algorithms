import numpy as np


class Layer:

    def __init__(self,input_size: int, output_size: int ):

        self.weights = np.random.rand(input_size,output_size)
        self.bias = np.random.rand(output_size)

    def forward(self,input):
        self.input = input
        output= input @ self.weights+ self.bias
        return output
    

    def backward(self, output_grad, lr):

        dw = self.input.T @ output_grad

        db = output_grad.sum(axis=0)

        grad_input = output_grad @ self.weights.T

        self.weights-= lr*dw

        self.bias -= lr*db

        return grad_input


