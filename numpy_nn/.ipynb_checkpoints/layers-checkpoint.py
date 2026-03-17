import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features)
        self.b = np.zeros((1, out_features))

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.dW = self.x.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T  