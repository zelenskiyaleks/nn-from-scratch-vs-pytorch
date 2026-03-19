import numpy as np

from numpy_nn.layers import Linear
from numpy_nn.activations import ReLU
from numpy_nn.model import Sequential
from numpy_nn.loss import MSELoss


np.random.seed(42)

X = np.random.randn(5, 2)
y = (X[:, 0] + X[:, 1]).reshape(-1, 1)

model = Sequential([
    Linear(2, 3),
    ReLU(),
    Linear(3, 1)
])

loss_fn = MSELoss()

# forward
out = model.forward(X)
loss = loss_fn.forward(out, y)

# backward
grad = loss_fn.backward()
model.backward(grad)

# analytical gradient
dW = model.layers[0].dW

# numerical gradient
epsilon = 1e-5
num_dW = np.zeros_like(dW)

for i in range(dW.shape[0]):
    for j in range(dW.shape[1]):
        original = model.layers[0].W[i, j]

        model.layers[0].W[i, j] = original + epsilon
        loss_plus = loss_fn.forward(model.forward(X), y)

        model.layers[0].W[i, j] = original - epsilon
        loss_minus = loss_fn.forward(model.forward(X), y)

        num_dW[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        model.layers[0].W[i, j] = original

print("Gradient diff:", np.abs(dW - num_dW).mean())