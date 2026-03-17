import numpy as np
from layers import Linear
from activations import ReLU
from loss import MSELoss
from model import Sequential

def train():
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1]).reshape(-1, 1)

    model = Sequential([
        Linear(2, 10),
        ReLU(),
        Linear(10, 1)
    ])

    loss_fn = MSELoss()
    lr = 0.01

    for epoch in range(200):
        out = model.forward(X)
        loss = loss_fn.forward(out, y)

        grad = loss_fn.backward()
        model.backward(grad)

        for layer in model.parameters():
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")