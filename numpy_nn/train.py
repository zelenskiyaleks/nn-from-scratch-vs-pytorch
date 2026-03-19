import numpy as np
from layers import Linear
from activations import ReLU
from loss import MSELoss
from model import Sequential
import matplotlib.pyplot as plt
import os

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
    losses = []

    for epoch in range(200):
        out = model.forward(X)
        loss = loss_fn.forward(out, y)

        losses.append(loss)

        grad = loss_fn.backward()
        model.backward(grad)

        for layer in model.parameters():
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.grid(True)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    save_path = os.path.join(os.path.dirname(__file__), "..", "results", "loss_curve.png")
    plt.savefig(save_path)
    plt.show()