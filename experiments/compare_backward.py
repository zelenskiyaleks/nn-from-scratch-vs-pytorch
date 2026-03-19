import numpy as np
import torch

from numpy_nn.layers import Linear
from numpy_nn.activations import ReLU
from numpy_nn.model import Sequential
from numpy_nn.loss import MSELoss

from torch_nn.model import TorchModel

np.random.seed(42)
torch.manual_seed(42)

#creating data

X_np = np.random.randn(5, 2)
y_np = (X_np[:, 0] + X_np[:, 1]).reshape(-1, 1)

X_torch = torch.tensor(X_np, dtype = torch.float32)
y_torch = torch.tensor(y_np, dtype = torch.float32)

# numpy model
np_model = Sequential([
    Linear(2, 10),
    ReLU(),
    Linear(10, 1)
])
loss_fn = MSELoss()

# torch model 
torch_model = TorchModel()

# copying the initial weights of Numpy model into the Torch model
#layer 1
torch_model.linear1.weight.data = torch.tensor(
    np_model.layers[0].W.T, dtype=torch.float32
)
torch_model.linear1.bias.data = torch.tensor(
    np_model.layers[0].b.flatten(), dtype=torch.float32
)

# layer 2
torch_model.linear2.weight.data = torch.tensor(
    np_model.layers[2].W.T, dtype=torch.float32
)
torch_model.linear2.bias.data = torch.tensor(
    np_model.layers[2].b.flatten(), dtype=torch.float32
)

# numpy backward

np_out = np_model.forward(X_np)
np_loss = loss_fn.forward(np_out, y_np)
grad = loss_fn.backward()
np_model.backward(grad)

np_dW1 = np_model.layers[0].dW
np_dW2 = np_model.layers[2].dW

# torch backward

torch_out = torch_model(X_torch)
loss = ((torch_out - y_torch)**2).mean()

loss.backward()
torch_dW1 = torch_model.linear1.weight.grad.numpy()
torch_dW2 = torch_model.linear2.weight.grad.numpy()


# comparison of results 

print("Layer1 diff:", np.abs(np_dW1 - torch_dW1.T).mean())
print("Layer2 diff:", np.abs(np_dW2 - torch_dW2.T).mean())