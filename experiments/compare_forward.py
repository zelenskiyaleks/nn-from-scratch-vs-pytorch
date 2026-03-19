import numpy as np
import torch

from numpy_nn.layers import Linear
from numpy_nn.activations import ReLU
from numpy_nn.model import Sequential

from torch_nn.model import TorchModel

np.random.seed(42)
torch.manual_seed(42)

#creating data

X_np = np.random.randn(5, 2)
X_torch = torch.tensor(X_np, dtype = torch.float32)

# numpy model
np_model = Sequential([
    Linear(2, 10),
    ReLU(),
    Linear(10, 1)
])

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


# forward pass

np_out = np_model.forward(X_np)
torch_out = torch_model(X_torch).detach().numpy()


print("NumPy output:", np_out)
print("Torch output:", torch_out)

print("Difference:", np.abs(np_out - torch_out).mean())