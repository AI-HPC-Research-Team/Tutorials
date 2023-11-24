# Deep Learning Tutorial

## [Training and Inference, Step by Step](https://pytorch.org/tutorials/recipes/recipes_index.html)

### 1. Loading data
``` python
# pip install torchaudio
import torch
import torchaudio
torchaudio.datasets.YESNO(
     root='./',
     url='http://www.openslr.org/resources/1/waves_yesno.tar.gz',
     folder_in_archive='waves_yesno',
     download=True)

# A data point in ``yesno`` is a tuple (waveform, sample_rate, labels) where labels
# is a list of integers with 1 for yes and 0 for no.
yesno_data = torchaudio.datasets.YESNO('./', download=True)

# Pick data point number 3 to see an example of the the ``yesno_data``:
n = 3
waveform, sample_rate, labels = yesno_data[n]
print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))
data_loader = torch.utils.data.DataLoader(yesno_data,
                                          batch_size=1,
                                          shuffle=True)
for data in data_loader:
  print("Data: ", data)
  print("Waveform: {}\nSample rate: {}\nLabels: {}".format(data[0], data[1], data[2]))
  break

```
### 2. Defining a neural network and training
``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through ``fc1``
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output

my_nn = Net()
optimizer = optim.SGD(my_nn.parameters(), lr=0.001, momentum=0.9)
my_nn.train()

```

### 3. Saving and loading models
``` python
# Specify a path
PATH = "state_dict_model.pt"

# Save
torch.save(my_nn.state_dict(), PATH)

# Load
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
# Equates to one random 28x28 image
random_data = torch.rand((1, 1, 28, 28))
result = my_nn(random_data)
print (result)
```

## [MLP examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn)
In this example we use the nn package to implement our polynomial model network:
``` python
# -*- coding: utf-8 -*-
import torch
import math


# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
# (3,), for this case, broadcasting semantics will apply to obtain a tensor
# of shape (2000, 3) 

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. The Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# The Flatten layer flatens the output of the linear layer to a 1D tensor,
# to match the shape of `y`.
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):

    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(xx)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# You can access the first layer of `model` like accessing the first item of a list
linear_layer = model[0]

# For linear layer, its parameters are stored as `weight` and `bias`.
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
```

## [Hyperparameter finetuning](https://blog.roboflow.com/what-is-hyperparameter-tuning/)

### Common hyperparameters
- **Learning Rate**: The learning rate determines the step size at which the model updates its parameters during training. It influences the convergence speed and stability of the training process. Finding an optimal learning rate is crucial to prevent underfitting or overfitting.
- **Batch Size**: The batch size determines the number of samples processed in each iteration during model training. It affects the training dynamics, memory requirements, and generalization ability of the model. Choosing an appropriate batch size depends on the available computational resources and characteristics of the dataset on which the model will be trained.
- **Network Architecture**: The network architecture defines the structure and connectivity of neural network layers. It includes the number of layers, the type of layers (convolutional, pooling, fully connected, etc.), and their configuration. Selecting an appropriate network architecture depends on the complexity of the task and the available computational resources.
- **Kernel Size**: In convolutional neural networks (CNNs), the kernel size determines the receptive field size used for feature extraction. It affects the level of detail and spatial information captured by the model. Tuning the kernel size is essential to balance local and global feature representation.
- **Dropout Rate**: Dropout is a regularization technique that randomly drops a fraction of the neural network units during training. The dropout rate determines the probability of “dropping” each unit. Dropout rate helps prevent overfitting by encouraging the model to learn more robust features and reduces the dependence on specific units.
- **Activation Functions**: Activation functions introduce non-linearity to the model and determine the output of a neural network node. Common activation functions include ReLU (Rectified Linear Unit), sigmoid, and tanh. Choosing an appropriate activation function can impact the model's capacity to capture complex relationships and its training stability.
- **Data Augmentation**: Data augmentation techniques, such as rotation, scaling, and flipping, enhance the diversity and variability of the training dataset. Hyperparameters related to data augmentation, such as rotation angle range, scaling factor range, and flipping probability, influence the augmentation process and can improve the model's ability to generalize to unseen data.
- **Optimization Algorithm**: The choice of optimization algorithm influences the model's convergence speed and stability during training. Common optimization algorithms include stochastic gradient descent (SGD), ADAM, and RMSprop. Hyperparameters related to the optimization algorithm, such as momentum, learning rate decay, and weight decay, can significantly impact the training process.

### Finetuning skills
- Manual hyperparameter tuning
    - Manual hyperparameter tuning is a method of adjusting the hyperparameters of a machine learning model through manual experimentation. It involves iteratively modifying the hyperparameters and evaluating the model's performance until satisfactory results are achieved. 
    - Although this can be a time-consuming process, manual tuning provides the flexibility to explore various hyperparameter combinations and adapt them to specific datasets and tasks.

- [Automatic hyperparameter tuning](https://www.semanticscholar.org/paper/On-Hyperparameter-Optimization-of-Machine-Learning-Yang-Shami/2e5d2f2dc01b150dffc163a9f457848e9b5b5c38)
    - Grid search
    - Random search 
    - Bayesian optimization
    - Particle swarm optimisation
    - Tree-Structured Parzen Estimator
    - ...


