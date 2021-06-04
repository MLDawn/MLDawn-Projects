import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import os

# Implementing relu()
def relu(z):
    if z > 0:
        return z
    else:
        return 0

# Define linewidth and fontsize to control the aesthetics of the plots easily
linewidth = 4
fontsize = 20

# Define a range of values for the inputs of relu(z)
z_range = np.arange(-5,5, 0.01)

plt.figure(figsize=(16,9))
# For each z in x_range compute relu(z)
y_relu = [relu(z) for z in z_range]
plt.plot(z_range, y_relu, c='b', linewidth= linewidth, label='Relu(z)')
plt.ylim(-5, 5)
plt.xlim(-5, 5)
plt.grid()
plt.legend(fontsize=fontsize, loc=2)
plt.show()

def grad_relu(z):
    if z > 0:
        return 1
    else:
        return 0


### The gradients of relu
y_relu = [relu(z) for z in z_range]
grad_y_relu = [grad_relu(z) for z in z_range]

plt.figure(figsize=(16, 9))
# The relu
plt.subplot(1,2,1)
plt.plot(z_range, y_relu, c='b',linewidth= linewidth, label='Relu(z)')
plt.legend(fontsize=fontsize,loc=2)
plt.grid()

### The gradients of relu
plt.subplot(1,2,2)
plt.plot(z_range, grad_y_relu, c='r',linewidth= linewidth, label='d Relu(z)/dz')
plt.legend(fontsize=fontsize,loc=2)
plt.grid()
plt.show()


# Demonstrating the flexibility of relu: relu(z),relu(-z),-relu(z),-relu(-z)
z_range = np.arange(-5,5, 0.01)

plt.figure(figsize=(16,9))
plt.suptitle('The Flexibility of Relu(z)', fontsize=fontsize)

plt.subplot(2,2,1)
y_relu = [relu(z) for z in z_range]
plt.plot(z_range, y_relu, c='b', linewidth= linewidth, label='Relu(z)')
plt.ylim(-5,5)
plt.xlim(-5,5)
plt.grid()
plt.legend(fontsize=fontsize, loc=2)

plt.subplot(2,2,2)
y_relu = [relu(-z) for z in z_range]
plt.plot(z_range, y_relu, c='k', linewidth= linewidth,label='Relu(-z)')
plt.ylim(-5,5)
plt.xlim(-5,5)
plt.legend(fontsize=fontsize,loc=1)
plt.grid()

plt.subplot(2,2,3)
y_relu = [-relu(z) for z in z_range]
plt.plot(z_range, y_relu, c='r', linewidth= linewidth,label='-Relu(z)')
plt.ylim(-5,5)
plt.xlim(-5,5)
plt.legend(fontsize=fontsize,loc=2)
plt.grid()

plt.subplot(2,2,4)
y_relu = [-relu(-z) for z in z_range]
plt.plot(z_range, y_relu, c='g', linewidth= linewidth,label='-Relu(-z)')
plt.ylim(-5,5)
plt.xlim(-5,5)
plt.legend(fontsize=fontsize,loc=1)
plt.grid()

plt.show()


# The rotation of the slope in relu
w_range = np.arange(0.5, 3.5, 0.5)
plt.figure(figsize=(16, 9))
plt.suptitle('Changing the slope of Relu(w*z) using a coefficient w', fontsize=fontsize)
for idx, w in enumerate(w_range):
    plt.subplot(2,3,idx+1)
    y_relu = [relu(w*z) for z in z_range]
    plt.plot(z_range, y_relu, c='b', linewidth=linewidth, label='w = %.2f' % w)
    plt.ylim(-1, 5)
    plt.xlim(-5, 5)
    plt.grid()
    plt.legend(fontsize=fontsize, loc=2)
plt.show()

# Shifting the relu horizontally
bias = np.arange(0.5, 3.5, 0.5)
plt.figure(figsize=(16, 9))
plt.suptitle('Shifting Relu(z+b) horizontally using a bias term b inside Relu()', fontsize=fontsize)
for idx, b in enumerate(bias):
    plt.subplot(2,3, idx+1)
    y_relu = [relu(z+b) for z in z_range]
    plt.plot(z_range, y_relu, c='b', linewidth=linewidth, label='b = %.2f' % b)
    plt.ylim(-1, 5)
    plt.xlim(-4, 4)
    plt.grid()
    plt.legend(fontsize=fontsize, loc=2)
plt.show()

# Shifting the relu vertically
bias = np.arange(0.5, 3.5, 0.5)
plt.figure(figsize=(16, 9))
plt.suptitle('Shifting Relu(z) + b vertically using a bias term b outside Relu()', fontsize=fontsize)
for idx, b in enumerate(bias):
    plt.subplot(2,3, idx+1)
    y_relu = [relu(z)+b for z in z_range]
    plt.plot(z_range, y_relu, c='b', linewidth=linewidth, label='b = %.2f' % b)
    plt.ylim(-1, 5)
    plt.xlim(-4, 4)
    plt.grid()
    plt.legend(fontsize=fontsize, loc=2)
plt.show()


# Defining the data and the ground-truth
x = torch.unsqueeze(torch.linspace(-10, 10, 300), dim=1)
y = x.pow(3)

# Setting the available device
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print("Device", device)

# Build a regression model clas
class Regressor(nn.Module):
    def __init__(self, n_hidden=2):
        super(Regressor, self).__init__()
        self.hidden = torch.nn.Linear(1, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, 1)  # output layer


    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


# number of relu() units
n_hidden = 7
# total number of epochs
n_epochs = 4000
# Building an object from the regressor class while  passing
# n_hidden and setting the model to train() mode
regressor = Regressor(n_hidden=n_hidden).train()
# Defining the optimizer
optimizer = torch.optim.SGD(regressor.parameters(), lr=0.0001)
# Defining MSE as the appropriate los function
# For regression.
loss_func = torch.nn.MSELoss()



plt.figure(figsize=(16, 9))
for epoch in range(n_epochs):
    # Put the model in training mode
    regressor.train()
    # This is there to clear the previous plot in the animation
    # After each epoch
    plt.clf()
    # input x to the regressor and receive the predicion
    y_hat = regressor(x)
    # Compute the loss between y_hat and the actual
    # Value of the ground-truth curve, y
    loss = loss_func(y_hat, y)
    # Compute the gradients w.r.t all the parameters
    loss.backward()
    # Update the parameters
    optimizer.step()
    # Zero out all the gradients before inputing the next data point
    # Into the regressor model
    optimizer.zero_grad()

    # Every 100 epoch evaluate do some plotting
    if epoch % 100 == 0:
        print('Epoch %d --- Loss %.5f' % (epoch+1, loss.data.numpy()))
        # Bbefore evaluation, put the model back to evaluation mode
        regressor.eval()
        # At this very moment of training, grab the current biases and weights
        # From the model object, namely, b_0, b_1, w_0, and w_1
        biases_0 = regressor.hidden.bias.cpu().detach().numpy()
        weights_0 = regressor.hidden.weight.squeeze(0).cpu().detach().numpy()
        biases_1 = regressor.predict.bias.cpu().detach().numpy() # This has ONLY 1 value
        weights_1 = regressor.predict.weight.squeeze(0).cpu().detach().numpy()

        # For the purpose of plotting consider the current range of
        # x as the inputs to EACH relu() individualy
        data = x.detach().numpy()
        # This will hold the UNLIMATE
        # prediction, that is, relu(input*w_0+b_0)*w_1 + b_1
        # We reset it before plotting the current status of the model
        # And the learned relu() functions
        sum_y_relu = []
        # For each relu() unit do the following
        for idx in range(n_hidden):

            plt.suptitle('Epoch=%d --- MSE loss= %.2f' % (epoch+1, loss.data.numpy()), fontsize=fontsize)
            # Plot output of the current relu() unit
            plt.subplot(1,3,1)
            plt.title('Relu(w_0*x + b_0)', fontsize=fontsize)
            y_relu = [relu(d*weights_0[idx]+biases_0[idx]) for d in data]
            plt.plot(data, y_relu)
            plt.ylim(-1,40)
            plt.grid()

            plt.subplot(1, 3, 2)
            # Plot output of the current relu(), multiplied by its
            # corresponding weight, w_1, and summed with the bias b_1
            plt.title('Relu(w_0*x + b_0)*w_1 + b_1',fontsize=fontsize)
            y_relu = [relu(d*weights_0[idx]+biases_0[idx])*weights_1[idx] + biases_1[0] for d in data]
            plt.plot(data,y_relu)
            plt.ylim(-500,900)
            plt.grid()

            # Kee adding the Relu(w_0*x + b_0)*w_1 + b_1 for each relu to the
            # sum_y_relu list. We will sum them up later to plot
            # The ULTIMATE predction of the model y_hat
            sum_y_relu.append([relu(d*weights_0[idx]+biases_0[idx])*weights_1[idx] + biases_1[0] for d in data])

        # Sum it all up
        sum_y_relu = np.sum(np.array(sum_y_relu),axis=0)
        plt.subplot(1, 3, 3)
        plt.title('y_hat)', fontsize=fontsize)
        plt.plot(x.data.numpy(), y.data.numpy(), color="k", label='Ground-truth')
        plt.plot(data,sum_y_relu, c='r', label='Prediction')
        plt.legend()
        plt.grid()

        # A slight delay in the animation
        plt.pause(0.1)
