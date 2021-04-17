from numpy import exp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def Loss(y):
    # Computes the loss for a given output of the model, y
    return (-y**2 + y**3)*exp(-(y**2))

def dLossdW(x,y):
    # Computes the gradient of the loss w.r.t the parameter w (Using the chain Rule)
    # First the derivative of the loss w.r.t y
    dlossdy = ((-2*y + 3*(y**2))*exp(-(y**2))) + ((-y**2 + y**3)*(-2*y*exp(-(y**2))))
    # Then the derivative of y w.r.t w
    dydw = x
    # Finally we return the multiplication of the these two, that is the gradient
    # of the loss w.r.t w
    dlossdw = dlossdy*dydw
    return dlossdw


################################### First plot a 3D error surface across all inputs and all outputs
# Define a range of values for input data
X = np.linspace(-5, 5, 100)
# Define a range of values for the parameter w
W = np.linspace(-0.7, 0.7, 100)
# Create a mesh grid of these to vectors
x, w = np.meshgrid(X, W)
# Compute the output of the model for each pair of values in the mesh
y = w*x
# Create a figure
fig = plt.figure(figsize=(16,9))
# Tell matplotlib that this is going to be a 3-dimensional plot
ax = plt.axes(projection='3d')
# use the plot_surface function and a nice cmap to plot the loss surface w.r.t all pairs of (x,w) in the mesh
ax.plot_surface(x, w, Loss(y), rstride=2, cstride=2,
                cmap='hot', edgecolor='none')

# Set labels for the axes
ax.set_xlabel('Input Data (x)', size=17)
ax.set_ylabel('The model parameter (w)', size=17)
ax.set_zlabel('Loss', size=17)
# Compute the value of the global minimum
plt.title('Global Minimum is %.2f' % np.min(Loss(y)), size=17)
# Show the 3-dimensional surface
plt.show()


#################################### plot a 2D error surface per each input value across a range of values in w
# 3 data points between -5 and 5 are selected
X = np.linspace(-5, 5, 3)
# A range of possible values for parameter w is selected
W = np.linspace(-0.7, 0.7, 100)
# Create a figure
plt.figure(figsize=(16,9))
# iterate through the entire dataset and for each value of x repeat the following
for x in X:
    # compute the output of the model
    y = W*x
    # Plot the current loss surface for x, across the the entire range of values
    # For the parameter w
    plt.plot(W, Loss(y), c='r', alpha=0.7)

# define the limits for horizontal axis (the weight axis)
plt.xlim(min(W), max(W))
# put the labels for weight and loss axes
plt.xlabel('Weight Values (w)', size=17)
plt.ylabel('One individual loss surface per input data (%d surfaces)' % len(X), size=17)
# Put grids on the plot for better visualization
plt.grid()
# Show the plot
plt.show()


##################################### plotting error surfaces, computing gradients at w_0, and computing w_new.
###################################### Finally, using w_not and w_new, we can plot the gradient vector##########
# Define a range of values for input data x
X = np.linspace(-5, 5, 10)
# Define a grid of values for the parameter w
W = np.linspace(-0.7, 0.7, 100)
# Define an initial value of w_not (where we start learning from)
w_not = -0.3
# Define the learning rate
eta = 0.01

# Create a figure in which we will put 2 subplots
fig = plt.figure(figsize=(16,9))
# Create the first subplot
plt.subplot(1, 2, 1)
# Give a title for the subplot
plt.title('Update vectors computed using \n '
          'the gradients at w_not for different error surfaces',size=17)
# This variable is giong to be used for plotting the update vectors! This
# Makes the vectors look nice and symmetrical
prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",shrinkA=0, shrinkB=0)

# This will hold the computed gradients at w_not, across all loss surfaces given each individual data x
gradients = []
# Go through the entire dataset X, and for each data point x do the following
for x in X:
    # Compute model output
    y = w_not*x
    # Compute the gradient of the error w.r.t the parameter w, given current value of the input data
    dedw = dLossdW(x,y)
    # Add the gradient to the list for future visualizations
    gradients.append(dedw)
    # Compute the new value of the parameter w NOTE: We are not yet updating w_not!
    w_new = w_not - eta*dedw
    # Plot the loss surface for the current input data x, across possible values of the parameter w
    plt.plot(W,Loss(W*x), c='r', alpha=0.7)
    # Plot the initial w_not and its corresponding loss value Loss(y) given x, so we know where on
    # The loss surface we reside
    plt.scatter(w_not, Loss(y), c='k')
    # Using the (w_not,Loss(y)) and the new value of the weight that results in the point (w_new, Loss(w_new*x))
    # Plot the update vector between w_not and w_new
    plt.annotate("", xy=(w_new, Loss(w_new*x)), xytext=(w_not, Loss(y)), arrowprops=prop)
    # Put a limit on the weight axis using the minimum and maximum values we have considered for w
    plt.xlim(min(W), max(W))

# Put labels per axis
plt.xlabel('Weight (w)',size=17)
plt.ylabel('%d Individual loss surfaces per data input x' % len(X), size=17)
# Plot a nice vertical blue line at w_not so we know where we stand INITIALLY across ALL loss surfaces
plt.axvline(w_not, ls='--', c='b',label='w_not=%.2f' % w_not)
# Show the legends
plt.legend()
# Put a nice grid
plt.grid()

# Prepare the second subplot
plt.subplot(1,2,2)
# Put a nice title for the histogram
plt.title('Frequency of the magnitudes of the computed gradients',size=17)
# Plot the histogram of gradients, along some nice statistics of these gradients
plt.hist(gradients, label='(Min, Max, Mean)=(%.2f,%.2f,%.2f)' % (np.min(gradients), np.max(gradients),np.mean(gradients)))
# Make the legends visible
plt.legend()
# Put some nice axis labels
plt.xlabel('Gradient values', size=17)
plt.ylabel('Frequency of %d gradients' % len(X), size=17)
# Put a nice grid
plt.grid()
# Show the whole plot with both subplots
plt.show()

##################################################### Let's travel 2-D with SGD (Updating w_not as we go)
# The description of these lines is same as before
X = np.linspace(-5, 5, 100)
W = np.linspace(-0.7, 0.7, 100)
w_not = -0.3
eta = 0.01
# We keep the initial value of w_not in a variable, because we WILL update w_not using SGD this time!
origin = w_not

plt.figure(figsize=(16,9))
prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",
                shrinkA=0, shrinkB=0)

for i, x in enumerate(X):

    # This is there to clear the previous plot in the animation, so the next one can be plotted over
    plt.clf()
    y = w_not * x
    dedw = dLossdW(x, y)
    # Remember, we just want to compute w_new first so we can plot the update vector
    # Between w_not and w_new, BEFORE we actually update w_not!
    w_new = w_not - eta * dedw

    # Compute the loss at both w_not and the newly computed w_new. These will help with plotting the update vectors
    y_w_not = Loss(w_not * x)
    y_w_new = Loss(w_new * x)

    # Let's keep the initial value of w_not (that we have saved in origin) and plot a vertical line on it
    # So we will always clearly see where SGD started its magic from!
    plt.axvline(origin, ls='--', c='b',label='origin =%.2f' % origin)

    # Plot the loss surface for the given data point x
    plt.plot(W, Loss(W * x), c='r', alpha=0.7, label = 'current loss =%.2f' % y_w_new)
    plt.legend()
    # Plot the current w_not and the corresponding loss value (i.e., y_w_not) BEFORE updating w_not
    plt.scatter(w_not, y_w_not, c='k')
    # Plot the update vector between w_not and w_new for the current loss surface, given the current input data x
    plt.annotate("", xy=(w_new, y_w_new), xytext=(w_not, y_w_not), arrowprops=prop)
    plt.grid()
    plt.xlabel('Weights (x)', size=17)
    plt.ylabel('Individual loss surface for a given input data (x)', size=17)
    plt.title('Changes in loss surface given the %d th input data' % (i + 1), size=17)
    # Put limits for a better animation
    plt.xlim(min(W), max(W))
    plt.ylim(-1.0, 0.5)
    # This controls the speed of the animation in terms of milli-seconds
    plt.pause(0.1)
    # Fanally: Before inputting the next input value into the model, we UPDATE w_not!!! This is where SGD
    # changes the current value of the learned parameter w, so that it could find the global minimum that is -0.76
    # In our example
    w_not = w_new