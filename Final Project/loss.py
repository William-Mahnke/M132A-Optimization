import torch
import math
from data import SampleData
import matplotlib.pyplot as plt


# initial function, subject to change
def initial_function(x, y):
    return torch.exp(-1 / (1 - ((x*x) - (y*y))))

'''
PDE: u_t = alpha * u_xx 1D heat equation
'''

# data is initial, boundary (left, right), interior so we need to compute loss for the three parts
def loss(data, network, alpha=1, num_interior=1000, num_initial=100, num_boundary=100):
    '''
    Loss function for a simple heat equation in 1D
    Args:
        - interior: input interior values
        - initial: input initial values
        - boundary: input boundary values
        - network: neural network to approximate the solution
        - alpha: coefficient for the heat diffusion term
    returns:
        - total_loss: Summed up loss for the initial, boundary, and interior conditions
    '''
    # split data into interior, initial, boundary
    interior = data[0:num_interior]
    initial = data[num_interior:num_interior+num_initial]
    boundary = data[num_interior+num_initial:]

    # interior points as (x, y, t)
    x = interior[:, 0].clone().detach().requires_grad_(True)
    y = interior[:, 1].clone().detach().requires_grad_(True)
    t = interior[:, 2].clone().detach().requires_grad_(True)
    
    # compute loss for initial, boundary, and interior
    interior_points = torch.cat((x.unsqueeze(1), t.unsqueeze(1), y.unsqueeze(1)), dim=1)
    u = network(interior_points)

    # Compute gradients
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    # Compute losses
    loss_interior = torch.mean((alpha * (u_xx + u_yy)- u_t) ** 2)
    loss_initial = torch.mean((network(initial) - (initial_function(initial[:, 0], initial[:, 1]))) ** 2)
    
    # If boundary conditions are u(-1,t) = 0 and u(1,t) = 0, keep as is
    boundary_loss = torch.mean((network(boundary)) ** 2)

    total_loss = loss_interior + loss_initial + boundary_loss
    
    return total_loss


if __name__ == "__main__":
    # test initial function
    X, Y = torch.meshgrid(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100))
    Z = initial_function(X, Y)
    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    plt.show()
