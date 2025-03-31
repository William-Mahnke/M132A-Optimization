# file to generate data for our PINN for a PDE
import torch
from matplotlib import pyplot as plt

class SampleData:
    def __init__(self, default_samples=1000):
        self.default_samples = default_samples
        torch.manual_seed(42)

    def interior_samples(self, t_max, num_samples=None):
        '''
        Generate interior samples for the Heat Equation
        - t_max: max t value
        - num_steps: number of steps to sample
        '''
        if num_samples is None:
            num_samples = self.default_samples

        # sample from uniform distribution
        d = torch.distributions.Uniform(-1, 1)
        d_t = torch.distributions.Uniform(0, t_max)
        x = d.sample((num_samples, 1))
        y = d.sample((num_samples, 1))
        t = d_t.sample((num_samples, 1))

        return torch.cat((x, y, t), dim=1)

    def boundary_samples(self, t_max, num_samples=None):
        '''
        Generate boundary samples for the Heat Equation
        - t_max: max t value
        - num_steps: number of steps to sample
        '''
        if num_samples is None:
            num_samples = self.default_samples
        
        # Sample time points
        d_t = torch.distributions.Uniform(0, t_max)
        t = d_t.sample((num_samples, 1))
        
        # Create separate tensors for each boundary
        samples_per_boundary = num_samples // 4
        
        # Left boundary (x=-1, y∈[-1,1])
        x_left = torch.full((samples_per_boundary, 1), -1.0)
        y_left = torch.rand((samples_per_boundary, 1)) * 2 - 1
        t_left = d_t.sample((samples_per_boundary, 1))
        left_boundary = torch.cat((x_left, y_left, t_left), dim=1)
        
        # Right boundary (x=1, y∈[-1,1])
        x_right = torch.full((samples_per_boundary, 1), 1.0)
        y_right = torch.rand((samples_per_boundary, 1)) * 2 - 1
        t_right = d_t.sample((samples_per_boundary, 1))
        right_boundary = torch.cat((x_right, y_right, t_right), dim=1)
        
        # Top boundary (y=1, x∈[-1,1])
        x_top = torch.rand((samples_per_boundary, 1)) * 2 - 1
        y_top = torch.full((samples_per_boundary, 1), 1.0)
        t_top = d_t.sample((samples_per_boundary, 1))
        top_boundary = torch.cat((x_top, y_top, t_top), dim=1)
        
        # Bottom boundary (y=-1, x∈[-1,1])
        x_bottom = torch.rand((samples_per_boundary, 1)) * 2 - 1
        y_bottom = torch.full((samples_per_boundary, 1), -1.0)
        t_bottom = d_t.sample((samples_per_boundary, 1))
        bottom_boundary = torch.cat((x_bottom, y_bottom, t_bottom), dim=1)
        
        return torch.cat((left_boundary, right_boundary, top_boundary, bottom_boundary), dim=0)


    def initial_samples(self, num_samples=None):
        '''
        Generate initial samples for the Heat Equation where t = 0
        '''
        if num_samples is None:
            num_samples = self.default_samples

        # sample from uniform distribution
        d = torch.distributions.Uniform(-1, 1)
        x = d.sample((num_samples, 1))
        y = d.sample((num_samples, 1))
        t = torch.zeros_like(x)

        return torch.cat((x, y, t), dim=1)


if __name__ == "__main__":
    test_sample = SampleData()
    interior_samples = test_sample.interior_samples(1, 5000)
    boundary_samples = test_sample.boundary_samples(1, 1000)
    initial_samples = test_sample.initial_samples(1000)

    # plot the data in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # scatter as (x, t, y)
    ax.scatter(interior_samples[:, 0], interior_samples[:, 2], interior_samples[:, 1], label='Interior')
    ax.scatter(boundary_samples[:, 0], boundary_samples[:, 2], boundary_samples[:, 1], label='Boundary')
    ax.scatter(initial_samples[:, 0], initial_samples[:, 2], initial_samples[:, 1], label='Initial')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('y')
    plt.show()