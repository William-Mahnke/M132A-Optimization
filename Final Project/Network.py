import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


from model import NeuralNetwork
from data import SampleData
from loss import loss
from training import Training
from IPython.display import display, clear_output 

class Wrapper:
    def __init__(self, **kwargs):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seed
        torch.manual_seed(42)

        # Default parameters
        defaults = { 
            "t_max": 1, 
            "num_interior": 1000, 
            "num_initial": 1000, 
            "num_boundary": 1000, 
            "num_epochs": 10000, 
            "optimizer": optim.SGD, 
            "lr": 0.01, 
            "activation": "sigmoid", 
            "alpha": 1, 
            "verbose": False
        }
        
        # Update defaults with user-provided values
        self.params = {**defaults, **kwargs}

        # Create model
        self.model = NeuralNetwork(hidden_dim=10, num_layers=3, activation=self.params["activation"])

        # Create optimizer
        self.optimizer = self.params["optimizer"](self.model.parameters(), lr=self.params["lr"])

        # Create data
        self.data = SampleData()
        interior = self.data.interior_samples(self.params["t_max"], self.params["num_interior"])
        initial = self.data.initial_samples(self.params["num_initial"])
        boundary = self.data.boundary_samples(self.params["t_max"], self.params["num_boundary"])

        data = torch.cat((interior, initial, boundary), dim=0) # create a single tensor for all data since sizes are different

        # Create training object
        self.training = Training(
            data,
            self.model, 
            self.params["num_epochs"], 
            self.optimizer, 
            self.params["loss_fn"], 
            self.params["alpha"], 
            self.device, 
            self.params["verbose"]
        )

    def summarize(self):
        self.training.summarize(self.params["t_max"])

    def train(self):
        losses = self.training.train()
        return losses

    def plot_results(self, losses):
        self.training.plot_results(losses)

    def plot_graph_animation(self):
        '''
        Generates a plot of the network's solution for sampled data.
        Interactive plot updates temperatures regularly to show diffusion of heat
        '''
        # generate new data for plotting 
        self.plotting_data = SampleData()
        interior = self.plotting_data.interior_samples(self.params["t_max"], self.params["num_interior"])
        x_test, y_test = interior[:, 0].unsqueeze(1), interior[:, 1].unsqueeze(1)
        
        # create figure and axes for scatterplots of the network's solution
        fig, ax = plt.subplots()
        plt.ion() # makes the plot interactive 
        
        for k in range(100):
            with torch.no_grad():
                # compute network's solution to heat equation
                temp_input = torch.cat((x_test, y_test, torch.ones_like(x_test) * k/100), 1)
                soln = self.model(temp_input)
            
            ax.cla()
            ax.scatter(x = x_test.detach(), y = y_test.detach(), c = soln.detach(),
                       vmin = 0, vmax = 0.5, cmap = 'viridis')
            fig.canvas.draw() # redraws the figure 
            fig.canvas.flush_events() 
            plt.pause(0.1)
        plt.ioff() # turns off interactive mode
        plt.show() # keeps pop up window open once loop iterates completely 
    
    def plot_graph_slider(self):
        '''
        Generates a plot of the network's solution for sampled data
        Interactive plot has a slider for t to show diffusion at selected times
        '''
        # sample new data for plotting 
        self.plotting_data = SampleData()
        interior = self.plotting_data.interior_samples(self.params["t_max"], self.params["num_interior"])
        x_test, y_test = interior[:, 0].unsqueeze(1), interior[:, 1].unsqueeze(1)

        # Create figure and axes for scatterplot
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25) # adjusts the graph to account for the slider

        # calculating network's solution for initial time (t = 0)
        initial_time = 0.0
        with torch.no_grad():
            temp_input = torch.cat((x_test, y_test, torch.ones_like(x_test) * initial_time), 1) # input for network to calculate solution
            soln = self.model(temp_input) # network's solution

        # creates the scatterplots
        scatter = ax.scatter(x_test.detach(), y_test.detach(), c=soln.detach().flatten(),
                         vmin=0, vmax=0.5, cmap='viridis')
        fig.colorbar(scatter, ax=ax)

        ax_time = plt.axes([0.25, 0.1, 0.5, 0.03])  # [left, bottom, width, height]
        time_slider = Slider(ax=ax_time, label='Time', valmin=0.0, valmax=1.0, valinit=initial_time) # creates the slider for time

        # Update function for the slider
        def update(val):
            time = time_slider.val
            with torch.no_grad():
                temp_input = torch.cat((x_test, y_test, torch.ones_like(x_test) * time), 1)
                soln = self.model(temp_input)
            scatter.set_array(soln.detach().flatten().numpy())  # Update the scatter plot colors
            fig.canvas.draw_idle()

        # Connect the slider to the update function
        time_slider.on_changed(update)

        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Network Solution to 2D Heat Equation')
        plt.show()
            
if __name__ == "__main__":
    # Create wrapper
    wrapper = Wrapper(
        t_max=1,
        num_epochs=10000,
        num_interior=5000, 
        num_initial=500, 
        num_boundary=1000,
        optimizer=optim.Adam,
        loss_fn=loss,
        lr=0.001,
        activation="sigmoid",
        alpha=1,
        verbose=True
    )
    wrapper.summarize()
    losses = wrapper.train()
    #wrapper.plot_results(losses)
    wrapper.plot_graph_slider()
