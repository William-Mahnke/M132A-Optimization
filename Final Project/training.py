import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import NeuralNetwork
from data import SampleData
from loss import loss, initial_function

class Training:
    '''
    Class that performs training processes and can summarize and plot results
    Args:
        model: NeuralNetwork
        data: SampleData (inherits from data.py)
        optimizer: torch.optim.Optimizer
        loss_fn: loss function
        num_epochs: int
        device: torch.device
    '''
    def __init__(self, data, model, num_epochs, optimizer, loss_fn, alpha, device, verbose=False):
        self.data = data
        self.model = model
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.device = device
        self.verbose = verbose
        # initialize loss history

    def train(self):
        '''
        Training Loop
        '''
        print("{:#^50}".format("Training on {}".format(self.device)))
        self.model.to(self.device)

        # initialize loss and history
        running_loss = 0.0
        loss_history = {'epochs': [], 'losses': []}

        # create a learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=1000, factor=0.1)

        # Send data to the device
        self.data = self.data.to(self.device)

        # training loop
        for epoch in range(self.num_epochs):
            # zero the gradients
            self.optimizer.zero_grad()

            # compute the loss (which passes each part of the data through the model)
            total_loss = self.loss_fn(self.data, 
                                      self.model,
                                      self.alpha)
            
            # backward propogation and optimization
            total_loss.backward()
            self.optimizer.step()

            # update loss history
            loss_history['epochs'].append(epoch)
            loss_history['losses'].append(total_loss.item())
            running_loss += total_loss.item()

            # print loss if verbose
            if self.verbose:
                if (epoch+1) % 1000 == 0:
                    print("Epoch: {}/{} Loss: {}".format(epoch+1, self.num_epochs, total_loss.item()))
            
            # update learning rate
            self.scheduler.step(total_loss)
            if self.verbose:
                if (epoch+1) % 1000 == 0:
                    print("Learning rate: {}".format(self.scheduler.get_last_lr()[0]))
        print("{:#^50}".format("Training Complete"))
        return loss_history


    def summarize(self, t_max):
        print("{:#^50}".format("Training Summary"))
        print("Training Device: {}".format(self.device))
        print("final time is {}".format(t_max))
        print("Training Data shapes: {}".format(self.data.shape)) # since our data is a single tensor
        print("Training Optimizer: {}".format(self.optimizer))
        print("Training Loss Function: {}".format(self.loss_fn))
        print("Training Number of Epochs: {}".format(self.num_epochs))
        print("Training diffusivity term: {}".format(self.alpha))
        print("Training Verbose: {}".format(self.verbose))
        print("{:#^50}".format(""))

    def plot_results(self, loss_history):
        epochs = loss_history['epochs']
        losses = loss_history['losses']
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, losses, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.legend()
        plt.show()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
