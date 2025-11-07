# M132A-Optimization
A collection of files from my work in M132A - Optimization: Theory and Applications

File Details:
* homework
  * HW2_code.ipynb: minimizing one-dimensional functions using both golden and fibonacci search, minimizing a multidimensional function using golden search
  * HW3_code.ipynb: minimizing a multidimensional function using gradient descent
  * HW4_code.ipynb: zero finding for a one-dimensional function, computing conditions of matrices before and after normalization, performing SVD on data and optimizing machine learning for distinguishing digits
  * HW5_code.ipynb: minimizing multidimensional function using Newton's method and a fixed-step gradient descent, rank approximation of images
* libs
  * functions.py: helper functions used for various assignments
  * reading_matlab_file.py: helper file created by the professor for reading matlab files into a python environment. all work for the class was done in python, but the textbook we used was based on matlab
* data
  * T2.mat
  * T7.mat
  * X2.mat
  * X7.mat
  * Te28.mat
  * Lte28.mat
  * building256.mat
* Final Project: deliverable for our final project, created and optimized a physics-informed which solves the heat equation
  * data.py: samples points from a square of side length two centered around the origin. sampling is split into the interior, boundary, and initial settings. includes a plot for the distribution of sampled points
  * loss.py: creates and computes the residual loss (mean squared error) function for the network
  * model.py: contains the architecture for the network, allows for changing number of hidden (linear) layers in the model
  * training.py: contains a class for training the model and plotting the residual loss over epochs. also includes the option to save the model
  * Network.py: wrapper class for training the model given specific parameters, plotting loss over epochs, and visualizing the model's solution
  * Figure_1.png: figure of an example of sampled points for the region used in the project's problem
  * report.pdf: final report for the project which provides more insight into the overview of the problem and methodology of our approach
