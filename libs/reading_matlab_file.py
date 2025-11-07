import scipy.io;
import numpy as np;

print("="*80);
print("Example codes for writing and reading .mat files");
print("Paul J. Atzberger");
print("-"*80);
print("");

# We create an example .mat file 
my_data = {'vec_a':np.array([1,2,3]),'vec_b':np.array([[5,6],[7,8]])};
filename = 'example_01.mat';
print("Writing a .mat file.");
print("filename = " + filename); 
print("my_data.keys() = " + str(my_data.keys()));
scipy.io.savemat(filename,my_data);
print("");

# We load the file using 
print("Reading a .mat file.");
print("filename = " + filename); 
mat_file = scipy.io.loadmat(filename);
print(""); 

print("Getting data.");
vec_a = mat_file['vec_a'];
vec_b = mat_file['vec_b'];


print("vec_a = \n" + str(vec_a));
print("vec_b = \n" + str(vec_b));

print("="*80); 