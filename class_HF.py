import numpy as np
import matplotlib as plt
import copy
import math
import scipy

class gaussian:                     #Defining a class for the gaussians to be saved
    def __init__(self, exp, pre_exp, R = None):
        self.exp = exp              #Value of exponent
        self.pre_exp = pre_exp      #Value of preexponent 
        self.R = np.array(R)        #Position of Gaussian
        self.value = None           #Integral
    def pr(self): 
        print("Exp is: ", self.exp, " pre_exp is: ", self.pre_exp, " Centered at: x= ", self.R[0], ", y= ", self.R[1], ", z= ", self.R[2])
    def integral_value(self):             #To be used only this integral is a result of multiplication of two gaussians
        self.value = (np.pi/self.exp)**(3/2) * self.pre_exp ### CHANGE IT HERE TO 1/2 IN CASE

        
class atom: 
    def __init__(self, x, y, z, Z, N, index, name, basis_set):
        self.x = x                  #x coordinate
        self.y = y                  #y coordinate
        self.z = z                  #z coordinate 
        self.Z = Z                  #Number of Protons/Charge 
        self.N = N                  #Number of electrons 
        self.name = name
        self.index = index          #Numbering of this atom
        self.basis = np.array(basis_set)        #basis set used to describe this atoms
    def update_basis_set_with_pos(self):        #Centering the basis set, at the position of atom
        for i in range(3):
            self.basis[i].R = np.array([self.x, self.y, self.z])    