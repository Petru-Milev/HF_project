import numpy as np
import matplotlib as plt
import copy
import math
import scipy
from functions_HF import *
from class_HF import *

#-------------------------------------------------------------------------------------------------------------------------
#Input section 

h1 = gaussian(3.425250914, 0.1543289673)
h2 = gaussian(0.6239137298, 0.5353281423)
h3 = gaussian(0.1688554040, 0.4446345422)

H_basis_set = [copy.deepcopy(h1), copy.deepcopy(h2), copy.deepcopy(h3)]

H1 = atom(-6.01026972, 2.65849961, 1.30023139, 1, 1, 1, "H", copy.deepcopy(H_basis_set)) #x, y, z, Z, N, index, basis_set
H2 = atom(-6.98826972, 2.65849961, 1.30023139, 1, 1, 2, "H", copy.deepcopy(H_basis_set))
H3 = atom(-5.93261081, 2.46073295, 0.00000000, 1, 1, 3, "H", copy.deepcopy(H_basis_set))
H4 = atom(-6.88461081, 2.46073295, 0.00000000, 1, 1, 4, "H", copy.deepcopy(H_basis_set))

list_atoms = [H1, H2, H3, H4]

name = "H4_better_conv.txt"

for smth in list_atoms:
    smth.update_basis_set_with_pos()

#------------------------------------------------------------------------------------------------------------------------------
#Step 2, calculating integrals 
nr_atoms = len(list_atoms)                                                  #Nr of atoms in the system
nr_electrons = get_number_of_electrons(list_atoms)                          #Nr of electrons in the system

S_matrix = calc_S_matrix(nr_atoms, list_atoms)                              #Overlap integrals 
T_matrix = calc_T_matrix(nr_atoms, list_atoms)                              #Kinetic Energy Integrals 

V_matrix_list = get_list_V_matrixes(list_atoms, nr_atoms)                   #A tensor containing all the nuclear atraction integrals 
H = T_matrix + sum(V_matrix_list)                                           #Core Hamiltonian Matrix 
k = calc_double_electron_matrix(nr_atoms, list_atoms)                       #Double electron integrals 

#-------------------------------------------------------------------------------------------------------------------------------
###printing data in a file 
file_1 = open(name, "w")
file_1.write("Cartesian Coordinates" + "\tAtom Name\tAtomCharge \n")
for atom_1 in list_atoms:
    a =  "{:.8f}".format(atom_1.x) + "\t" + "{:.8f}".format(atom_1.y) + "\t" + "{:.8f}".format(atom_1.z) + "\t" + str(atom_1.name) + "\t" + str(atom_1.Z)
    file_1.write(a + "\n")

file_1.write("\n *** Basis sets used\n\n")

for atom_1 in list_atoms:
    a = str(atom_1.name) + " Z = "+ str(atom_1.Z) + "\tAtom Index " + str(atom_1.index) + "\n"
    file_1.write(a)
    for b in atom_1.basis:
        file_1.write(str(b.exp) + "\t" + str(b.pre_exp) + "\n")
    file_1.write("\n")

file_1.write("\n *** Overlap Matrix --------------------------------------\n\n")
matrix_str = np.array2string(S_matrix, separator=",", formatter={"int": lambda x: str(x)})
file_1.write(matrix_str)

file_1.write("\n\n *** T matrix --------------------------------------\n\n")
matrix_str = np.array2string(T_matrix, separator=",", formatter={"int": lambda x: str(x)})
file_1.write(matrix_str)

file_1.write("\n\n *** H matrix --------------------------------------\n\n")
matrix_str = np.array2string(H, separator=",", formatter={"int": lambda x: str(x)})
file_1.write(matrix_str)
#-------------------------------------------------------------------------------------------------------------------
#Diagonalization of Matrixes
# Calculate the eigenvalues and eigenvectors

eigenvalues, eigenvectors = scipy.linalg.eigh(S_matrix)

# Diagonalize the matrix Step 3

diagonalized_matrix = np.dot(np.dot(scipy.linalg.inv(eigenvectors), S_matrix), eigenvectors)

A = diagonalized_matrix 
B = np.diag(A)
B = B**(-1/2)
S_1div2 = np.diag(B)

U = eigenvectors

#X = np.dot(np.dot(U, S_1div2), U.T)  
X = np.dot(U, S_1div2)

file_1.write("\n\n *** X matrix. Canonical orthogonalization --------------------------------------\n\n")
matrix_str = np.array2string(X, separator=",", formatter={"int": lambda x: str(x)})
file_1.write(matrix_str)

#Getting cannonical orthogonalization matrix (above)

# Step 4 Finish --------------------------------------------------------
#Step 4, 5, 6

P_old = np.zeros((nr_atoms, nr_atoms)) #First Guess Matrix
F = np.zeros((nr_atoms, nr_atoms)) 

E_old = 10000                                              #Some value for starting energy

nn_repulsion = calc_nuclear_nuclear_repulsion(list_atoms)  #Repulsion Between Helium Atoms 
cycles = 0 
conv_crit_eng = 10**(-10)                                     #Energy Convergence Criteria 
conv_crit_cycles = 400                                     #Cycles Convergence Criteria
file_1.write("\n\n *** SCF Method --------------------------------------\n\n")
file_1.write("nn repulsion energy: " + str(nn_repulsion) + "\n")
while True:
    cycles += 1
    F = calc_F_matrix(F, H, P_old,k, nr_atoms)              #First guess at Fock Matrix
    E_new = get_E_value(P_old, H, F, nr_atoms)              #Calculating Electronic Energy
    E_new_nn = E_new + nn_repulsion                         #Adding nucleus nucleus repulsion to electronic energy
    file_1.write("Cycle: " + str(cycles) + " Energy_Total: " + str(E_new_nn) + "\n")    #Saving energy Values to output file
    print("Cycle Number: ", cycles, " Energy: ", E_new_nn)  #Printing it
    F_i = np.dot(X.T, np.dot(F, X))                         #Calculating the transformed Fock matrix F_i
    epsilon, C_i = scipy.linalg.eigh(F_i)                   #Diagonalizing F_i, to get C_i and epsilon
    epsilon = np.diag(epsilon)
    C = np.dot(X, C_i)                                      #Calculating C
    P_new = get_P_matrix(C, nr_atoms, nr_electrons)         #Getting new density Matrix P
    
    if abs(E_old - E_new) < conv_crit_eng:                  #Checking convergence of energy
        break 
    E_old = E_new
    P_old = P_new
    if cycles > conv_crit_cycles:                           #Checking convergence of cycles 
        break

file_1.close()                                              #Closing the output File 