import numpy as np
import matplotlib as plt
import copy
import math
import scipy
from class_HF import gaussian
from class_HF import atom

def mult_gaussians(g1, g2):
    new_R = g1.R - g2.R
    K = np.exp((-g1.exp*g2.exp)/(g1.exp + g2.exp) * (np.linalg.norm(new_R))**2)
    Rp = (g1.exp*g1.R + g2.exp*g2.R)/(g1.exp + g2.exp)
    new_exp = g1.exp + g2.exp 
    new_gaussian = gaussian(new_exp, K, Rp)
    new_gaussian.integral_value()
    a = new_gaussian.value
    return new_gaussian

def multicenter(a):
    n_centers = len(a)
    if n_centers == 2: 
        return mult_gaussians(a[0], a[1])
    new_center = mult_gaussians(a[0], a[1])
    const = new_center.pre_exp
    for i in range(2, n_centers):
        old_center = new_center
        new_center = mult_gaussians(old_center, a[i])
        const *= new_center.pre_exp
    new_center.pre_exp = const
    return new_center

def T(A,B):                   #Kinetic energy calc
    new_R = A.R - B.R
    value = A.exp * B.exp / (A.exp + B.exp) * (3 - 2 * A.exp * B.exp / (A.exp + B.exp) * (np.linalg.norm(new_R))**2) * \
    (np.pi/(A.exp + B.exp))**(3/2) * np.exp(-A.exp*B.exp/(A.exp + B.exp) * (np.linalg.norm(new_R))**2)
    #value = value * A.pre_exp * B.pre_exp
    return value

def V(A, B, Rc, Z):                #Potential ener
    
    new_R_AB = A.R - B.R
    Rp = (A.exp*A.R + B.exp*B.R)/(A.exp + B.exp)
    new_R_PC = Rp - Rc    
    if np.linalg.norm(new_R_PC) == 0:
        erfr = 1
    else:
        erfr = 1/2 * (np.pi / ((A.exp + B.exp) * (np.linalg.norm(new_R_PC))**2)) ** (1/2) * math.erf(((A.exp + B.exp)*np.linalg.norm(new_R_PC)**2)**(1/2))
    value = -2*np.pi/(A.exp + B.exp) * Z * np.exp(-A.exp*B.exp/(A.exp + B.exp) * (np.linalg.norm(new_R_AB))**2) * erfr 
    #value = value * A.pre_exp * B.pre_exp
    return value

def two_electron(A, B, C, D):
    Rp = (A.exp*A.R + B.exp*B.R)/(A.exp + B.exp)
    Rq = (C.exp*C.R + D.exp*D.R)/(C.exp + D.exp)
    new_R_AB = A.R - B.R
    new_R_CD = C.R - D.R
    new_R_PQ = Rp - Rq
    value1 = 2*(np.pi**(5/2))/((A.exp + B.exp)*(C.exp + D.exp) * (A.exp + B.exp + C.exp + D.exp) ** (1/2))
    value2 = np.exp(-A.exp*B.exp/(A.exp + B.exp) * ((np.linalg.norm(new_R_AB))**2) -  C.exp*D.exp/(C.exp + D.exp) * ((np.linalg.norm(new_R_CD))**2))
    if (np.linalg.norm(new_R_PQ)) == 0:
        value3 = 1
    else: 
        value_for_erf = (A.exp + B.exp)*(C.exp + D.exp) / (A.exp + B.exp + C.exp + D.exp) * ((np.linalg.norm(new_R_PQ))**2)
        value3 = 1/2 * (np.pi/(value_for_erf))**(1/2) * math.erf(value_for_erf**(1/2))
    value = value1 * value2 * value3 
    return value

def S(A, B):
    new_R = A.R - B.R
    value = (np.pi/(A.exp + B.exp))**(3/2) * np.exp((-1*A.exp*B.exp)/(A.exp + B.exp) * (np.linalg.norm(new_R))**2)
    return value

def calc_S_matrix(n, list_atoms):
    ###########################################
    def mult_basis_set(b1, b2):
        length_b1 = len(b1)
        length_b2 = len(b2)
        sum_1 = 0
        for i in range(length_b1):
            for i1 in range(length_b2): 
                sum_1 +=  (2*(b1[i].exp)/np.pi)**(3/4) *(2*(b2[i1].exp)/np.pi)**(3/4) * b1[i].pre_exp * b2[i1].pre_exp * S(b1[i], b2[i1])
    ####INTEGRALS NEED TO BE NORMALIZED####
        return sum_1
    #############################################
    S_matrix = np.zeros((n, n))
    for u in range(n):             #Rows
        for v in range(n):         #Column (same as element of a row)
            S_matrix[u][v] = mult_basis_set(list_atoms[u].basis, list_atoms[v].basis)
    return S_matrix
            
def calc_T_matrix(n, list_atoms):
    
        ###########################################
    def mult_basis_set(b1, b2):
        length_b1 = len(b1)
        length_b2 = len(b2)
        sum_1 = 0
        for i in range(length_b1):
            for i1 in range(length_b2): 
                sum_1 +=  (2*(b1[i].exp)/np.pi)**(3/4) * (2*(b2[i1].exp)/np.pi)**(3/4) * b1[i].pre_exp * b2[i1].pre_exp * T(b1[i], b2[i1])
        return sum_1
    #############################################
    
    T_matrix = np.zeros((n, n))
        
    for u in range(n):
        for v in range(n):
            T_matrix[u][v] = mult_basis_set(list_atoms[u].basis, list_atoms[v].basis)
    return T_matrix


def calc_V_matrix(n, list_atoms, atom):
    
    def mult_basis_set(b1, b2, Rc, Z):
        length_b1 = len(b1)
        length_b2 = len(b2)
        sum_1 = 0
        for i in range(length_b1):
            for i1 in range(length_b2): 
                sum_1 +=   (2*(b1[i].exp)/np.pi)**(3/4) * (2*(b2[i1].exp)/np.pi)**(3/4) * b1[i].pre_exp * b2[i1].pre_exp * V(b1[i], b2[i1], Rc, Z)
        return sum_1
    
    V_matrix = np.zeros((n,n))
    for u in range(n):
        for v in range(n):
            V_matrix[u][v] = mult_basis_set(list_atoms[u].basis, list_atoms[v].basis, (atom.x, atom.y, atom.z), atom.Z)
    return V_matrix

def calc_double_electron_matrix(n, list_atoms):
#-----------------------------------------------------------    
    def mult_basis_set(b1, b2, b3, b4): 
        sum_1 = 0
        for i in range(len(b1)):
            for i1 in range(len(b2)):
                for i2 in range(len(b3)):
                    for i3 in range(len(b4)):
                        const = (2*(b1[i].exp)/np.pi)**(3/4) * (2*(b2[i1].exp)/np.pi)**(3/4) * (2*(b3[i2].exp)/np.pi)**(3/4) * (2*(b4[i3].exp)/np.pi)**(3/4)
                        const = const * b1[i].pre_exp * b2[i1].pre_exp * b3[i2].pre_exp * b4[i3].pre_exp
                        sum_1 += const * two_electron(b1[i], b2[i1], b3[i2], b4[i3])
        return sum_1
#-----------------------------------------------------------                        
    
    double_electron_matrix = np.zeros((n,n,n,n)) #!!!
    
    for u in range(n):
        for v in range(n):
            for p in range(n):
                for q in range(n):
                    double_electron_matrix[u][v][p][q] = mult_basis_set(list_atoms[u].basis, list_atoms[v].basis, 
                                                                       list_atoms[p].basis, list_atoms[q].basis)
    return double_electron_matrix


def calc_F_matrix(F, H, P, k, n):
    for u in range(n):
        for v in range(n):
            F[u][v] = H[u][v]
            for i_l in range(n):
                for i_s in range(n):
                    F[u][v] += P[i_l][i_s]*(k[u][v][i_s][i_l] - 1/2 * k[u][i_l][i_s][v])
    return F

def get_P_matrix(C, n, N):       #n - number of Atoms, N - number of electrons 
    P = np.zeros((n,n))
    b = int(np.round(N/2, 0))
    N = 3 
    for u in range(n):
        for v in range(n):
            P[u][v] = 0
            for a in range(b):
                P[u][v] += 2*C[u][a]*C[v][a]
    return P

def get_E_value(P, H, F, n):
    E = 0 
    for u in range(n):
        for v in range(n):
            E += 1/2 * P[u][v]*(H[u][v] + F[u][v])
    return E


#------------------------------------------------------------------------------
#Calculating coulombic repulsion between nucleai. In atomic units. 
#list_atoms - a list containing all atoms in the system 
#x1, y1, z1, x2, y2, z2 - coordinates of atom 1 and 2 
#sum_1 - nuclear repulsion 

def calc_nuclear_nuclear_repulsion(list_atoms): 
    
    def distance(x1, y1, z1, x2, y2, z2):
        d = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 
        return np.sqrt(d)
    
    sum_1 = 0 
    n = len(list_atoms)
    for i in range(n):
        for i1 in range(n): 
            if i1 > i: 
                sum_1 += list_atoms[i].Z*list_atoms[i1].Z/distance(list_atoms[i].x, list_atoms[i].y, list_atoms[i].z, list_atoms[i1].x, list_atoms[i1].y, list_atoms[i1].z)
    return sum_1

#------------------------------------------------------------------------------

#Getting all the V_matrixes, keeping them in a list, which then is summed up to calculate the H matrix

def get_list_V_matrixes(list_atoms, nr_atoms):
    
    result = []
    for i in list_atoms: 
        V_matrix = calc_V_matrix(nr_atoms, list_atoms, i)
        result.append(V_matrix)
    return result
#--------------------------------------------------------------------------------
#getting the number of electrons in the studied system 

def get_number_of_electrons(list_atoms):
    sum_1 = 0 
    for i in list_atoms:
        sum_1 += i.N
    return sum_1 