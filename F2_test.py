import numpy as np
import matplotlib.pyplot as plt
from math import ceil

FontSize = 20
# define paramters 
Nv = 10 # number of velocity points
Nx = 10 # number of space points

dv = 1 # velocity step
dx = 1 # space step

vmax = (Nv-1) * dv /2 # maximum velocity (not independent)

q = 1 # change
m = 1 # mass


def ddslash(a, b): # a/b with integer division
    return int(a - b * (ceil(a/b) - 1))


def kron_delta(a, b): #§ Kronecker delta
    a = int(a)
    b = int(b)
    if a == b:
        return 1
    else:
        return 0
    
    
def U_n_converter(f_arr, E_arr): # converts the f_ij and E_i arrays to a single array of U_n values
    un = np.zeros(Nx*(Nv+1))
    for n in range(0, Nx*(Nv+ 1)):
        n_phys = n + 1
        if n_phys <= Nx * Nv:
            un[n] = f_arr[ceil(n_phys/Nv) - 1, ddslash(n_phys, Nv) - 1]
        else:
            un[n] = E_arr[n_phys - Nx*Nv -1]
    return un

# def U_tensor_U(f_arr, E_arr): #
#     u = U_n_converter(f_arr, E_arr)

#     N = (Nx + 1) * Nv

#     u_tensor_u = np.zeros(N**2)
#     for n in range(0, N**2):
#         u_tensor_u[n] = u[ceil(n/N)-1] * u[ddslash(n, Nv) - 1]
#     return u_tensor_u

def U_tensor_U(U):
    N = (Nv + 1) * Nx
    U_tensor_U = np.zeros(N**2)
    for n in range(0, N**2):
        n_phys = n + 1
        U_tensor_U[n] = U[ceil(n_phys/N)-1] * U[ddslash(n_phys, N) - 1]
    return U_tensor_U

def create_F2(Nx, Nv, dv):

    N  = (Nv + 1)*Nx
    F2 = np.zeros((N, N**2)) # create F^(2)_n,k

    K_const = -q / (2 * m * dv)

    for n in range(0, N):
        n_phys = n + 1
        n_bar = Nx * Nv * N + n_phys + N * (ceil(n_phys/Nv)-1)
        for k in range(0, N**2):
            k_phys = k + 1
            if k_phys > Nx * Nv * N:
                if ddslash(n_phys,  Nv) == 1:
                    F2[n, k] = K_const * kron_delta(k_phys, n_bar + 1)                   
                elif ddslash(n_phys, Nv) == Nv:
                    F2[n, k] = -K_const * kron_delta(k_phys, n_bar - 1)   
                else: 
                    F2[n, k] =  K_const* (kron_delta(k_phys, n_bar + 1) - kron_delta(k_phys, n_bar - 1))

    return F2

F2 = create_F2(Nx, Nv, dv)




"""Define initial condition"""
random_seed = 11145
np.random.seed(random_seed)
f = np.random.rand(Nx, Nv) 

# f = np.zeros((Nx, Nv))
# f[:, ::2] = 1
# f[:, 1::2] = 0.5

f = f / (np.sum(f) * (Nx-1) * dx * (Nv-1)* dv) 

plt.imshow(f)
plt.colorbar()
plt.show()

int_f_dv = np.sum(f, axis=1) # sums out the rows of the f_ij matrix, the rows corsopnd to constant space slices of f_ij, so this is exactly what we need to approimate an integral over velocities 
E_i = np.cumsum(int_f_dv)

dt = 1

inintial_un= U_n_converter(f, E_i)

initial_u_tensor_u = U_tensor_U(inintial_un)

"""Result from method 1"""
u_evolved_1_form = F2 @ initial_u_tensor_u * dt + inintial_un
# u_evolved_1_form = F2 @ np.kron(inintial_un, inintial_un) * dt + inintial_un

def propogate_u():

    def propogate_f(i, j):
        ii = i - 1
        ji = j - 1

        if j == 1:
            # print("edge1")
            return -q / (2 * m * dv) * E_i[ii] * f[ii, ji+1]
            
        elif j == Nv:
            # print("edgeNv")
            return q / (2 * m * dv) * E_i[ii] * f[ii, ji-1]
            
        else:
            # print("middle")
            return -q / (2 * m * dv) * E_i[ii] * (f[ii, ji+1] - f[ii, ji-1])

    f_evolved  = np.zeros_like(f)

    for i in range(0, Nx):
        i_phys = i + 1
        for j in range(0, Nv):
            j_phys = j + 1
            f_evolved[i, j] = f[i, j] + propogate_f(i_phys, j_phys) * dt

    u_evolved = U_n_converter(f_evolved, E_i)

    return u_evolved

u_evolved_2 = propogate_u()


diff = (u_evolved_1_form - u_evolved_2) / u_evolved_2
# for i, elem in enumerate(diff):
#     if elem != 0:
#         print(i)

plt.plot(diff, "o--")

plt.xlabel("n")
plt.ylabel("relative difference in u_n evolved")
plt.vlines(Nv*Nx, np.min(diff), np.max(diff), color="black", linestyles="dashed")
plt.show()



plt.plot(u_evolved_2, "o-", label="2")
plt.plot(u_evolved_1_form, "x--", label="1")
plt.vlines(Nv*Nx, np.min(u_evolved_2), np.max(u_evolved_2), color="black", linestyles="dashed")
plt.legend()
plt.show()



"""F2 visualisation"""
plt.imshow(F2, cmap="plasma")
plt.colorbar()

plt.show()

non_zero_indices = np.nonzero(F2)

# Create a scatter plot
plt.scatter(non_zero_indices[0], non_zero_indices[1], marker='o', c=F2[non_zero_indices], cmap='plasma')

plt.colorbar()
plt.title(r'Non-zero elements in $F^(2)$ matrix', fontsize=FontSize)
plt.xlabel('Row Index (n)', fontsize=FontSize)
plt.ylabel('Column Index (k)',  fontsize=FontSize)
plt.show()
