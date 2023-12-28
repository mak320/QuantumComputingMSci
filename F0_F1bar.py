"""
F0 inhomogenity and F1 collisional term implemented hre

"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil

FontSize = 20
# define paramters 
Nv = 10 # number of velocity points
Nx = 10 # number of space points

dv = 1 # velocity step
dx = 1 # space step


v_max = (Nv-1) * dv /2 
vj_vals = np.linspace(-v_max, v_max, Nv)
b = 1 # m/(2 kB T)

vmax = (Nv-1) * dv /2 # maximum velocity (not independent)


N_particles =  10 # number of particles
q =  1 # charge
eps_0 = 1 # permittivity
c = dv * q / eps_0 

def ddslash(a, b): # a/b with integer division
    return a - b * (ceil(a/b) - 1) 


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


def create_F1_matrix():
    F1 = np.zeros(((Nv+1)*Nx, (Nv+1)*Nx))

    for n in range(0, (Nv+1)*Nx):
        n_phys = n + 1
        pre_fact = (vmax - (ddslash(n_phys, Nv) - 1) * dv) / dx

        if n_phys <= Nv:
            for k1 in range(0, (Nv+1)*Nx):
                k1_phys = k1 + 1
                F1[n, k1] = pre_fact * (kron_delta(k1_phys, n_phys + Nv) +kron_delta(k1_phys, n_phys + Nv * (Nx-1)))

        elif n_phys > Nv and n_phys <= Nv*(Nx-1):
            for k2 in range(0, (Nv+1)*Nx):
                k2_phys = k2 + 1
                F1[n, k2] = pre_fact * (kron_delta(k2_phys, n_phys + Nv) - kron_delta(k2_phys, n_phys - Nv))

        elif n_phys > Nv*(Nx-1) and n_phys <= Nv*Nx:
            for k3 in range(0, (Nv+1)*Nx):
                k3_phys = k3 + 1
                F1[n, k3] = pre_fact * (kron_delta(k3_phys, n_phys - Nv * (Nx -1)) + kron_delta(k3_phys, n_phys - Nv))

        else:
            for k in range(0, (Nv+1)*Nx):
                k_phys = k + 1
                S = 0 
                for j in range(1, Nv+1): # here j is a physical index
            
                    # S += kron_delta(k_phys, (ddslash(n_phys, Nv) - 1) * Nv + j) * (-vmax + (j - 1) * dv)
                    S += kron_delta(k_phys, (n_phys -Nx * Nv - 1)*Nv + j) * (-vmax + (j - 1) * dv)

                #boundary = (vmax/2)  * (kron_delta(k_phys, (ddslash(n_phys, Nv) - 1) * Nv + 1) - kron_delta(k_phys, ddslash(n_phys, Nv)))
                boundary = (vmax/2)  * (kron_delta(k_phys, (n_phys -Nx * Nv - 1)*Nv + 1) - kron_delta(k_phys, (n_phys -Nx * Nv - 1)*Nv + Nv))

                F1[n, k] = c * ( S + boundary)
    return F1
        
"""Plot F1_hat (non-collisional)"""    
F1_hat = create_F1_matrix()

plt.imshow(F1_hat, cmap="plasma")
plt.xlabel("n", fontsize = FontSize)
plt.ylabel("k", fontsize = FontSize)
plt.title(r"$[\hat{F}^{(1)}]_{n, k}$", fontsize = FontSize)
plt.colorbar()
plt.show()


def vj(j):
    vmax = (Nv-1) * dv /2 
    return -vmax + (j-1) * dv


def collision_time(j):
    tau0 = 1
    a = 1
    return tau0 + a * vj(j)**4


def create_F1_bar():
    N = (Nv+1)*Nx
    F1_bar = np.zeros(((Nv+1)*Nx, (Nv+1)*Nx))
    for n in range(0, Nx*Nv): # non-zero elemets only for 
        n_phys = n + 1
        for k in range(0, N):
            k_phys = k + 1
            F1_bar[n, k] = kron_delta(k_phys, n_phys) / collision_time(ddslash(n_phys, Nv))
            
    return -F1_bar

F1_bar = create_F1_bar()
plt.imshow(F1_bar, cmap="plasma")
plt.xlabel("n", fontsize = FontSize)
plt.ylabel("k", fontsize = FontSize)
plt.title(r"$[\bar{F}^{(1)}]_{n, k}$", fontsize = FontSize)
plt.colorbar()
plt.show()

F1 = F1_bar + F1_hat
plt.imshow(F1, cmap="plasma")
plt.xlabel("n", fontsize = FontSize)
plt.ylabel("k", fontsize = FontSize)
plt.title(r"$[F^{(1)}]_{n, k}$", fontsize = FontSize)
plt.colorbar()
plt.show()



def normalisation():
    # Calculate the sum term
    sum_term = np.sum(np.exp(-b * vj_vals**2))
    # Calculate M
    M = (N_particles / (dx * dx * (Nx - 1))) / (sum_term - np.exp(-b * v_max**2))
    return M


def create_F0():
    N = (Nv+1)*Nx
    F0 = np.zeros(N)
    M = normalisation()
    for n in range(0, Nx*Nv): # non-zero elemets only for n <=NxNv 
        n_phys = n + 1
        F0[n] = M * np.exp(-b * vj(ddslash(n_phys, Nv))**2)/collision_time(ddslash(n_phys, Nv))
    return F0

F0 = create_F0()
plt.plot(F0)
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot F1_hat
im1 = axs[0].imshow(F1_hat, cmap="plasma")
axs[0].set_title(r"$[\hat{F}^{(1)}]_{n, k}$", fontsize=FontSize)
axs[0].set_xlabel("n", fontsize=FontSize)
axs[0].set_ylabel("k", fontsize=FontSize)

# Plot F1_bar
im2 = axs[1].imshow(F1_bar, cmap="plasma")
axs[1].set_title(r"$[\bar{F}^{(1)}]_{n, k}$", fontsize=FontSize)
axs[1].set_xlabel("n", fontsize=FontSize)


# Plot F1
im3 = axs[2].imshow(F1, cmap="plasma")
axs[2].set_title(r"$[F^{(1)}]_{n, k}$", fontsize=FontSize)
axs[2].set_xlabel("n", fontsize=FontSize)


# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im3, cax=cbar_ax,)
cbar.set_label('Value of Matrix Element', rotation=270, labelpad=15, fontsize=FontSize-5)

plt.show()
