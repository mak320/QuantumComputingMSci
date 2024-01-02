import numpy as np
import matplotlib.pyplot as plt
from math import ceil

FontSize = 20
# define paramters
Nv = 10  # number of velocity points
Nx = 10  # number of space points
N = Nv*Nx

dv = 1  # velocity step
dx = 1  # space step

vmax = (Nv-1) * dv / 2  # maximum velocity (not independent)

q = 1  # charge
m= 1 # mass 
eps_0 = 1  # permittivity
N_particles = 1



def ddslash(a, b):  # a/b with integer division
    return a - b * (ceil(a/b) - 1)


def kron_delta(a, b):  # § Kronecker delta
    a = int(a)
    b = int(b)
    if a == b:
        return 1
    else:
        return 0
    
def vj(j):
    vmax = (Nv-1) * dv /2 
    return -vmax + (j-1) * dv

def create_F1_hat():
    N = (Nv+1)*Nx
    F1_regular = np.zeros((N,N))
    F1_gauss = np.zeros((N,N))

    for n in range(0, N):
        n_phys = n + 1
        v_nddNv = vj(ddslash(n_phys, Nv))
        prefact = -v_nddNv / (2 * dx)

        if n_phys <= Nv:
            for k1 in range(0, N):
                k1_phys = k1 + 1
                F1_regular[n, k1] = prefact * (kron_delta(k1_phys, n_phys + Nv) -
                                                kron_delta(k1_phys, n_phys + Nv * (Nx-1))) 
            
        elif Nv < n_phys <= Nv*(Nx-1):
            for k2 in range(0, N):
                k2_phys = k2 + 1
                F1_regular[n, k2] = prefact * (kron_delta(k2_phys, n_phys + Nv) -
                                                kron_delta(k2_phys, n_phys - Nv))
        
        else:
            for k3 in range(0, N):
                k3_phys = k3 + 1
                F1_regular[n, k3] = prefact * (kron_delta(k3_phys, n_phys - Nv * (Nx -1)) -
                                                kron_delta(k3_phys, n_phys - Nv))
                
    for n in range(0, N):
        n_phys = n + 1
        prefact_gauss = -q**2 * N_particles / (2 * m * eps_0 * dv) * (ceil(n/Nv) - 1) / (Nx - 1)
        
        if ddslash(n_phys, Nv) == 1:
            for k1 in range(0, N):
                k1_phys = k1 + 1
                F1_gauss[n, k1] = prefact_gauss * kron_delta(k1_phys, n_phys +1)
        elif ddslash(n_phys, Nv) == Nv:
            for k2 in range(0, N):
                k2_phys = k2 + 1
                F1_gauss[n, k2] = -prefact_gauss * kron_delta(k2_phys, n_phys -1)
        else: 
            for k3 in range(0, N):
                k3_phys = k3 + 1
                F1_gauss[n, k3] = prefact_gauss * (kron_delta(k3_phys, n_phys +1) - kron_delta(k3_phys, n_phys -1))  

    return F1_regular + F1_gauss


F1_hat_gauss = create_F1_hat()

plt.imshow(F1_hat_gauss, cmap="plasma")
plt.xlabel("n", fontsize=FontSize)
plt.ylabel("k", fontsize=FontSize)
plt.title(r"$[\hat{F}^{(1)}]_{n, k}$", fontsize=FontSize)
plt.colorbar()
plt.show()