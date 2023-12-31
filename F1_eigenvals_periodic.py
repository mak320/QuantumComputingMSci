import numpy as np
from math import ceil
import matplotlib.pyplot as plt

Nv_arr = np.arange(3, 20, 1) # number of velocity points
Nx_arr = np.arange(3, 20, 1) # number of space points

dv = 1 # velocity step
dx = 1 # space step

# maximum velocity (not independent)

q =  1 # charge
eps_0 = 1 # permittivity
c = dv * q / eps_0 



def ddslash(a, b): # a/b with integer division
    return a - b * (ceil(a/b) - 1) 

def kron_delta(a, b):
    a = int(a)
    b = int(b)
    if a == b:
        return 1
    else:
        return 0 # Kronecker delta
    



def create_F1_matrix():
    F1 = np.zeros((N,N))

    for n in range(0, N):
        n_phys = n + 1
        v_nddNv = -vmax + (ddslash(n_phys, Nv) - 1) * dv

        pre_fact = - v_nddNv / (2 * dx)

        if n_phys <= Nv:
            for k1 in range(0, N):
                k1_phys = k1 + 1
                F1[n, k1] = pre_fact * (kron_delta(k1_phys, n_phys + Nv) - kron_delta(k1_phys, n_phys + Nv * (Nx-1)))

        elif n_phys > Nv and n_phys <= Nv*(Nx-1):
            for k2 in range(0, N):
                k2_phys = k2 + 1
                F1[n, k2] = pre_fact * \
                    (kron_delta(k2_phys, n_phys + Nv) - kron_delta(k2_phys, n_phys - Nv))

        elif n_phys > Nv*(Nx-1) and n_phys <= Nv*Nx:
            for k3 in range(0, N):
                k3_phys = k3 + 1
                F1[n, k3] = pre_fact * (kron_delta(k3_phys, n_phys - Nv * (Nx-1)) - kron_delta(k3_phys, n_phys - Nv))

        else:
            for k in range(0, (Nv+1)*Nx):
                k_phys = k + 1
                S = 0
                for j in range(1, Nv+1):  # here j is a physical index
                    v_j = -vmax + (j - 1) * dv
                    # S += kron_delta(k_phys, (ddslash(n_phys, Nv) - 1) * Nv + j) * (-vmax + (j - 1) * dv)
                    S += kron_delta(k_phys, (n_phys - Nx *
                                    Nv - 1) * Nv + j) * v_j

                # boundary = (vmax/2)  * (kron_delta(k_phys, (ddslash(n_phys, Nv) - 1) * Nv + 1) - kron_delta(k_phys, ddslash(n_phys, Nv)))
                boundary = (vmax/2) * (kron_delta(k_phys, (n_phys - Nx * Nv - 1)
                                                  * Nv + 1) - kron_delta(k_phys, (n_phys - Nx * Nv - 1)*Nv + Nv))

                F1[n, k] = c * (S + boundary)
    return F1

eigenvals = []
for Nx in Nx_arr:
    for Nv in Nv_arr:
        vmax = (Nv-1) * dv /2 
        N = (Nv+1)*Nx
        F1 = create_F1_matrix()
        lamda = np.linalg.eigvals(F1)
        lamda_max  = max(lamda.real)
        eigenvals.append(lamda_max)

eigenvals = np.array(eigenvals)


plt.hist(eigenvals, histtype="stepfilled", alpha=0.5, edgecolor="black")
plt.xlabel(r"Largest real part of $F_1$ eigenvalues")
plt.ylabel("Count")
plt.title(r"$3 \leq N_x \leq 20$ and $3 \leq N_v \leq 20$")
plt.show()
