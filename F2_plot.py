import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

FontSize = 20
# define paramters 
Nv = 5 # number of velocity points
Nx = 5 # number of space points
N  = (Nv + 1)*Nx

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



def U_tensor_U(U):
    N = (Nv + 1) * Nx
    U_tensor_U = np.zeros(N**2)
    for n in range(0, N**2):
        n_phys = n + 1
        U_tensor_U[n] = U[ceil(n_phys/N)-1] * U[ddslash(n_phys, N) - 1]
    return U_tensor_U

def create_F2():

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

F2 = create_F2()

# Create a figure with inset_axes for zooming
fig, ax = plt.subplots(figsize=(6, 8))
axins = inset_axes(ax, width="200%", height="200%")

# Plot the entire F2 matrix (small)
cax1 = ax.imshow(F2, cmap="plasma")



# Set labels and title for the small plot
ax.set_xlabel("k", fontsize=FontSize)
ax.set_ylabel("n", fontsize=FontSize)
ax.set_title(r"$[F^{(2)}]_{n, k}$", fontsize=FontSize)

# Plot the zoomed-in submatrix (large)
cax2 = axins.imshow(F2[:, Nx * Nv * N:], cmap="plasma")

# Set labels and title for the zoomed-in plot
axins.set_xlabel("k", fontsize=FontSize)
axins.set_ylabel("n", fontsize=FontSize)

plt.show()