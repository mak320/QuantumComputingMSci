import numpy as np
from math import ceil
import matplotlib.pyplot as plt
FontSize = 20

Nx = 15
Nv = 10
N = Nx * (Nv + 1)


dv = 1 # velocity step
dx = 1 # space step

vmax = (Nv-1) * dv /2 

# maximum velocity (not independent)

q =  1 # charge
eps_0 = 0.001 # permittivity
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


def create_F1_hat():
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


def vj(j):
    vmax = (Nv-1) * dv /2 
    return -vmax + (j-1) * dv


def collision_time(j, scale):
    tau0 = 1
    a = 1
    return (tau0) / scale

def create_F1_bar(scale):
    N = (Nv+1)*Nx
    F1_bar = np.zeros(((Nv+1)*Nx, (Nv+1)*Nx))
    for n in range(0, Nx*Nv): # non-zero elemets only for 
        n_phys = n + 1
        for k in range(0, N):
            k_phys = k + 1
            F1_bar[n, k] = kron_delta(k_phys, n_phys) / collision_time(ddslash(n_phys, Nv), scale)
            
    return -F1_bar

F1_hat = create_F1_hat()
F1_bar_weak = create_F1_bar(scale = 1)
F1_bar_strong = create_F1_bar(scale = 100)

F1_weak = F1_hat + F1_bar_weak
F1_strong = F1_hat + F1_bar_strong

plt.imshow(F1_weak)
plt.colorbar()
plt.title(r"$F_1$ weak")
plt.show()

plt.imshow(F1_strong)
plt.colorbar()
plt.title(r"$F_1$ s")
plt.show()

#calcualte the eigenvalues and eigenvectors of F1_weak and F1_stong visulatise  the eigenvalues next to eachother
eigenvalues_weak, eigenvectors_weak = np.linalg.eig(F1_weak)    
eigenvalues_strong, eigenvectors_strong = np.linalg.eig(F1_strong)


plt.plot(eigenvalues_weak.real, eigenvalues_weak.imag, "o", label = r"$z= 1$")
plt.plot(eigenvalues_strong.real, eigenvalues_strong.imag, "o", label = r"$z= 100$")
plt.xlabel(r"$\Re(\lambda)$", fontsize = FontSize)
plt.ylabel(r"$\Im(\lambda)$", fontsize = FontSize)
plt.legend(fontsize = FontSize) 
plt.show()

eigenvectors_strong_positive = []
eigenvalues_strong_positive = []
for i, e in enumerate(eigenvalues_strong):
    if e>=0:
        eigenvectors_strong_positive.append(eigenvectors_strong[:, i])
        eigenvalues_strong_positive.append(e)

eigenvectors_strong_positive = np.array(eigenvectors_strong_positive, dtype=complex)  
eigenvalues_strong_positive = np.array(eigenvalues_strong_positive, dtype=complex)

# calculate the outerporducts of all eigvenecors in eigenvectors_strong_positive and add these together to form a matrix called positive_eigen_projection
positive_eigen_projection = np.zeros((Nx*(Nv+1), Nx*(Nv+1)), dtype=complex)
for i, e in enumerate(eigenvectors_strong_positive):
        positive_eigen_projection += eigenvalues_strong_positive[i] * np.outer(e, e.conjugate())


fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Plot real part
im1 = ax1.imshow(positive_eigen_projection.real)
ax1.set_title('Real Part')

# Plot imaginary part
im2 = ax2.imshow(positive_eigen_projection.imag)
ax2.set_title('Imaginary Part')

# Create a common colorbar
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [x, y, width, height]
cbar = fig.colorbar(im2, cax=cax)
plt.show()


# Analyze F1_weak and F1_strong using SVD
U_weak, Sigma_weak, Vt_weak = np.linalg.svd(F1_weak)
U_strong, Sigma_strong, Vt_strong = np.linalg.svd(F1_strong)


# construct the best rank n estimator of F1_strong 
def rank_r_approximation(matrix, r):
    U, Sigma, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Keep only the first r singular values
    Sigma_r = np.diag(Sigma[:r])
    
    # Construct the rank-r approximation
    matrix_r_approx = np.dot(U[:, :r], np.dot(Sigma_r, Vt[:r, :]))
    
    return matrix_r_approx


# calculate rank-r approximations of F1_weak and F1_strong for r 1, .... N 
rank_r_approximations_weak = []
rank_r_approximations_strong = []
for r in range(1, N+1):
    rank_r_approximations_weak.append(rank_r_approximation(F1_weak, r))
    rank_r_approximations_strong.append(rank_r_approximation(F1_strong, r))


approx_weak = np.array(rank_r_approximations_weak)
approx_stong = np.array(rank_r_approximations_strong)

plt.imshow(approx_weak[50])
plt.show()

