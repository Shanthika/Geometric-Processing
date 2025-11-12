import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
# This is the mapping of 1D and 2D surface to a spherical signal
# cos(x) + i sin(x) = e^(ix)
# Parameterization of an annulus (circle with small width)
# def surface(u, v):
#     A = np.array([[2, 0], [0, 1]])
#     u_mat = np.array([[np.cos(2*np.pi*u)],[np.sin(2*np.pi*u)]])
#     x = u_mat.transpose()@A
#     return x[:,0,0], x[:,0,1]


# # Plot the "surface" and tangents
# u = np.linspace(0, 2 * np.pi, 200)
# x, y = surface(u, 0.0)

# plt.figure(figsize=(6,6))
# plt.plot(x, y, color='gray', label='Curve x(u,v0)')


# plt.axis('equal')
# plt.title("Tangents x_u and x_v on a Circle-like Surface (1D->2D extension)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.grid(True)
# plt.show()

# 3D surface

# --- Sphere parameterization ---
# def surface(u, v):
#     A = np.array([[1, 0.5, 0], [0, 1.5, 0], [0, 0, 1]])
#     x = np.cos(u) * np.sin(v)
#     y = np.sin(u) * np.sin(v)
#     z = np.cos(v)
#     u_mat = np.stack((x,y,z),-1)
#     x = u_mat@A 
#     return  x[...,0], x[...,1],x[...,2] 

# # --- Create the figure ---
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')


# # Sphere surface for context
# u = np.linspace(0, 2*np.pi, 60)
# v = np.linspace(0, np.pi, 30)
# U, V = np.meshgrid(u, v)
# X, Y, Z = surface(U, V)
# ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.6, edgecolor='none')


# # Formatting
# ax.set_box_aspect([1,1,1])
# ax.set_title("Sphere: Tangents x_u, x_v and Normal n")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.legend()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ Flat domain (FFT Laplacian)
# ---------------------------

N = 128
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

# Test function
f_flat = np.exp(-10 * (X**2 + Y**2))  # Gaussian

# FFT
fhat = np.fft.fft2(f_flat)
kx = np.fft.fftfreq(N, d=(x[1]-x[0]))
ky = np.fft.fftfreq(N, d=(y[1]-y[0]))
KX, KY = np.meshgrid(kx, ky)

# Laplacian in Fourier space
lap_fhat = -(2 * np.pi)**2 * (KX**2 + KY**2) * fhat
lap_flat = np.real(np.fft.ifft2(lap_fhat))

# ---------------------------
# 2️⃣ Sphere domain (finite difference approx Laplacian)
# ---------------------------

n_theta = 100  # azimuth divisions
n_phi = 50     # polar divisions
theta = np.linspace(0, 2*np.pi, n_theta)
phi = np.linspace(0, np.pi, n_phi)
theta_grid, phi_grid = np.meshgrid(theta, phi)

# Test function on sphere
f_sphere = np.sin(phi_grid) * np.cos(theta_grid)

# Approximate Laplace-Beltrami operator
dphi = phi[1] - phi[0]
dtheta = theta[1] - theta[0]

lap_sphere = np.zeros_like(f_sphere)

for i in range(1, n_phi-1):
    for j in range(n_theta):
        jp = (j+1) % n_theta
        jm = (j-1) % n_theta
        lap_sphere[i, j] = (
            (1/np.sin(phi[i])) * ( (np.sin(phi[i+1])*f_sphere[i+1,j] - np.sin(phi[i-1])*f_sphere[i-1,j]) / (2*dphi) ) +
            (f_sphere[i, jp] - 2*f_sphere[i,j] + f_sphere[i, jm]) / (dtheta**2)
        )

# ---------------------------
# 3️⃣ Plotting
# ---------------------------

fig, axs = plt.subplots(1, 2, figsize=(12,5))

# Flat domain Laplacian
im0 = axs[0].imshow(lap_flat, extent=(-1,1,-1,1), cmap='coolwarm')
axs[0].set_title("FFT Laplacian (Flat Domain)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
fig.colorbar(im0, ax=axs[0])

# Spherical Laplacian
im1 = axs[1].imshow(lap_sphere, extent=(0, 2*np.pi, 0, np.pi), cmap='coolwarm', origin='lower')
axs[1].set_title("Approx Laplace-Beltrami (Sphere)")
axs[1].set_xlabel("theta")
axs[1].set_ylabel("phi")
fig.colorbar(im1, ax=axs[1])

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Sphere resolution
n_theta, n_phi = 100, 50
theta = np.linspace(0, 2*np.pi, n_theta)
phi = np.linspace(0, np.pi, n_phi)
theta_grid, phi_grid = np.meshgrid(theta, phi)

# 3x3 matrix A (can rotate/scale)
A = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])

# 3D coordinates using A · [sin; cos] parameterization
X = np.cos(theta_grid) * np.sin(phi_grid)
Y = np.sin(theta_grid) * np.sin(phi_grid)
Z = np.cos(phi_grid)
points = np.stack([X, Y, Z], axis=-1)
XYZ = points @ A.T  # Apply matrix

# Define a scalar function on the sphere
f = XYZ[...,0] + XYZ[...,1] + XYZ[...,2]  # simple sum of coords

# Approximate Laplace-Beltrami using finite differences
dphi = phi[1]-phi[0]
dtheta = theta[1]-theta[0]

lap_f = np.zeros_like(f)
for i in range(1, n_phi-1):
    for j in range(n_theta):
        jp = (j+1) % n_theta
        jm = (j-1) % n_theta
        lap_f[i,j] = (
            (1/np.sin(phi[i])) * ((np.sin(phi[i+1])*f[i+1,j] - np.sin(phi[i-1])*f[i-1,j])/(2*dphi)) +
            (f[i,jp] - 2*f[i,j] + f[i,jm])/(dtheta**2)
        )

# Plotting
fig, axs = plt.subplots(1,2, figsize=(12,5))
im0 = axs[0].imshow(f, extent=(0,2*np.pi,0,np.pi), origin='lower', cmap='viridis')
axs[0].set_title("Scalar function f on sphere")
axs[0].set_xlabel("theta"); axs[0].set_ylabel("phi")
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(lap_f, extent=(0,2*np.pi,0,np.pi), origin='lower', cmap='coolwarm')
axs[1].set_title("Laplace-Beltrami of f")
axs[1].set_xlabel("theta"); axs[1].set_ylabel("phi")
fig.colorbar(im1, ax=axs[1])

plt.tight_layout()
plt.show()
