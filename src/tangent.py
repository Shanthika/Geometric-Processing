import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#2D circle 
# def partial_2d(u0,v0):
#     # Compute partial derivatives
#     xu0 = np.array([-np.sin(u0) * (1 + v0), np.cos(u0) * (1 + v0)])  
#     xv0 = np.array([np.cos(u0), np.sin(u0)])                        
#     return xu0,xv0

# # Parameterization of an annulus (circle with small width)
# def surface(u, v):
#     x = (1 + v) * np.cos(u)
#     y = (1 + v) * np.sin(u)
#     return x, y


# def plot(angle):
        
#     # Pick a point
#     u0 = angle  # 45 degrees
#     v0 = 0.0        # on the middle circle; 
#     xu,xv = partial_2d(u0,v0)


#     # Compute point coordinates
#     x0, y0 = surface(u0, v0)

#     # Normalize for visualization
#     xu = xu / np.linalg.norm(xu)
#     xv = xv / np.linalg.norm(xv)

#     plt.scatter(x0, y0, color='black', zorder=5)

#     # Draw tangent vectors at the point
#     plt.quiver(x0, y0, xu[0], xu[1], angles='xy', scale_units='xy', scale=4, color='blue', label='x_u (tangent along u)')
#     plt.quiver(x0, y0, xv[0], xv[1], angles='xy', scale_units='xy', scale=4, color='green', label='x_v (tangent along v)')


# # Plot the "surface" and tangents
# u = np.linspace(0, 2 * np.pi, 200)
# x, y = surface(u, 0.0)

# plt.figure(figsize=(6,6))
# plt.plot(x, y, color='gray', label='Curve x(u,v0)')

# plot(0)
# plot(np.pi / 4)
# plot(np.pi / 2)
# plot(np.pi )
# plot((3*np.pi )/ 2)

# plt.axis('equal')
# plt.title("Tangents x_u and x_v on a Circle-like Surface (1D->2D extension)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.grid(True)
# plt.show()

# 3D surface

# --- Sphere parameterization ---
def surface(u, v):
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    return x, y, z

# --- Partial derivatives ---
def partials(u, v):
    xu = np.array([-np.sin(u)*np.sin(v), np.cos(u)*np.sin(v), 0])  # ∂x/∂u
    xv = np.array([np.cos(u)*np.cos(v), np.sin(u)*np.cos(v), -np.sin(v)])  # ∂x/∂v
    n = np.cross(xu, xv)
    n /= np.linalg.norm(n)
    return xu, xv, n

def plot(angle):
    u0, v0 = angle[0], angle[1]
    x0, y0, z0 = surface(u0, v0)
    xu, xv, n = partials(u0, v0)

    # Tangent and normal vectors at (u0,v0)
    J = np.array([xu,xv]).transpose()
    ws = np.array([[u0],[v0]])
    wd = J@ws
    breakpoint()
    # Normalize tangent vectors for display
    xu /= np.linalg.norm(xu)
    xv /= np.linalg.norm(xv)
    wd /= np.linalg.norm(wd)

    ax.quiver(x0, y0, z0, xu[0], xu[1], xu[2], color='blue', length=0.3, label='x_u (tangent along u)')
    ax.quiver(x0, y0, z0, xv[0], xv[1], xv[2], color='green', length=0.3, label='x_v (tangent along v)')
    ax.quiver(x0, y0, z0, n[0], n[1], n[2], color='red', length=0.3, label='n = x_u × x_v (normal)')
    ax.quiver(x0, y0, z0, wd[0], wd[1], wd[2], color='black', length=0.3, label='n = x_u × x_v (normal)')

    # Mark the point
    ax.scatter(x0, y0, z0, color='black', s=40)

# --- Create the figure ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


# Sphere surface for context
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
U, V = np.meshgrid(u, v)
X, Y, Z = surface(U, V)
ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.6, edgecolor='none')

#plot 
plot([0.0, 0.0])
plot([np.pi / 4, np.pi / 4])
plot([np.pi / 2, np.pi / 2])
plot([-np.pi , np.pi])
plot([(3*np.pi )/ 2, -(3*np.pi )/ 2])

# Formatting
ax.set_box_aspect([1,1,1])
ax.set_title("Sphere: Tangents x_u, x_v and Normal n")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()


# breakpoint()

# import trimesh
# import igl
# import potpourri3d as p3d
# import cv2

# mesh = trimesh.load('../data/meshes/unit_sphere.ply')
# verts = mesh.vertices
# faces = mesh.faces
# num_verts = verts.shape[0]


# # Laplacian matrix L
# L_dense = p3d.cotan_laplacian(verts, faces, denom_eps=1e-10)
# col = L_dense.toarray()@L_dense.toarray().transpose()
# col = (col-col.min())/(col.max()-col.min())
# cv2.imwrite('test.png',col*255) 

