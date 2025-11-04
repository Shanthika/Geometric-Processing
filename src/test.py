# import numpy as np
# import matplotlib.pyplot as plt

# # --- 1D surface: a circle in 2D space ---
# # Parameterization
# u = np.linspace(0, 2 * np.pi, 200)
# x = np.cos(u)
# y = np.sin(u)+np.cos(u)

# # Choose a point on the circle
# u0 = np.pi / 4  # 45 degrees
# x0, y0 = np.cos(u0), np.sin(u0)+np.cos(u0)

# # Compute tangent and normal vectors
# xu = -np.sin(u0)  # dx/du
# yu = np.cos(u0)-np.sin(u0)   # dy/du
# tangent = np.array([xu, yu])
# normal = np.array([-x0, -y0])  # points toward center

# # Normalize for visualization
# tangent = tangent / np.linalg.norm(tangent)
# normal = normal / np.linalg.norm(normal)

# # --- Plot ---
# plt.figure(figsize=(6, 6))
# plt.plot(x, y, label='Circle surface', color='gray')
# plt.scatter(x0, y0, color='black', zorder=5)

# # Tangent and normal arrows
# plt.quiver(x0, y0, tangent[0], tangent[1], angles='xy', scale_units='xy', scale=4, color='blue', label='Tangent (x_u)')
# plt.quiver(x0, y0, normal[0], normal[1], angles='xy', scale_units='xy', scale=4, color='red', label='Normal')

# # Axes & labels
# plt.axis('equal')
# plt.title("Parametric Curve: Circle x(u) = (cos(u), sin(u))")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.grid(True)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Create surface parameters
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
u, v = np.meshgrid(u, v)

# Parametric surface: sphere of radius 1
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

# Pick a specific point (u0, v0)
u0, v0 = np.pi / 4, np.pi / 4
x0 = np.cos(u0) * np.sin(v0)
y0 = np.sin(u0) * np.sin(v0)
z0 = np.cos(v0)

# Compute partial derivatives (x_u and x_v)
x_u = np.array([-np.sin(u0)*np.sin(v0), np.cos(u0)*np.sin(v0), 0])  # ∂x/∂u
x_v = np.array([np.cos(u0)*np.cos(v0), np.sin(u0)*np.cos(v0), -np.sin(v0)])  # ∂x/∂v

# Compute normal vector (cross product)
n = np.cross(x_u, x_v)
n = n / np.linalg.norm(n)

# --- Plot the surface and vectors ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='lightgray', alpha=0.5, rstride=2, cstride=2, linewidth=0)

# Draw point
ax.scatter(x0, y0, z0, color='k', s=50)

# Tangent vectors
ax.quiver(x0, y0, z0, x_u[0], x_u[1], x_u[2], color='blue', length=0.3, normalize=True, label=r'$x_u$')
ax.quiver(x0, y0, z0, x_v[0], x_v[1], x_v[2], color='green', length=0.3, normalize=True, label=r'$x_v$')
ax.quiver(x0, y0, z0, n[0], n[1], n[2], color='red', length=0.3, normalize=True, label='normal')

# Labels and view
ax.set_title("Tangent Vectors $x_u$ and $x_v$ on a Sphere")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.set_box_aspect([1,1,1])
plt.show()
