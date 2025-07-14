import numpy as np
import pyvista as pv
import easyplot as ep

# Create a PyVista plotter.
plotter = pv.Plotter()

# Create grid data for a surface.
X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = np.sin(np.sqrt(X**2 + Y**2))
surface_mesh = ep.surface(X, Y, Z, plotter=plotter, cmap='viridis')

# Create a simple patch (triangle) as another 3D object.
vertices = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [0.5, 1, 0]])
faces = [3, 0, 1, 2]  # single triangle
patch_mesh = ep.patch(vertices, faces, plotter=plotter, color='orange', opacity=0.7)

# Create an hgtransform group for 3D objects.
group3d = ep.HgTransform3D(plotter)

# Add both meshes to the transformation group.
group3d.add(surface_mesh)
group3d.add(patch_mesh)

# Define a 4x4 transformation matrix for 3D.
# For example, a rotation around the Z-axis by 45 degrees.
theta = np.deg2rad(45)
cos_t, sin_t = np.cos(theta), np.sin(theta)
matrix3d = np.array([
    [cos_t, -sin_t, 0, 0],
    [sin_t,  cos_t, 0, 0],
    [0,      0,     1, 0],
    [0,      0,     0, 1]
])

# Apply the transformation.
group3d.set_transform(matrix3d)

# Start the interactive 3D rendering.
plotter.show()