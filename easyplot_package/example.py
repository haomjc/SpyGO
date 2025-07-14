import numpy as np
import matplotlib.pyplot as plt
import easyplot as ep

# Create a figure and axes.
fig, ax = plt.subplots()

# Plot a line and a scatter on the same axes.
line_artist = ep.line(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)), ax=ax, color='blue')
scatter_artist = ep.scatter(np.linspace(0, 10, 50), np.cos(np.linspace(0, 10, 50)), ax=ax, color='red')

# Create an hgtransform group for the axes.
group = ep.HgTransform2D(ax)

# Add both artists to the transformation group.
group.add(line_artist)
group.add(scatter_artist)

# Define an affine transformation matrix (3x3) for 2D.
# For example, a rotation of 45 degrees.
theta = np.deg2rad(45)
cos_t, sin_t = np.cos(theta), np.sin(theta)
matrix2d = np.array([
    [cos_t, -sin_t, 0],
    [sin_t,  cos_t, 0],
    [0,      0,     1]
])

# Apply the transformation to both artists.
group.set_transform(matrix2d)

plt.show()
