import numpy as np
from vispy import scene
from vispy.scene import visuals

# Create a canvas with a 3D viewport
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Create grid data
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X**2 + Y**2)

# Create a surface plot
surface = scene.visuals.SurfacePlot(z=Z, color=(0.5, 0.6, 1, 0.8), shading='smooth')
view.add(surface)

# Adjust the axes scaling
surface.transform = scene.transforms.STTransform(scale=(2, 0.5, 1))

# Add grid for better reference
grid = visuals.GridLines()
view.add(grid)

# Add axes with labels
axis_x = scene.Axis(pos=[[0, 0], [10, 0]],
                    tick_direction=(0, -1),
                    axis_label='X-axis',
                    axis_font_size=12,
                    major_tick_length=10,
                    tick_width=2,
                    text_color='white')

axis_y = scene.Axis(pos=[[0, 0], [0, 10]],
                    tick_direction=(-1, 0),
                    axis_label='Y-axis',
                    axis_font_size=12,
                    major_tick_length=10,
                    tick_width=2,
                    text_color='white')

axis_z = scene.Axis(pos=[[0, 0], [0, 10]],
                    tick_direction=(-1, 0),
                    axis_label='Z-axis',
                    axis_font_size=12,
                    major_tick_length=10,
                    tick_width=2,
                    text_color='white',
                    axis_label_margin=30)

# Position and transform the axes to align with the plot
axis_x.transform = scene.transforms.STTransform(translate=(0, -12, 0))
axis_y.transform = scene.transforms.STTransform(translate=(-12, 0, 0))
axis_z.transform = scene.transforms.STTransform(translate=(-12, 0, 0))

# Add the axes to the view
view.add(axis_x)
view.add(axis_y)
view.add(axis_z)

# Run the application
canvas.app.run()