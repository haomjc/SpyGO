# -*- coding: utf-8 -*-
"""
Plotting package for matlab-like synthax within the pyvista library

@author: Eugeniu Grabovic
"""

import numpy as np
import sys
import screwCalculus as sc
import time
from graphical_primitives import create_parallelepiped, createCylinder, apply_homogeneous_transform, patch_points
import pyvista as pv
import pyvistaqt as pvq

# function which checks if the code is ran by an interactive jupyter notebook or a simple terminal script
def is_notebook():
    try:
        # Check if running in an IPython environment and if it's using the notebook frontend
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except:
        return False
    return False

# Figure container
class Figure:
    
    app = []
    figure = []
    container = []
    nObjects = 0
    parent = []
    animFun = None
    animIter = 0
    animData = None

    def __init__(self, title = 'Title') -> None:
        self.app = pv
        self.figure = pvq.BackgroundPlotter(title = title)
        self.axes = self.figure.show_axes()
        self.grid = self.figure.show_grid()
        self.bounds = self.figure.show_bounds(grid='back', location='outer', ticks='both', all_edges=True)
        if not is_notebook():
            if isinstance(self.figure, pv.Plotter):
                self.figure.show(jupyter_backend='static')
            else:
                self.figure.show()


    def updateImage(self):
        for i in range(0, self.nObjects):
            self.container[i].applyTransform()
        self.figure.show_bounds(grid='back', location='outer', ticks='both', all_edges=True)
        self.figure.reset_camera()
        self.figure.render()

    def addPlotObject(self, object):
        self.container.append(object)
        self.nObjects += 1

    def removeGraphics(self, object):
        self.figure.removeItem(object.GraphicsItem)
        self.container.remove(object)
        self.nObjects -= 1

    def xlabel(self, label):
        self.figure.add_text(label, position='lower_edge', font_size=10, color='black')

    def ylabel(self, label):
        self.figure.add_text(label, position='left_edge', font_size=10, color='black')

    def show(self, planar = False):
        if is_notebook():
            if planar:
                self.figure.view_xy()
            
            if isinstance(self.figure, pv.Plotter):
                self.figure.show(jupyter_backend='static')
            else:
                self.figure.show()

        else:
            if planar:
                self.figure.view_xy()
            if isinstance(self.figure, pv.Plotter):
                self.figure.show(interactive=True, auto_close=False)
            else:
                self.figure.show()

    def set_scale(self, x_scale, y_scale, z_scale):
        self.figure.set_scale(x_scale, y_scale, z_scale)
        
class go():  # graphical object container

    def __init__(self, fig, parent = None):
        self.parent = parent
        self.children = []
        self.currentTransform = np.eye(4)
        self.currentGlobal = np.eye(4)
        self.parentTransform = np.eye(4)
        self.figure = fig
        self.object = []
        if parent is not None:
            self.parent.addChildren(self) # add this object to the parent's children list
            self.setTransform(parent.currentGlobal) # set the transform to the parent's global transform
        return

    def updateData(self, points):
        """
        manually update x, y, z data of a graphical object
        """
        if isinstance(self.object, list):
            for i in range(0, len(self.object)):
                self.object[i].updateData(points[i])
            return
        self.object.points = points

    def getData(self):

        if isinstance(self.object, list):
            points = []
            for i in range(0, len(self.object)):
                points.append(self.object[i].getData())
            return points
        
        points = self.points
        return points

    def applyTransform(self):
        T_global = self.currentGlobal
        points = self.getData()
        points = apply_homogeneous_transform(points, T_global)
        self.updateData(points)

    def setTransform(self, T = None):
        """
        T = homogeneous matrix transform
        """
        if T is None:
            T = self.currentTransform

        T_global = self.parentTransform@T
        self.currentTransform = T # keep track of the current (local) transform
        self.currentGlobal = T_global
        if len(self.children)>0: # update the transform to the childrens
            for i in range(0, len(self.children)):
                self.children[i].parentTransform = T_global
                self.children[i].setTransform()

    def addChildren(self, object):
        self.children.append(object)
        object.parent = self

    def addParent(self, object):
        self.parent = object
        object.children.append(self)

class surface(go):
    """
    surface graphical object.
    """
    def __init__(self, fig, X = None, Y = None, Z = None, parent = None, color = 'blue', show_edges = False, edge_color = 'black', face_color = 'white') -> None:
        super().__init__(fig, parent)
        self.description = 'surface'
        
        self.object = fig.app.StructuredGrid(X, Y, Z)
        fig.figure.add_mesh(self.object,
                    show_edges = show_edges,
                    edge_color = edge_color,
                    color = face_color)
                #      scalar_bar_args={
                #      "title": "Custom Data",  # Title of the color bar
                #      "vertical": True,        # Vertical orientation of the color bar
                #      "title_font_size": 10,   # Font size of the title
                #      "label_font_size": 8,    # Font size of the labels
                #  }
        self.points = self.object.points
        self.figure.addPlotObject(self)

class patch(go):
    def __init__(self, fig, X = None, Y = None, Z = None, parent = None, color = 'blue', show_edges = False, edge_color = 'black', face_color = 'white') -> None:
        super().__init__(fig, parent)
        x, y, z, triangles, faces, points = patch_points(X, Y, Z)
        self.faces = faces
        self.description = 'patch'
        self.object= fig.app.PolyData(points, faces)
        fig.figure.add_mesh(self.object,
            show_edges = show_edges,
            edge_color = edge_color,
            color = face_color)
        self.figure.addPlotObject(self)
        self.points = self.object.points
        return

class line(go):

    def __init__(self, fig, x, y, z = None, color = 'black', line_width = 10, parent = None) -> None:
        super().__init__(fig, parent)
        self.XData = x
        self.YData = y
        if z is None:
            z = x*0
        self.ZData = z
        n_points = len(x)
        points = np.column_stack([x, y, z])
        lines = np.column_stack((np.full(n_points-1, 2), np.arange(n_points-1), np.arange(1, n_points))).flatten()
        self.object = fig.app.PolyData(points)
        self.object.lines = lines
        fig.figure.add_mesh(self.object,
            color = color,
            line_width = line_width,
        )
        self.figure.addPlotObject(self)
        self.points = self.object.points # transpose to have the same format as the other objects
            
class revolute_joint(go):
    """
    revolute joint object.
    """
    def __init__(self, fig, r, ax, lenTop, lenBot, axialOff = 0, parent = None, color = 'blue', show_edges = False, edge_color = 'black', face_color = 'white', N = 15) -> None:
        super().__init__(fig, parent)
        self.points = []
        X, Y, Z, _, _ = createCylinder(r, ax, lenTop, lenBot, axialOff, N=N)
        # main cylinder body
        self.object.append(surface(fig, X = X, Y = Y, Z = Z, parent = None, color = color, show_edges = show_edges, edge_color = edge_color, face_color = face_color))
        self.points.append(self.object[0].getData())

        # top patch
        x = X[0,:]; y = Y[0,:]; z = Z[0,:]
        self.object.append(patch(fig, X = x, Y = y, Z = z, parent = None, color = color, show_edges = show_edges, edge_color = edge_color, face_color = face_color))
        self.points.append(self.object[1].getData())

        # bottom patch
        x = X[1,:]; y = Y[1,:]; z = Z[1,:]
        self.object.append(patch(fig, X = x, Y = y, Z = z, parent = None, color = color, show_edges = show_edges, edge_color = edge_color, face_color = face_color))
        self.points.append(self.object[2].getData())

        self.description = 'revolute_joint'
        self.figure.addPlotObject(self)

class prismatic_joint(revolute_joint):
    """
    prismatic joint
    """
    def __init__(self, fig, r, ax, lenTop, lenBot, axialOff = 0, parent = None, color = 'blue', show_edges = False, edge_color = 'black', face_color = 'white') -> None:
        super().__init__(fig, r, ax, lenTop, lenBot, axialOff = 0, parent = None, color = color, show_edges = show_edges, edge_color = edge_color, face_color = face_color, N=5)

#=====================================================================================================================
#
# TUTORIAL/DEBUGGING MAINs
#
#=====================================================================================================================

def main():

    X, Y, Z, vertices, faces = createCylinder(0.5, np.array([0,0,1]), 1, 1, 0)
    F = Figure()

    

    S = surface(F, X,Y,Z)
    X = X[0,:]; Y = Y[0,:]; Z = Z[0,:]
    P = patch(F, X,Y,Z)
    rev = revolute_joint(F, 0.5, np.array([0,1,0]), 1, 1, 0)
    rev.setTransform(sc.TtX(5)@sc.TtY(5))
    prism = prismatic_joint(F, 0.5, np.array([1,1,1]), 1, 1, 0, parent = rev)
    L = line(F, np.array([1.0,2,3,4]), np.array([0, 1.0, 8,1]), parent = rev)
    

    print(L.currentGlobal)
    F.updateImage()
    F.show()               # start widget

if __name__ == "__main__":
    main()


def main2():
    import pyvista as pv
    import numpy as np

    # Create a plotter and add an initial mesh (e.g., a sphere)
    plotter = pv.Plotter()
    mesh = pv.Sphere(radius=0.5)
    plotter.add_mesh(mesh, show_edges=True, color='lightblue')
    plotter.show_grid()
    
    

    # Modify the mesh (e.g., apply a translation)
    mesh.translate([2, 2, 0])

    # Add the modified mesh to the plotter (or just modify the existing mesh)
    # plotter.add_mesh(mesh, show_edges=True, color='lightblue')

    # Update the axis range to fit the new mesh extents
    plotter.reset_camera()  # Resets the camera and updates the bounds

    # Show the updated plot
    plotter.render()  # Manually render the plot again if necessary
    # Display the initial plot
    plotter.show(auto_close=False)  # Keep the plot open

def main_pyvista_tutorial():
    import pyvista as pv
    import numpy as np
    import time

    # Create a plotter object
    plotter = pv.Plotter()

    # Create a sphere and add it to the plotter
    sphere = pv.Sphere()
    actor = plotter.add_mesh(sphere, color='cyan')

    # Define number of frames for the animation
    n_frames = 100

    # Show the plotter in interactive mode
    plotter.show(auto_close=True)

    # Loop over frames to rotate the sphere in real-time
    for i in range(n_frames):
        # Update the rotation of the sphere by rotating around the Z-axis
        actor.rotate_x(360*100 / n_frames)
        
        # Render the scene to update the display
        plotter.render()
        # Add a short delay to slow down the animation (optional)
        plotter.sleep(0.05)

    # Keep the window open after the animation completes
    plotter.close()
