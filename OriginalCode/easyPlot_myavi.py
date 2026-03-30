# -*- coding: utf-8 -*-
"""
Plotting package for matlab-like synthax within the pyqtgraph library

@author: Eugeniu Grabovic
"""

from mayavi import mlab
import numpy as np
import sys
import screwCalculus as sc
import time
from graphical_primitives import create_parallelepiped, createCylinder, apply_homogeneous_transform, patch_points
import pyvista as pv
from pyvistaqt import BackgroundPlotter


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

    def __init__(self, mlab, title = 'Title', nlabels = 20) -> None:
        self.app = mlab
        self.figure = mlab.figure(title, bgcolor=(1,1,1), fgcolor=(0,0,0), size = [1600, 1400])
        # Add a dummy object to the scene (e.g., a line or a point)
        self.origin = mlab.points3d(0, 0, 0, line_width = 0.0001, mode = 'point')  # Create a dummy point at the origin
        self.axes = mlab.axes(xlabel='X', ylabel='Y', zlabel='Z', figure = self.figure, nb_labels = nlabels,
                               extent = [-1, 1,-1, 1,-1, 1])

    def updateImage(self):
        for i in range(0, self.nObjects):
            self.container[i].applyTransform()
        self.app.draw(self.figure)

    def addPlotObject(self, object):
        self.container.append(object)
        self.nObjects += 1

    def removeGraphics(self, object):
        self.figure.removeItem(object.GraphicsItem)
        self.container.remove(object)
        self.nObjects -= 1

    # def fontSize(self, sz):
    #     self.axes.axes.font_factor = sz

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
            self.parent.addChildren(self)
        return

    def updateData(self, X, Y, Z):
        """
        manually update x, y, z data of a graphical object
        """
        if isinstance(self.object, list):
            for i in range(0, len(self.object)):
                self.object[i].object.mlab_source.set(x=X[i], y=Y[i], z=Z[i])
            
            return
        self.object.mlab_source.set(x=X, y=Y, z=Z)

    def getData(self):
        return self.XData, self.YData, self.ZData

    def applyTransform(self):
        T_global = self.currentGlobal
        X, Y, Z = self.getData()
        X, Y, Z = apply_homogeneous_transform(X, Y, Z, T_global)
        self.updateData(X, Y, Z)

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
    def __init__(self, fig, X = None, Y = None, Z = None, parent = None, color = (0.6, 0.3, 0.9), show = True) -> None:
        super().__init__(fig, parent)
        self.XData = X
        self.YData = Y
        self.ZData = Z
        self.description = 'surface'
        self.object = fig.app.mesh(X, Y, Z, color=color, figure = fig.figure,
            line_width = 1,
            tube_radius = None,
            mode = 'sphere',
            representation = 'surface')
        self.figure.addPlotObject(self)
        if show:
            self.figure.app.show()

class patch(go):
    def __init__(self, fig, X = None, Y = None, Z = None, parent = None, color = (0.6, 0.3, 0.9), show = True) -> None:
        super().__init__(fig, parent)
        X, Y, Z, triangles = patch_points(X, Y, Z)
        self.XData = X
        self.YData = Y
        self.ZData = Z
        self.triangles = triangles
        self.description = 'patch'
        self.object= fig.app.triangular_mesh(X, Y, Z, triangles, color=color, figure = fig.figure,
            line_width = 1,
            tube_radius = None,
            mode = 'sphere',
            representation = 'surface')
        self.figure.addPlotObject(self)
        if show:
            self.figure.app.show()
        return

class line(go):

    def __init__(self, fig, x, y, z = None, color = (0,0,0), lineWidth = 5, parent = None, show = True) -> None:
        super().__init__(fig, parent)
        self.XData = x
        self.YData = y
        if z is None:
            z = x*0
        self.ZData = z
        self.object = fig.app.plot3d(x, y, z, color = color, line_width = lineWidth, figure = fig.figure, tube_radius = None)
        self.figure.addPlotObject(self)
        if show:
            self.figure.axes.axes.bounds = self.object.actor.actor.bounds
            self.figure.app.show()
            

class revolute_joint(go):
    """
    revolute joint object.
    """
    def __init__(self, fig, r, ax, lenTop, lenBot, axialOff = 0, parent = None, color = (0.6, 0.3, 0.9), show = True, N = 15) -> None:
        super().__init__(fig, parent)
        X, Y, Z, _, _ = createCylinder(r, ax, lenTop, lenBot, axialOff, N=N)
        self.XData = []; self.YData = []; self.ZData = []
        
        # main cylinder body
        self.object.append(surface(fig, X = X, Y = Y, Z = Z, parent = None, color = (0.6, 0.3, 0.9), show = False))
        self.XData.append(X); self.YData.append(Y); self.ZData.append(Z)

        # top patch
        X = X[0,:]; Y = Y[0,:]; Z = Z[0,:]
        self.object.append(patch(fig, X = X, Y = Y, Z = Z, parent = None, color = (0.6, 0.3, 0.9), show = False))
        X, Y, Z = self.object[1].getData()
        self.XData.append(X); self.YData.append(Y); self.ZData.append(Z)

        # bottom patch
        X = self.XData[0][1,:]; Y = self.YData[0][1,:]; Z = self.ZData[0][1,:]
        self.object.append(patch(fig, X = X, Y = Y, Z = Z, parent = None, color = (0.6, 0.3, 0.9), show = False))
        X, Y, Z = self.object[2].getData()
        self.XData.append(X); self.YData.append(Y); self.ZData.append(Z)

        self.description = 'revolute_joint'
        self.figure.addPlotObject(self)
        if show:
            self.figure.app.show()

class prismatic_joint(revolute_joint):
    """
    prismatic joint
    """
    def __init__(self, fig, r, ax, lenTop, lenBot, axialOff = 0, parent = None, color = (0.6, 0.3, 0.9), show = True) -> None:
        super().__init__(fig, r, ax, lenTop, lenBot, axialOff = 0, parent = None, color = (0.6, 0.3, 0.9), show = show, N = 5)

#=====================================================================================================================
#
# TUTORIAL/DEBUGGING MAINs
#
#=====================================================================================================================

def main():

    pX, pY, pZ, vertices, faces = createCylinder(0.5, np.array([0,0,1]), 1, 1, 0)
    x = np.array([0, 1, 2, 5])
    y = np.array([0, 1, 3, 8])
    z = np.array([0, 1, 1, 0])

    F = Figure() # init figure
    line1 = line(F, x, y, z, color = 'red')
    x = np.linspace(-4,4,100)
    y = np.linspace(-4,4,100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X)  + np.cos(Y)
    # surf = surface(F, vert=vertices, faces=faces, parent = line1)
    
    surf = surface(F, X = x, Y = y, Z = Z)
    

    def animationFun(objs, iter, userData):
        q = userData(iter)
        objs[0].applyTransform(sc.TrotY(q))
        objs[1].applyTransform(sc.TrotX(iter*0.1))

    userData =  lambda x: x/10

    F.setAnimFun(animationFun, userData) # set custom animation  function
    F.animation()              # start animation

    for i in range(0,200):
        surf.applyTransform(sc.TrotY(i*0.1))
        line1.applyTransform(np.eye(4))
        F.updateImage()
    
    F.start()                  # start widget

def main2():
    from mayavi import mlab
    import numpy as np
    import time

    # Create a grid for the mesh (initial parametric surface)
    x, y = np.mgrid[-3:3:50j, -3:3:50j]
    z = np.sin(x**2 + y**2)

    # Plot the initial mesh
    mesh = mlab.mesh(x, y, z, colormap="cool")

    # Function to animate the mesh dynamically
    def animate_mesh():
        for t in np.linspace(0, 10 * np.pi, 200):  # Time variable for animation
            # Update Z coordinates (e.g., creating a wave pattern over time)
            z_new = np.sin(x**2 + y**2 + t)  # Modify Z based on time
            
            # Update the mesh with new Z values
            mesh.mlab_source.set(z=z_new)
            
            # Process the UI events to refresh the scene and display the changes
            mlab.process_ui_events()
            
            # Control the speed of the animation (small pause)
            time.sleep(0.01)

    # Show the initial plot and keep the window open for interaction
    mlab.show(stop=True)

    # Start the animation
    animate_mesh()

def main_myavi_tutorial():
    import numpy as np
    from mayavi import mlab

    @mlab.animate(delay = 100)
    def updateAnimation():
        t = 0.0
        while True:
            ball.mlab_source.set(x = np.cos(t), y = np.sin(t), z = 0)
            t += 0.1
            yield

    ball = mlab.points3d(np.array(1.), np.array(0.), np.array(0.))

    updateAnimation()
    mlab.show()

def main_debug():
    from mayavi import mlab
    F = Figure(mlab)

    x = np.linspace(-np.pi, np.pi, 100)
    y = np.sin(x)
    l1 = line(F, x, y, show = True)


if __name__ == "__main__":
    # main()
    # main2()
    # main_myavi_tutorial()
    main_debug()
