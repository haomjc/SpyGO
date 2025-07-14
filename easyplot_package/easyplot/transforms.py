# easyplot/transforms.py

import numpy as np

# -----------------------------
# 2D Transformation (Matplotlib)
# -----------------------------
import matplotlib.transforms as mtransforms

class HgTransform2D:
    """
    A class to group multiple 2D Matplotlib artists so that an affine 
    transformation can be applied to all of them simultaneously.
    """
    def __init__(self, ax):
        """
        Parameters:
            ax : matplotlib.axes.Axes
                The axes containing the artists.
        """
        self.ax = ax
        self.transform = mtransforms.Affine2D()  # initial identity transform
        self.artists = []
        self.original_transforms = {}

    def add(self, artist):
        """
        Add a Matplotlib artist to the transformation group.
        
        Parameters:
            artist: A Matplotlib artist (e.g., Line2D, PathCollection, etc.)
        """
        # Store the original transform for future updates.
        self.original_transforms[artist] = artist.get_transform()
        self.artists.append(artist)
        # Immediately update the artist's transform.
        artist.set_transform(self.transform + self.original_transforms[artist])

    def set_transform(self, matrix):
        """
        Update the transformation applied to all grouped artists.
        
        Parameters:
            matrix: A 3x3 numpy array representing an affine transformation.
                    (Last row is typically [0, 0, 1].)
        """
        self.transform = mtransforms.Affine2D(matrix)
        for artist in self.artists:
            orig = self.original_transforms[artist]
            artist.set_transform(self.transform + orig)
        # Redraw the canvas.
        self.ax.figure.canvas.draw_idle()


# -----------------------------
# 3D Transformation (PyVista)
# -----------------------------
import pyvista as pv
import vtk

class HgTransform3D:
    """
    A class to group multiple PyVista mesh objects so that an affine 
    transformation can be applied to all of them simultaneously.
    
    Note: PyVista does not have a built-in grouping transform, so this class
    applies the transformation to a stored copy of each mesh.
    """
    def __init__(self, plotter):
        """
        Parameters:
            plotter : pyvista.Plotter
                The plotter in which the meshes are rendered.
        """
        self.plotter = plotter
        self.transform = vtk.vtkTransform()  # initial identity transform
        self.meshes = []
        self.original_meshes = {}

    def add(self, mesh):
        """
        Add a PyVista mesh to the transformation group.
        
        Parameters:
            mesh : A PyVista mesh (e.g., StructuredGrid, PolyData, etc.)
        """
        self.meshes.append(mesh)
        # Keep a copy of the original mesh for re-transformations.
        self.original_meshes[id(mesh)] = mesh.copy()

    def set_transform(self, matrix):
        """
        Update the transformation applied to all grouped meshes.
        
        Parameters:
            matrix: A 4x4 numpy array representing an affine transformation.
        """
        # Reset and update the vtkTransform using the provided matrix.
        self.transform.Identity()
        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, matrix[i][j])
        self.transform.SetMatrix(vtk_matrix)
        
        # For each mesh, apply the transformation to its original copy.
        for i, mesh in enumerate(self.meshes):
            orig_mesh = self.original_meshes.get(id(mesh))
            if orig_mesh is not None:
                new_mesh = orig_mesh.copy()
                new_mesh.transform(self.transform)
                # Remove the old mesh actor and add the new one.
                self.plotter.remove_actor(mesh)
                self.plotter.add_mesh(new_mesh)
                # Update the stored mesh and its original copy.
                self.meshes[i] = new_mesh
                self.original_meshes[id(new_mesh)] = orig_mesh
        self.plotter.render()
