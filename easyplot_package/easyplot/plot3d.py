# easyplot/plot3d.py

import pyvista as pv
import numpy as np

def surface(X, Y, Z, plotter=None, **kwargs):
    """
    Plot a 3D surface similar to MATLAB's `surface` function.
    
    Parameters:
        X, Y, Z : 2D array-like
            Grid coordinates (all must have the same shape).
        plotter : pyvista.Plotter, optional
            The plotter to use. If None, a new Plotter is created.
        **kwargs: Additional keyword arguments passed to add_mesh.
    Returns:
        mesh: The PyVista mesh that was added.
    """
    grid = pv.StructuredGrid(X, Y, Z)
    if plotter is None:
        plotter = pv.Plotter()
    actor = plotter.add_mesh(grid, **kwargs)
    plotter.add_axes()
    return grid  # Return the mesh for further manipulation


def patch(vertices, faces, plotter=None, **kwargs):
    """
    Plot a 3D patch (polygon) similar to MATLAB's `patch` function.
    
    Parameters:
        vertices : (n_points, 3) array-like
            Coordinates of the vertices.
        faces : list or array
            Face connectivity in PyVista's format. For a single polygon, it should be:
            [n, i0, i1, ..., i(n-1)] where `n` is the number of points.
        plotter : pyvista.Plotter, optional
            The plotter to use. If None, a new Plotter is created.
        **kwargs: Additional keyword arguments passed to add_mesh.
    Returns:
        mesh: The PyVista mesh that was added.
    """
    mesh = pv.PolyData(vertices, faces)
    if plotter is None:
        plotter = pv.Plotter()
    actor = plotter.add_mesh(mesh, **kwargs)
    plotter.add_axes()
    return mesh  # Return the mesh for further manipulation
