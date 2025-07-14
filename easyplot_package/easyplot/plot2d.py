# easyplot/plot2d.py

import matplotlib.pyplot as plt

def line(x, y, ax=None, **kwargs):
    """
    Plot a 2D line plot similar to MATLAB's `line` function.

    Parameters:
        x : array-like
            X-coordinates.
        y : array-like
            Y-coordinates.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        **kwargs: Additional keyword arguments passed to ax.plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    line_obj, = ax.plot(x, y, **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Line Plot')
    ax.grid(True)
    return line_obj  # Return the artist for possible grouping/transformation


def scatter(x, y, ax=None, **kwargs):
    """
    Plot a 2D scatter plot similar to MATLAB's `scatter` function.

    Parameters:
        x : array-like
            X-coordinates.
        y : array-like
            Y-coordinates.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        **kwargs: Additional keyword arguments passed to ax.scatter.
    """
    if ax is None:
        fig, ax = plt.subplots()
    scatter_obj = ax.scatter(x, y, **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Scatter Plot')
    ax.grid(True)
    return scatter_obj  # Return the artist for possible grouping/transformation
