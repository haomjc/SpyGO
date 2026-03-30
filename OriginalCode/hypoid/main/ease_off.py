import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from mpl_toolkits.mplot3d import Axes3D

def set_data_aspect_ratio(ax, aspect_ratio):
    """
    Sets the aspect ratio of a 3D plot similar to MATLAB's `set(gca, 'DataAspectRatio', [x, y, z])`.

    Parameters:
    ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis object.
    aspect_ratio (tuple or list of 3 values): Desired aspect ratio for x, y, and z axes.
    """

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # Compute the center of each axis
    xcenter = np.mean(xlim)
    ycenter = np.mean(ylim)
    zcenter = np.mean(zlim)

    # Compute the full range of each axis
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    zrange = zlim[1] - zlim[0]

    # Scale the ranges according to the desired aspect ratio
    max_range = max(xrange / aspect_ratio[0], yrange / aspect_ratio[1], zrange / aspect_ratio[2])
    
    new_xrange = max_range * aspect_ratio[0]
    new_yrange = max_range * aspect_ratio[1]
    new_zrange = max_range * aspect_ratio[2]

    # Set new axis limits while keeping the center fixed
    ax.set_xlim([xcenter - new_xrange / 2, xcenter + new_xrange / 2])
    ax.set_ylim([ycenter - new_yrange / 2, ycenter + new_yrange / 2])
    ax.set_zlim([zcenter - new_zrange / 2, zcenter + new_zrange / 2])

    return 

def ease_off_9DoF(v9DoF):
    """
    Vectorized 9 DoF ease-off function.
    
    Parameters
    ----------
    v9DoF : array_like of shape (9,)
        Coefficients in R^9.
    
    Returns
    -------
    E9DoF : callable
        Function (r, s) -> array, vectorized over r,s.
    """
    v9DoF = np.asarray(v9DoF, dtype=float)
    def E9DoF(r, s):
        r = np.asarray(r, dtype=float)
        s = np.asarray(s, dtype=float)

        # bilinear
        h1 = 0.25 * (1 + r) * (1 + s)
        h2 = 0.25 * (1 - r) * (1 + s)
        h3 = 0.25 * (1 - r) * (1 - s)
        h4 = 0.25 * (1 + r) * (1 - s)

        # cubic
        h5 = 0.5 * (1 - r**2) * (1 + s)
        h6 = 0.5 * (1 - s**2) * (1 - r)
        h7 = 0.5 * (1 - r**2) * (1 - s)
        h8 = 0.5 * (1 - s**2) * (1 + r)

        # quartic
        h9 = (1 - r**2) * (1 - s**2)

        # midpoints
        H5 = h5 - 0.5 * h9
        H6 = h6 - 0.5 * h9
        H7 = h7 - 0.5 * h9
        H8 = h8 - 0.5 * h9
        H9 = h9

        # vertices
        H1 = h1 - 0.5 * H5 - 0.5 * H8 - 0.25 * H9
        H2 = h2 - 0.5 * H5 - 0.5 * H6 - 0.25 * H9
        H3 = h3 - 0.5 * H6 - 0.5 * H7 - 0.25 * H9
        H4 = h4 - 0.5 * H7 - 0.5 * H8 - 0.25 * H9

        # linear combination (vectorized dot product)
        H = np.stack([H1, H2, H3, H4, H5, H6, H7, H8, H9], axis=0)

        # if isinstance(v9DoF, ca.SX):
        #     out = H1*v9DoF[0] + H2*v9DoF[1] + H3*v9DoF[2]+\
        #     H4*v9DoF[3] + H5*v9DoF[4] + H6*v9DoF[5]+\
        #     H7*v9DoF[6] + H8*v9DoF[7] + H9*v9DoF[8]
        #     return out
        out = np.tensordot(v9DoF, H, axes=(0, 0))
        return out

    return E9DoF

def ease_off_5DoF(v5DoF):
    
    PA = v5DoF[0] # pressure angle mod.
    SA = v5DoF[1] # pressure angle mod.
    PC = v5DoF[2] # pressure angle mod.
    LC = v5DoF[3] # pressure angle mod.
    TW = v5DoF[4] # pressure angle mod.

    def E5DoF(u, v):
        return PA*v + SA*u + PC*v**2 + LC*u**2 + TW*u*v
    
    return E5DoF

def ease_off_fillet(E_discrete, n_div_fillet, deg = 0.7):
    u_num = np.linspace(0, 1, n_div_fillet+1).reshape(-1, 1)
    E_flank_fillet = E_discrete[0,:].reshape(1, -1)
    u_num = u_num[0:-1]
    E_fillet = E_flank_fillet*u_num**deg
    return np.vstack((E_fillet, E_discrete))

def compute_ease_off(vDoF, n_prof = 11, n_face = 22, n_fillet = 6):
    """
    Ease-off function with a continuous (non-smooth) transition towards the rootline, where no material should be removed
    The z-R grid shall be constructed accordingly outside the funciton
    """
    vDoF = np.array(vDoF)
    if max(vDoF.shape) == 5:
        E_fun = ease_off_5DoF(vDoF)
    else:
        E_fun = ease_off_9DoF(vDoF)

    U, V = np.meshgrid(np.linspace(-1, 1, n_prof), np.linspace(-1, 1, n_face))

    E_num = E_fun(U, V)
    E_num = ease_off_fillet(E_num, n_fillet)
    return E_num

def plot_ease_off(E_fun, number_U = 50, number_V = 50, labels = ['u', 'v', 'E [$\mu$m]'], aspect_ratio = [1,1,1]):

    if callable(E_fun):
        U, V = np.meshgrid(np.linspace(-1, 1, number_U), np.linspace(-1, 1, number_V))
        E = E_fun(U, V)

    else:
        U, V = number_U, number_V
        E = E_fun

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    red_color = np.array([1, 0, 0, 1]) 
    surf = ax.plot_surface(U, V, E, cmap='viridis', edgecolor = (0.3, 0.3, 0.3), facecolors = np.tile(red_color, U.shape + (1,)) ) 
    # surf_base = ax.plot_surface(U, V, E*0, edgecolor = 'k', facecolors=np.zeros(U.shape + (4,)))
    plt.contour(U, V, E, levels=10, linewidth = 4, offset = 0)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    # 
    # ax.set_aspect('equal', adjustable='box', anchor='C')
    set_data_aspect_ratio(ax, aspect_ratio)
    # Show the plot
    plt.show()

    return

def main():

    U, V = np.meshgrid(np.linspace(-1, 1, 22), np.linspace(-1, 1, 11))
    # u -> x-axis (along the facewidth direction)
    # v -> y-axis (along the profile direction)
    
    # v5dof_vec = np.array([0, -100, 50, 25, 0])/1000
    # E_fun = ease_off_5DoF(v5dof_vec)

    v9_dof = np.array([50,50,50,50,50,50,50,50,0])/1000
    E_fun = ease_off_9DoF(v9_dof)

    E = E_fun(U, V)
    plot_ease_off(E, U, V, aspect_ratio=[1,1,0.1])

    return

if __name__ == '__main__':
    main()