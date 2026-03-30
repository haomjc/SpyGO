import numpy as np
import screwCalculus as sc

def patch_points(x, y, z):

    n_points = len(x)
    xG = np.sum(x)/n_points
    yG = np.sum(y)/n_points
    zG = np.sum(z)/n_points

    x = np.concatenate(([xG], x))
    y = np.concatenate(([yG], y))
    z = np.concatenate(([zG], z))
    # Create the triangular mesh
    triangles = [[i, (i + 1) % n_points, n_points] for i in range(n_points)]
    faces = np.hstack([np.array([3] * n_points)[:, None], triangles])
    points = np.vstack((x.T, y.T, z.T)).T

    return x, y, z, triangles, faces, points

def createCylinder(r, ax, lenTop, lenBot, axialOff, N=15):
    """
    r = radius
    ax = axis direction (3x1 array)
    lenTop = axial length along positive ax direction
    lenBot = axial length along negative ax direction (note it is a POSITIVE value)
    axialOff = offset along axis (lenTop and lenBot refer to this new origin translated along axis)
    """

    theta = np.linspace(0, 2*np.pi, N).reshape(1, N)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    zTop = lenTop*np.ones((1, N))
    zBot = -lenBot*np.ones((1, N))
    ptTop = np.concatenate((x, y, zTop), axis = 0)
    ptBot = np.concatenate((x, y, zBot), axis = 0)

    ax = ax/np.linalg.norm(ax)
    theta = np.arccos(ax[2])
    phi = np.arctan2(ax[0], ax[1])
    R = sc.rotZ(-phi)@sc.rotX(-theta)
    ptTop = R@ptTop
    ptBot = R@ptBot
    ptX = np.concatenate((ptTop[0, :].reshape(1, N), ptBot[0, :].reshape(1, N)), axis = 0)
    ptY = np.concatenate((ptTop[1, :].reshape(1, N), ptBot[1, :].reshape(1, N)), axis = 0)
    ptZ = np.concatenate((ptTop[2, :].reshape(1, N), ptBot[2, :].reshape(1, N)), axis = 0) - axialOff

    # vertices and faces for the lateral surface
    X = np.expand_dims(
        np.array([ptBot[0, 0:N-1], ptBot[0, 1:N], ptTop[0, 0:N-1], ptTop[0, 1:N]]).T.flatten(),
        axis = 1)
    Y = np.expand_dims(
        np.array([ptBot[1, 0:N-1], ptBot[1, 1:N], ptTop[1, 0:N-1], ptTop[1, 1:N]]).T.flatten(),
        axis = 1)
    Z = np.expand_dims(
        np.array([ptBot[2, 0:N-1], ptBot[2, 1:N], ptTop[2, 0:N-1], ptTop[2, 1:N]]).T.flatten(),
        axis = 1) - axialOff

    
    vertices = np.concatenate((X,Y,Z), axis = 1)
    faces = np.empty(((N-1)*2, 3), dtype=int)

    for k in range(0, N-1):
        j = k*4
        ii = k*2
        faces[ii  , :] = np.array([j, j+1, j+2], dtype=int)
        faces[ii+1, :] = np.array([j+1, j+3, j+2], dtype=int)

    return (ptX, ptY, ptZ, vertices, faces) #, verticesTop, verticesBot, faces, faces

def create_parallelepiped(r, ax, lenTop, lenBot, axialOff):
    """
    It is just a cylinder with 5 sampling points :) (4 vertices) 5-th point coincides with the first to close the loop"""
    ptX, ptY, ptZ, vertices, faces = createCylinder(r, ax, lenTop, lenBot, axialOff, N=5)
    return (ptX, ptY, ptZ, vertices, faces)

def apply_homogeneous_transform(points, transform_matrix):

    # Flatten the arrays and add homogeneous coordinate (w = 1) to each point
    if isinstance(points, list):

        points_new = [0]*len(points)

        for i in range(0, len(points)):
            points_new[i] = apply_homogeneous_transform(points[i], transform_matrix)

        return points_new
    
    shp = points.shape
    if shp[1]<shp[0]:
        points = points.T

    points = np.vstack([points, np.ones((1, max(shp)))])

    # Apply the transformation matrix (4x4) to the points (4xN)
    transformed_points = transform_matrix @ points
    
    # Reshape the transformed points back to the original shape of the mesh
    points_new = transformed_points[0:3,:].T
    
    return points_new