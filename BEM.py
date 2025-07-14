import numpy as np
from math import pi
from math import log, sqrt
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from qpsolvers import solve_qp
import time
from gams import GamsWorkspace
import mplcursors
from general_utils import set_data_aspect_ratio, s_runge_map
from  scipy.sparse import csr_matrix


import numpy as np


def compute_surface_area(n, hx, hy):

    nz = n[2,:]
    nz[nz < 0.2] = 0.2
    nz = 1/(n[2,:])
    
    
    deltaX = hx*np.sqrt(n[0,:]**2 + nz**2)*nz
    deltaY = hy*np.sqrt(n[1,:]**2 + nz**2)*nz
    deltaX = hx/(np.abs(np.sqrt(1 - (n[0,:]**2))) + 1e-2)
    deltaY = hy/(np.abs(np.sqrt(1 - (n[1,:]**2))) + 1e-2)
    dA = deltaX*deltaY
    total_area = np.sum(dA)
    return dA, total_area, deltaX, deltaY

def compliance_uniform_rectangular_pressure(iprime, jprime, i, j, hx, hy):
    """
    z direction compliance on point (i,j) for a uniform rectangular pressure distribution on (iprime,jprime)
    """
    deltai=iprime-i
    deltaj=jprime-j
    k=(deltai+0.5)*hx
    m=(deltaj+0.5)*hy
    l=(deltai-0.5)*hx
    n=(deltaj-0.5)*hy

    k2m2 = k**2+m**2
    l2n2 = l**2+n**2
    k2n2 = k**2+n**2
    l2m2 = l**2+m**2
    sk2n2 = np.sqrt(k2n2)
    sl2m2 = np.sqrt(l2m2)
    sk2m2 = np.sqrt(k2m2)
    sl2n2 = np.sqrt(l2n2)

    K_zz = (k*np.log((m+sk2m2)/(n+sk2n2))\
              +l*np.log((n+sl2n2)/(m+sl2m2))\
                +m*np.log((k+sk2m2)/(l+sl2m2))\
                    +n*np.log((l+sl2n2)/(k+sk2n2)))
    
    term1 = m*np.log((k2m2)/(l2m2))
    term2 = n*np.log((l2n2)/(k2n2))

    term3 = k*(np.arctan(m/k) - np.arctan(n/k))
    term4 = l*(np.arctan(n/l) - np.arctan(m/l))

    K_zx = 0.5*(term1 + term2) + term3 + term4

    term1 = k*np.log((k2m2)/(k2n2))
    term2 = l*np.log((l2n2)/(l2m2))

    term3 = m*(np.arctan(k/m) - np.arctan(l/m))
    term4 = n*(np.arctan(l/n) - np.arctan(k/n))

    K_zy = 0.5*(term1 + term2) + term3 + term4
    
    
    return K_zz, K_zx, K_zy

def simulate_bem(surface1, surface2, load, material_properties, mesh, use_normals = True, solver = 'proxqp', **kwargs):
    """
    surface1: function handle for the top surface
    surface2: function handle for the bottom surface
    load: load magnitude
    material_properties: dictionary with the material properties
    mesh: dictionary with the mesh properties
    friction: friction coefficient
    """
    # metrial properties
    E1 = material_properties['E1']
    E2 = material_properties['E2']
    ni1 = material_properties['ni1']
    ni2 = material_properties['ni2']
    G1 = E1/2/(1+ni1)
    G2 = E2/2/(1+ni2)
    friction = material_properties['f']

    # mesh data
    hx = mesh['hx']
    hy = mesh['hy']
    w_x = mesh['w_x']
    w_y = mesh['w_y']

    # material properties for compliance matrix
    B1_zz = (1-ni1**2)/E1/np.pi
    B2_zz = (1-ni2**2)/E2/np.pi
    B1_zx = 1*(1-2*ni1)/4/np.pi/G1
    B2_zx = 1*(1-2*ni2)/4/np.pi/G2

    P = load

    # Build mesh
    x_bounds = [-w_x/2, w_x/2]
    y_bounds = [-w_y/2, w_y/2]

    range_x = np.arange(x_bounds[0], x_bounds[1] + hx, hx)
    range_y = np.arange(y_bounds[0], y_bounds[1] + hy, hy)
    x,y = np.meshgrid(range_x, range_y) # rectangle centers
    idx, idy = np.meshgrid(np.arange(0, range_x.shape[0], 1), np.arange(0, range_y.shape[0], 1))

    shp = x.shape

    z1 = surface1['point'](x,y)
    z2 = surface2['point'](x,y)

    n1 = surface1['normal'](x.flatten().reshape(1, -1), y.flatten().reshape(1, -1))
    n2 = surface2['normal'](x.flatten().reshape(1, -1), y.flatten().reshape(1, -1))
    n_avg = (n1 + n2)/np.linalg.norm(n1 + n2, axis=0)
    proj1 = np.sum(n_avg * n1, axis=0)
    proj2 = np.sum(n_avg * n2, axis=0)

    projk = n_avg[2, :].reshape(-1, 1)

    dA, total_area, deltaX, deltaY = compute_surface_area(n_avg, hx, hy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    ax.plot_surface(x, y, dA.reshape(shp), edgecolor='black')
    plt.show()
    if use_normals == False:
        proj1 = np.ones(proj1.shape)
        proj2 = np.ones(proj2.shape)
        projk = np.ones(projk.shape)

    # compliance matrix
    V = np.array([x.flatten(), y.flatten()]).T
    n = V.shape[0]
    i = idx.flatten()
    j = idy.flatten()
    Sc_zz = np.zeros((n,n))
    Sc_xz = np.zeros((n,n))
    Sc_yz = np.zeros((n,n))
    xx = np.zeros((n,n))
    yy = np.zeros((n,n))

    for k in range(n):
        K_zz, K_zx, K_zy = compliance_uniform_rectangular_pressure(i[k], j[k], i[k:n], j[k:n], deltaX[k], deltaY[k])
        Sc_zz[k, k:n] = K_zz
        Sc_zz[k:n, k] = K_zz
        Sc_xz[k, k:n] = -K_zx
        Sc_xz[k:n, k] = -K_zx
        Sc_yz[k, k:n] = -K_zy
        Sc_yz[k:n, k] = -K_zy
    dA = 1/deltaX/deltaY
    Sp = Sc_zz*(B1_zz*proj1.reshape(1, -1) + B2_zz*proj2.reshape(1, -1))*dA
    # Sp = Sc_zz*(B1_zz + B2_zz)
    # Sp[Sp < max(Sp.flatten())/100] = 0

    # unloaded gap
    eps = z1 - z2
    x_scale = P/(n/4)*np.ones((n+1,1))
    x_scale[-1] = np.max(eps)
    x_scale = np.ones((n+1,1))
    E = eps.flatten()
    o = np.zeros((n,1))
    e = np.ones((n,1))
    G = np.c_[-Sp, e*projk]*x_scale.T
    H = 0.5*x_scale.T*np.r_[np.c_[Sp, o], np.c_[o.T, 0]]*x_scale
    q = np.r_[o, -np.atleast_2d(P)]*x_scale
    h = E.reshape(-1, 1)*projk
    lb = np.zeros((n+1,))
    A = (np.ones((n,))*projk.T).flatten()
    A = np.concatenate((A, np.atleast_1d(0)), axis=0).reshape(1, -1)
    b = np.array([P]).reshape(1, -1)


    tic = time.time()
    sol = solve_qp(P = H + 1e-6* np.eye(H.shape[0]), q = q, G = G, h = h, A = A, b = b, lb=lb,  solver=solver, verbose = True, **kwargs)
    sol = sol*x_scale.flatten()
    print('Elapsed time: ', time.time() - tic)

    return sol, Sc_zz*B1_zz, Sc_zz*B2_zz, z1, z2, x, y

def post_processing(sol, S1, S2, z1, z2, x, y, mesh):

    hx = mesh['hx']
    hy = mesh['hy']
    wx = mesh['w_x']
    wy = mesh['w_y']

    n = sol.shape[0]-1
    e = np.ones((n,1))

    # post processing
    forces = sol[0:n]
    rigid_body_displacement = sol[n]
    p = forces/(hx*hy)
    row,col = z1.shape
    p = np.reshape(p, (row,col))
    # body 1 displacement
    uz1 = (S1)@forces  - rigid_body_displacement*e.flatten()
    # body 2 displacement
    uz2 = (S2)@forces

    fig = plt.figure()
    norm = colors.Normalize(vmin=0, vmax = p.max())
    colormap = cm.viridis
    face_colors = colormap(norm(p))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    # plot deformed surfaces and pressure as cdata
    ax.plot_surface(x, y, (z1 + 0*np.reshape(uz1, z1.shape)), edgecolor='black', facecolors=face_colors)
    ax.plot_surface(x, y, (z2 - 0*np.reshape(uz2, z1.shape)), edgecolor='black', facecolors=face_colors)

    # set custom axis aspect ratio
    set_data_aspect_ratio(ax, [1,1,1])
    plt.draw()
    plt.show()

    # plot just the pressure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, p, edgecolor='black', cmap='viridis')
    # Add interactive cursor
    cursor = mplcursors.cursor(surf, hover=True)

    # Show (x, y, z) when hovering
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(f"x={sel.target[0]:.2f}\n"
                                f"y={sel.target[1]:.2f}\n"
                                f"z={sel.target[2]:.2f}")
    plt.show()

    pmax = np.max(p.flatten())
    area = np.sum(p.flatten()/pmax > 1e-2)*hx*hy
    delta = np.max(((S1)@forces).flatten() + ((S2)@forces).flatten())
    return rigid_body_displacement, area, pmax, p, delta

def Hertzian_contact(radius1, radius2, load, E1, E2, ni1, ni2):

    E = 1/(1/E1 + 1/E2)
    R = 1/(1/radius1 + 1/radius2)
    
def main():
    # surface 1: top surface
    coeffs = np.array([2.1942,   -0.8122,    0.0822])*1e4
    import casadi as ca
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')

    R = 3 # mm cylinder radius
    f1 = lambda x,y: -(-ca.fabs(y)**2 + R**2)**0.5 + R*1.001
    f1_expr = f1(x, y)
    f1_dx = ca.jacobian(f1_expr, x)
    f1_dy = ca.jacobian(f1_expr, y)
    n1_expr = ca.cross(ca.vertcat(1,0,f1_dx), ca.vertcat(0,1,f1_dy))
    n1_expr = n1_expr/ca.norm_2(n1_expr)
    n1_fun = ca.Function('n1', [x, y], [n1_expr.T@ca.vertcat(0,0,1)])

    # surface 2: bottom surface
    w = 0.82 # mm
    f2 = lambda x,y: (coeffs[0]*x**8 + coeffs[1]*x**6 + coeffs[2] - 822)/1000*(ca.logic_and(x>= - w/2, x<=w/2))+\
        (-0.05)*(x<-w/2) + (-0.05)*(x>w/2) + y*0
    f2_expr = f2(x, y)
    f2_dx = ca.jacobian(f2_expr, x)
    f2_dy = ca.jacobian(f2_expr, y)
    n2_expr = ca.cross(ca.vertcat(1,0,f2_dx), ca.vertcat(0,1,f2_dy))
    n2_expr = n1_expr/ca.norm_2(n2_expr)
    n2_fun = ca.Function('n1', [x, y], [n2_expr.T@ca.vertcat(0,0,1)])

    # surfaces plot
    import pylab as plt
    # make sure that the figures do not pause the code execution, and not close them
    # plt.ion()
    X, Y = np.meshgrid(np.linspace(-1.2*w/2, 1.2*w/2, 20), np.linspace(-0.5*w/2, 0.5*w/2, 20))

    z1 = f1(X,Y).full()
    z2 = f2(X,Y).full()
    # Create figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    ax.plot_surface(X, Y, z1, cmap='viridis')
    ax.plot_surface(X, Y, z2, cmap='viridis')

    # Labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Show plot
    plt.show()

    # material properties
    material_properties = {'E1': 206e3, 'E2': 206e3, 'ni1': 0.3, 'ni2': 0.3, 'f': 0.0}
    mesh = {'hx': w/50, 'hy': w/30, 'w_x': 1.2*w, 'w_y': 1*w}
    load = 500

    surface1 = {'point': lambda x,y: f1(x,y).full(), 'normal': lambda x,y: n1_fun(x,y).full()}
    surface2 = {'point': lambda x,y: f2(x,y).full(), 'normal': lambda x,y: n2_fun(x,y).full()}
    sol, S1, S2, z1, z2, x, y = simulate_bem(surface1, surface2, load, material_properties, mesh)

    hx = mesh['hx']
    hy = mesh['hy']
    n = sol.shape[0]-1
    e = np.ones((n,1))

    # post processing
    forces = sol[0:n]
    rigid_body_displacement = sol[n]
    p = forces/(hx*hy)
    row,col = z1.shape
    p = np.reshape(p, (row,col))
    # body 1 displacement
    uz1 = (S1)@forces  - rigid_body_displacement*e.flatten()
    # body 2 displacement
    uz2 = (S2)@forces

    fig = plt.figure()
    norm = colors.Normalize(vmin=0, vmax = p.max())
    colormap = cm.viridis
    face_colors = colormap(norm(p))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,0.5])
    # plot deformed surfaces and pressure as cdata
    ax.plot_surface(x, y, 10*(z1 + np.reshape(uz1, z1.shape)), edgecolor='black', facecolors=face_colors)
    ax.plot_surface(x, y, 10*(z2 - np.reshape(uz2, z1.shape)), edgecolor='black', facecolors=face_colors)

    # set custom axis aspect ratio
    ax.set_xlim([-1.2*w/2, 1.2*w/2])
    ax.set_ylim(ax.get_xlim())
    plt.draw()
    plt.show()

    # plot just the pressure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, p, edgecolor='black', cmap='viridis')
    # Add interactive cursor
    cursor = mplcursors.cursor(surf, hover=True)

    # Show (x, y, z) when hovering
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(f"x={sel.target[0]:.2f}\n"
                                f"y={sel.target[1]:.2f}\n"
                                f"z={sel.target[2]:.2f}")
    plt.show()
    return

def main_2():

    import casadi as ca

    
    x  = ca.SX.sym('x')
    y  = ca.SX.sym('y')

    c0 = 0.0025 # mm
    n = 0
    R2 = 25
    c = 2 ** n * c0
    R1 = R2 - c

    # surface 1: top surface
    f1_expr = ca.if_else( x**2 + y **2 <= R1 ** 2, - (-y**2 - x**2 + R1**2)**(0.5), R1*1.1)
    f1 = ca.Function('f1', [x, y], [f1_expr])
    f1_dx = ca.jacobian(f1_expr, x)
    f1_dy = ca.jacobian(f1_expr, y)
    n1_expr = ca.vertcat(-f1_dx, -f1_dy, 1)
    n1_expr = n1_expr / ca.norm_2(n1_expr)
    n1_fun = ca.Function('n1', [x, y], [n1_expr])
    surface1 = {'point': lambda x,y: f1(x,y).full(), 'normal': lambda x,y: n1_fun(x,y).full()}

    # surface 2: bottom surface
    f2_expr = ca.if_else( x**2 + y **2 <= R2 ** 2, - (-y**2 - x**2 + R2**2)**(0.5), R2)
    f2 = ca.Function('f2', [x, y], [f2_expr])
    f2_dx = ca.jacobian(f2_expr, x)
    f2_dy = ca.jacobian(f2_expr, y)
    n2_expr = ca.vertcat(-f2_dx, -f2_dy, 1)
    n2_expr = n2_expr / ca.norm_2(n2_expr)
    n2_fun = ca.Function('n2', [x, y], [n2_expr])
    surface2 = {'point': lambda x,y: f2(x,y).full(), 'normal': lambda x,y: n2_fun(x,y).full()}

    material_properties = {'E1': 200e3, 'E2': 200e3, 'ni1': 0.3, 'ni2': 0.3, 'f': 0.0}
    mesh = {'hx': R2/10, 'hy': R2/10, 'w_x': 3*R1, 'w_y': 3*R1}
    load = 2000 # N

    sol, S1, S2, z1, z2, x, y = simulate_bem(surface1, surface2, load, material_properties, mesh, solver = 'proxqp')

    post_processing(sol, S1, S2, z1, z2, x, y, mesh)

    return

if __name__ == '__main__':
    main()