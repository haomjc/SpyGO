import numpy as np
from math import pi
from math import log, sqrt
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

def Szz(iprime, jprime, i, j, hx, hy, ni, G):
    """
    z direction compliance on point (i,j) for a uniform rectangular pressure distribution on (iprime,jprime)
    """
    deltai=iprime-i
    deltaj=jprime-j
    k=(deltai+0.5)*hx
    m=(deltaj+0.5)*hy
    l=(deltai-0.5)*hx
    n=(deltaj-0.5)*hy

    sk2n2 = np.sqrt(k**2+n**2)
    sl2m2 = np.sqrt(l**2+m**2)
    sk2m2 = np.sqrt(k**2+m**2)
    sl2n2 = np.sqrt(l**2+n**2)

    S = (1-ni)/(2*pi*G)
    return S*(k*np.log((m+sk2m2)/(n+sk2n2))\
              +l*np.log((n+sl2n2)/(m+sl2m2))\
                +m*np.log((k+sk2m2)/(l+sl2m2))\
                    +n*np.log((l+sl2n2)/(k+sk2n2)))



def main():
    # Try to evaluate the solution of Bussinesq integral and find the trend w.r.t. to the distance 

    import casadi as ca
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    # surface 1: top surface
    R = 3 # mm cylinder radius
    f1 = lambda x,y: -np.sqrt(-y**2 + R**2) + R*1.0001
    f1_expr = f1(x, y)
    f1_dx = ca.jacobian(f1_expr, x)
    f1_dy = ca.jacobian(f1_expr, y)
    n1_expr = ca.cross(ca.vertcat(1,0,f1_dx), ca.vertcat(0,1,f1_dy))
    n1_expr = n1_expr/ca.norm_2(n1_expr)
    n1_fun = ca.Function('n1', [x, y], [n1_expr.T@ca.vertcat(0,0,1)])

    w = 1
    s = 4 
    s2 = 4
    L = 0.6

    R = w/2
    f2 = lambda x,y: ( (-ca.fabs(x)**s + R**s)**(1/s) -R)*(x<=-w/2*L) + \
    ((-ca.fabs(x)**s +R**s)**(1/s) - R)*(x>=w/2*L) + \
    ((-ca.fabs(w/2*L)**s +R**s)**(1/s) - R + ((-ca.fabs(x)**s2 +R**s2)**(1/s2) - R) - ((-ca.fabs(w/2*L)**s2 +R**s2)**(1/s2) - R))*(ca.logic_and((x<=w/2*L), (x>=-w/2*L)))
    f2_expr = f2(x, y)
    f2_dx = ca.jacobian(f2_expr, x)
    f2_dy = ca.jacobian(f2_expr, y)
    n2_expr = ca.cross(ca.vertcat(1,0,f2_dx), ca.vertcat(0,1,f2_dy))
    n2_expr = n1_expr/ca.norm_2(n2_expr)
    n2_fun = ca.Function('n1', [x, y], [n2_expr.T@ca.vertcat(0,0,1)])

    # metrial properties
    E1 = 200000 # MPa
    E2 = 200000 # MPa
    ni1 = 0.3
    ni2 = 0.3
    hx = w/30
    hy = w/60
    B1p = (1-ni1**2)/E1/hx/hy
    B2p = (1-ni2**2)/E2/hx/hy
    P = 500 # N

    # surfaces plot
    import pylab as plt
    X, Y = np.meshgrid(np.linspace(-1.2*w/2, 1.2*w/2, 20), np.linspace(-0.5*w/2, 0.5*w/2, 20))

    z1 = f1(X,Y)
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

    # Build mesh
    x_bounds = [-1.3*w/2, 1.3*w/2]
    y_bounds = [-0.5*w/2, 0.5*w/2]

    range_x = np.arange(x_bounds[0], x_bounds[1] + hx, hx)
    range_y = np.arange(y_bounds[0], y_bounds[1] + hy, hy)
    x,y = np.meshgrid(range_x, range_y) #
    idx, idy = np.meshgrid(np.arange(0, range_x.shape[0], 1), np.arange(0, range_y.shape[0], 1))
    return

if __name__ == '__main__':
    main()