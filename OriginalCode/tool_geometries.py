import casadi as ca
import screwCalculus as sc
import numpy as np
from solvers import fsolve_casadi
import time


# function workaround to set equal axis ratio in matplotlib
# Equal aspect workaround
def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])
    return

def rack_profile2D(m, alpha, rc, rho_blade, rho_flankrem, flankrem_depth, x, alphaFlank, rack_addendum_coeff):

    height = m * rack_addendum_coeff

    # I am defining the rack profile in the x-z plane
    alpha = alpha * np.pi / 180  # Convert to radians
    CxB = np.pi*m/4 + rho_blade * np.cos(alpha)  # X coordinate of the base circle
    CzB = rho_blade * np.sin(alpha) - x  # Y coordinate of the base circle

    theta_blade_end = np.arcsin((CzB + flankrem_depth + x) / rho_blade)
    theta_blade_start = np.arcsin( (CzB + rc+ x - height)/(rho_blade+rc))
    Xf = CxB - np.cos(theta_blade_start) * (rho_blade + rc)  # X coordinate of the startif tip fillet point
    theta_flankrem_start = theta_blade_end - alphaFlank
    CzF = CzB + np.sin(theta_flankrem_start)*rho_flankrem - np.sin(theta_blade_end)*rho_blade
    CxF = CxB + np.cos(theta_flankrem_start)*rho_flankrem - np.cos(theta_blade_end)*rho_blade
    theta_flankrem_end = np.arcsin((CzF + 1.25*m + x) / rho_flankrem)

    u1 = Xf
    u2 = u1 + rc*(np.pi/2 - theta_blade_start)
    u3 = u2 + rho_blade*(theta_blade_end - theta_blade_start)
    u4 = u3 + rho_flankrem*(theta_flankrem_end - theta_flankrem_start)

    u_sym = ca.SX.sym('u')

    topland_bool = ca.logic_and(u_sym >= 0, u_sym <= u1)
    fillet_bool = ca.logic_and(u_sym > u1, u_sym <= u2)
    blade_bool = ca.logic_and(u_sym > u2, u_sym <= u3)
    flankrem_bool = u_sym > u3

    angle = topland_bool*0 + \
        fillet_bool*((u_sym - u1)/rc) + \
        blade_bool*((u_sym - u2)/rho_blade + theta_blade_start) + \
        flankrem_bool*((u_sym - u3)/rho_flankrem + theta_flankrem_start)
    
    cosa = ca.cos(angle)
    sina = ca.sin(angle)
    point1 = topland_bool*ca.vertcat(u_sym, height - x)
    point2 = fillet_bool*ca.vertcat(Xf + rc*sina, height - x + rc*(cosa - 1))
    point3 = blade_bool*ca.vertcat(CxB - rho_blade*cosa, CzB - rho_blade*sina)
    point4 = flankrem_bool*ca.vertcat(CxF - rho_flankrem*cosa, CzF - rho_flankrem*sina)

    point = point1 + point2 + point3 + point4

    normal1 = topland_bool*ca.vertcat(0, 1)
    normal2 = fillet_bool*ca.vertcat(sina, cosa)
    normal3 = blade_bool*ca.vertcat(cosa, sina)
    normal4 = flankrem_bool*ca.vertcat(cosa, sina)

    normal = normal1 + normal2 + normal3 + normal4

    # casadi functions
    point_func = ca.Function('point_func', [u_sym], [point])
    normal_func = ca.Function('normal_func', [u_sym], [normal])

    # return the functions
    return point_func, normal_func, u1, u2, u3, u4

def hiirt_tool2D(H, alpha, s = None, rho_blade = 500000, rho_fillet = None):
    
    alpha = alpha * np.pi / 180  # Convert to radians
    p = H/2 * np.tan(alpha) # tool semi width
    Cxb =  p + rho_blade * np.cos(alpha)  # X coordinate of the base circle
    Cyb = rho_blade*np.sin(alpha)
    
    if rho_fillet is None:
        rho_fillet_sym = ca.SX.sym('rho_fillet')
        theta_blade_start_expr = ca.acos(Cxb/(rho_blade + rho_fillet_sym))
        eq = s - rho_fillet_sym * (1 - ca.sin(theta_blade_start_expr))

        sol, *dump = fsolve_casadi(eq, rho_fillet_sym, [], s, [])
        rho_fillet = sol[0]
    
    theta_blade_start = np.arccos(Cxb/(rho_fillet + rho_blade))
    s = rho_fillet * (1 - np.sin(theta_blade_start))  # fillet height
    Cfy = Cyb - (rho_fillet + rho_blade)*np.sin(theta_blade_start)
    c = H/2 - (rho_fillet + Cfy)
    h2 = H/2 - c - s
    theta_blade_end = np.arcsin((Cyb + h2)/rho_blade)
    u1 = rho_fillet*(np.pi/2 - theta_blade_start)
    u2 = u1 + rho_blade*(theta_blade_end - theta_blade_start)
    
    u_sym = ca.SX.sym('u')

    fillet_bool = ca.logic_and(u_sym >= 0, u_sym <= u1)
    # blade_bool = ca.logic_and(u_sym > u1, u_sym <= u2*1.1)
    blade_bool = u_sym > u1

    angle = fillet_bool*(u_sym/rho_fillet) + \
        blade_bool*((u_sym-u1)/rho_blade + theta_blade_start)
    
    point1 = fillet_bool*ca.vertcat(rho_fillet*ca.sin(angle), Cfy + rho_fillet*ca.cos(angle) + H/2)
    point2 = blade_bool*ca.vertcat(Cxb - rho_blade*ca.cos(angle), Cyb - rho_blade*ca.sin(angle) + H/2)

    point = point1 + point2

    normal1 = fillet_bool*ca.vertcat(ca.sin(angle), ca.cos(angle))
    normal2 = blade_bool*ca.vertcat(ca.cos(angle), ca.sin(angle))

    normal = normal1 + normal2

    # casadi functions
    point_func = ca.Function('point_func', [u_sym], [point])
    normal_func = ca.Function('normal_func', [u_sym], [normal])

    point_height = Cfy + rho_fillet
    # return the functions
    return point_func, normal_func, u1, u2, point_height, s

def hiirt_cutter_surf(diameter, H, alpha, s = None, rho_blade = 500000, rho_fillet = None):

    p, n, u1, u2, point_height = hiirt_tool2D(H, alpha, s, rho_blade, rho_fillet)

    u = ca.SX.sym('u')
    theta = ca.SX.sym('theta')
    p_surf_expr = sc.rigidInverse(sc.TtY(point_height - diameter/2) @ sc.TrotX(theta)) @ ca.vertcat(p(u), 1, 1)
    n_surf_expr = sc.rigidInverse(sc.TtY(point_height - diameter/2) @ sc.TrotX(theta)) @ ca.vertcat(n(u), 1, 0)
    p_surf_fun = ca.Function('p_surf_fun', [u, theta], [p_surf_expr])
    n_surf_fun = ca.Function('n_surf_fun', [u, theta], [n_surf_expr])

    return p_surf_fun, n_surf_fun, u1, u2

def main_debug():
    # Example usage
    m = 1.0
    alpha = 20.0
    rc = 0.1
    rho_blade = 9000
    rho_flankrem = 9000
    flankrem_depth = 1*m
    x = 0
    alphaFlank = 0
    rack_addendum_coeff = 1.25

    point_func, normal_func, u1, u2, u3, u4 = rack_profile2D(m, alpha, rc, rho_blade, rho_flankrem, flankrem_depth, x, alphaFlank, rack_addendum_coeff)

    # Test the functions with a sample input
    u_test = np.linspace(0, u4 + 0.1, 100).reshape(1, -1)  # Sample input for u
    points = point_func(u_test)
    normals = normal_func(u_test)

    print("Points: ", points)
    print("Normals: ", normals)

    # plot with matplotlib
    import matplotlib.pyplot as plt

    points = points.full()
    normals = normals.full()
    plt.figure(figsize=(10, 5))
    plt.plot(points[0,:], points[1,:], label='Rack Profile')
    plt.quiver(points[0,:], points[1,:], normals[0,:], normals[1,:], angles='xy', scale_units='xy', scale=5, color='r', label='Normals')
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Rack Profile and Normals')
    
    plt.axis('equal')
    plt.show()

def parabolic_path(lengthwise_crowning, alpha, Dext, Dint):

    eL = lengthwise_crowning/np.sin(alpha*np.pi/180)
    Kc = -4*eL/(Dext/2 - Dint/2)**2
    t = ca.SX.sym('t')
    p_expr = ca.vertcat(t, 0, Kc*t**2)

    tvec = ca.jacobian(p_expr, t)
    dsdt = ca.norm_2(tvec)
    tvec = tvec/dsdt
    nvec = ca.vertcat(0,1,0)
    bvec = ca.cross(tvec, nvec)
    G = ca.vertcat(ca.horzcat(tvec, nvec, bvec, p_expr), ca.horzcat(0, 0, 0, 1))

    G_fun = ca.Function('G_fun', [t], [G])
    p_fun = ca.Function('p_fun', [t], [p_expr])
    tvec_fun = ca.Function('tvec_fun', [t], [tvec])
    nvec_fun = ca.Function('nvec_fun', [t], [nvec])
    bvec_fun = ca.Function('bvec_fun', [t], [bvec])

    return G_fun, p_fun, tvec_fun, nvec_fun, bvec_fun

def hiirt_kinematics(Dext, Dint, Lc, theta_error, alpha_error, Xm_error, Ym_error, delta_error, alpha_tool, H):

    # motion parameter
    s = ca.SX.sym('s')
    G_cp, *rest = parabolic_path(Lc, alpha_tool, Dext, Dint)
    G_cp_expr = G_cp(s)

    mean_point = (Dext/2 - Dint/2)/2
    T = ca.vertcat(ca.cos(delta_error), ca.sin(delta_error), -ca.sin(alpha_error))
    t = T/ ca.norm_2(T)
    n = ca.vertcat(-ca.sin(delta_error), ca.cos(delta_error), 0)
    b = ca.cross(t, n)

    # Gcp = sc.TtZ(0)@sc.TrotZ(Xm_error/Dext*2) @ sc.TtX(Dext/2) @ ca.vertcat(ca.horzcat(t, n, b, ca.vertcat(0,0,0)), ca.horzcat(0, 0, 0, 1)) @ sc.TrotX(theta_error) @ sc.TtX(-mean_point)  @ G_cp_expr
    Gcp = sc.TtZ(75) @ sc.TrotZ(-Xm_error/Dext*2) @ sc.TtX(Dext/2) @ sc.TtZ(0) @ sc.TrotZ(delta_error) @ sc.TrotY(alpha_error) @ sc.TrotX(theta_error) @ sc.TtX(-mean_point) @ G_cp_expr
    Gcp_fun = ca.Function('Gcp_fun', [s], [Gcp])

    return Gcp_fun

def main2():

    Lc    = 0.1
    alpha = 25
    Dext  = 200
    Dint  = 175
    H     = 10
    theta_error = 0
    alpha_error = 0
    Xm_error    = 0
    Ym_error    = 0
    delta_error = 0
    rho_fillet  = 1

    profile_fun, _, u1, u2, _ = hiirt_tool2D(H, alpha, s=None, rho_blade=30, rho_fillet=rho_fillet)
    
    
    G_parabolic, p_fun, tvec_fun, nvec_fun, bvec_fun = parabolic_path(Lc, alpha, Dext, Dint)

    print(tvec_fun(1).full())
    print(nvec_fun(1).full())
    print(bvec_fun(1).full())

    Gcp = hiirt_kinematics(Dext, Dint, Lc, theta_error, alpha_error, Xm_error, Ym_error, delta_error, alpha)

    u = ca.SX.sym('u')
    s = ca.SX.sym('s')

    Gcp_expr = Gcp(s)
    profile_expr = profile_fun(u)
    surf_expr = Gcp(s) @ ca.vertcat(0, profile_expr[0], -profile_expr[1], 1)
    surf_expr_left = Gcp(s) @ ca.vertcat(0, -profile_expr[0], -profile_expr[1], 1)
    surf_fun = ca.Function('surf_fun', [u, s], [surf_expr])
    surf_fun_left = ca.Function('surf_fun_left', [u, s], [surf_expr_left])

    s_num = np.linspace(-(Dext/2-Dint/2)/2, (Dext/2-Dint/2)/2, 100)
    u_num = np.linspace(0, u2+0.2, 100)

    profile_points = profile_fun(u_num.reshape(1,-1)).full()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(profile_points[0, :], profile_points[1, :], label='Rack Profile')
    plt.axis('equal')
    plt.show()
    print(Gcp(1).full())


    S, U = np.meshgrid(s_num, u_num)
    points = surf_fun(U.reshape(1, -1), S.reshape(1, -1))
    points = points.full()

    X = points[0, :].reshape(U.shape)
    Y = points[1, :].reshape(U.shape)
    Z = points[2, :].reshape(U.shape)

    points_left = surf_fun_left(U.reshape(1, -1), S.reshape(1, -1))
    points_left = points_left.full()
    X_left = points_left[0, :].reshape(U.shape)
    Y_left = points_left[1, :].reshape(U.shape)
    Z_left = points_left[2, :].reshape(U.shape)

    # X = np.concatenate((X, np.flip(X_left, axis  = 1)), axis=1)
    # Y = np.concatenate((Y, np.flip(Y_left, axis  = 1)), axis=1)
    # Z = np.concatenate((Z, np.flip(Z_left, axis  = 1)), axis=1)

    X = np.concatenate((np.flip(X, axis = 0), X_left), axis=0) # np.flip(X, axis = 0)
    Y = np.concatenate((np.flip(Y, axis = 0), Y_left), axis=0)
    Z = np.concatenate((np.flip(Z, axis = 0), Z_left), axis=0)

    # plot with matplotlib

    import easy_plot as ep
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    fig = ep.Figure()
    surface = ep.surface(fig, X, Y, Z)
    fig.updateImage()
    app.exec_()

    # fit with a nurbs surface
    import nurbs as nrb
    surf = nrb.Nurbs(knotsU = None, knotsV = None, degU = 3, degV = 3, control_points = None)

    Q = {'X': X, 'Y': Y, 'Z': Z}
    surf.fit(Q, 3, 3, [20, 30])
    # surf.plot()

    stp_writer = nrb.initialize_step_writer()
    nurbs_OCC = nrb.Nurbs_to_STEPwriter(surf, stp_writer)

    # write step to file
    stp_writer.Write('test.step')

    from OCC.Display.SimpleGui import init_display
    display, start_display, add_menu, add_function_to_menu = init_display()

    display.DisplayShape(nurbs_OCC)

    start_display()

    # export the nurbs surface to a step file
    return


if __name__ == "__main__":
    main2()
    # main_debug()  