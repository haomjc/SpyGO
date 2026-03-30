from gears.main.core import rackCutter
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import screwCalculus as sc
import easy_plot as ep
from nurbs import Nurbs

def involute_curve():
    rb = ca.SX.sym('rb')
    th = ca.SX.sym('theta')
    sth  = ca.sin(th)
    cth = ca.cos(th)
    OC = rb*ca.vertcat(-sth, 
                    cth,
                    0)
    
    CP = rb*th*ca.vertcat(cth,
                       sth,
                       0)
    
    OP = ca.Function('pos', [rb, th], [OC+CP]) # position vector

    t = ca.Function('t', [th], [ca.vertcat(-sth,
                   cth,
                   0
        )])# tangent vector
    
    n = ca.Function('n', [th], [-ca.vertcat(
        cth,
        sth,
        0)
        ]) # normal vector

    b = ca.Function('b', [th], [ca.cross(t(th), n(th))]) # bi-normal vector

    return OP, t, n, b

def straight_crown_wheel_surf(flank = 'right'):

    u = ca.SX.sym('u') # profile parameter
    s = ca.SX.sym('s') # lenghtwise curve parameter

    # rack cutter profile parameters
    params = ca.SX.sym('par', 7, 1)
    m = params[0]
    alpha_cnv  = params[1]
    alpha_cvx = params[2]
    rc_cnv = params[3]
    rc_cvx = params[4]
    x = params[5]
    b = params[6]
    par_cutter_cnv = ca.vertcat(m, alpha_cnv, rc_cnv, x, b)
    par_cutter_cvx = ca.vertcat(m, alpha_cvx, rc_cvx, x, b)

    rack_sym = rackCutter().create_standard(parametric=True, height_coeff=1.25)
    p_expr = rack_sym.p_fun(par_cutter_cnv, u)
    n_expr = rack_sym.n_fun(par_cutter_cnv, u)

    p_rack = ca.vertcat(p_expr[0], 0, -p_expr[1], 1)
    n_rack = ca.vertcat(n_expr[0], 0, -n_expr[1], 0)

    if flank.lower() == 'left':
        p_expr = rack_sym.p_fun(par_cutter_cvx, u)
        n_expr = rack_sym.n_fun(par_cutter_cvx, u)
        p_rack = ca.vertcat(-p_expr[0], 0, -p_expr[1], 1)
        n_rack = ca.vertcat(-n_expr[0], 0, -n_expr[1], 0)

    # compute Frenet Serret frame and the HMT
    p_expr = ca.vertcat(0,s,0)
    t_expr = ca.vertcat(0,1,0)
    n_expr = ca.vertcat(-1,0,0)
    b_expr = ca.vertcat(0,0,1)
    G = ca.vertcat(
        ca.horzcat(n_expr, t_expr, b_expr, p_expr),
        ca.horzcat(0,0,0,1)
    )

    S = G @ p_rack
    S_n = G @ n_rack

    point_fcn = ca.Function('point', [params, ca.vertcat(u, s)], [S])
    normal_fcn = ca.Function('normal', [params, ca.vertcat(u, s)], [S_n])
    return point_fcn, normal_fcn

def crown_wheel_surf(flank = 'concave'):

    u = ca.SX.sym('u')
    theta = ca.SX.sym('theta')

    params = ca.SX.sym('par', 8, 1)
    m = params[0]
    alpha_cnv  = params[1]
    alpha_cvx = params[2]
    rc_cnv = params[3]
    rc_cvx = params[4]
    x = params[5]
    b = params[6]

    par_cutter_cnv = ca.vertcat(m, alpha_cnv, rc_cnv, x, b)
    par_cutter_cvx = ca.vertcat(m, alpha_cvx, rc_cvx, x, b)
    rb = params[7]

    rack_sym = rackCutter().create_standard(parametric=True, height_coeff=1.25)
    p_expr = rack_sym.p_fun(par_cutter_cnv, u)
    n_expr = rack_sym.n_fun(par_cutter_cnv, u)

    p_rack = ca.vertcat(p_expr[0], 0, -p_expr[1], 1)
    n_rack = ca.vertcat(n_expr[0], 0, -n_expr[1], 0)

    if flank.lower() == 'convex':
        p_expr = rack_sym.p_fun(par_cutter_cvx, u)
        n_expr = rack_sym.n_fun(par_cutter_cvx, u)
        p_rack = ca.vertcat(-p_expr[0], 0, -p_expr[1], 1)
        n_rack = ca.vertcat(-n_expr[0], 0, -n_expr[1], 0)

    p, t, n, b = involute_curve()

    p_expr = p(rb, theta)
    t_expr = t(theta)
    n_expr = n(theta)
    b_expr = b(theta)

    G = ca.vertcat(
        ca.horzcat(n_expr, t_expr, b_expr, p_expr),
        ca.horzcat(0,0,0,1)
    )

    S = G @ p_rack
    S_n = G @ n_rack

    return ca.Function('point', [params, ca.vertcat(u, theta)], [S]), ca.Function('normal', [params, ca.vertcat(u, theta)], [S_n])

def gear_wheel_kinematics():

    psi = ca.SX.sym('psi')
    kinematic_params = ca.SX.sym('kinem_params', 6, 1)

    ratio_roll = kinematic_params[0]
    delta_g = kinematic_params[1]
    blank_tilt = kinematic_params[2]
    x_offset = kinematic_params[3]
    y_offset = kinematic_params[4]
    z_offset = kinematic_params[5]

    psi_w = -psi * ca.sin(delta_g)*ratio_roll
    # Rwg = sc.rotZ(-psi_w) @ sc.rotX(-delta_g-np.pi/2) @ sc.rotZ(psi)
    Tfw = sc.TtX(x_offset) @ sc.TtY(y_offset) @ sc.TtZ(-z_offset) @ sc.TrotX(+delta_g+np.pi/2) @ sc.TrotZ(psi_w)
    Tfg = sc.TtZ(-102.5) @ sc.TrotY(blank_tilt) @ sc.TtZ(+102.5) @ sc.TrotZ(psi)
    Twg = sc.rigidInverse(Tfw) @ Tfg

    Rwg = Twg[0:3,0:3]
    dwg = Twg[0:3, 3]
    

    k_g = Rwg @ ca.vertcat(0,
                           0,
                           1)

    k_w = ca.vertcat(0,
                     0,
                     1)
    
    omega_rel = -ratio_roll*ca.sin(delta_g)*k_w - k_g

    T_fun = ca.Function('Twg', [kinematic_params, psi], [Twg])
    omega_fun = ca.Function('omega_rel', [kinematic_params, psi], [omega_rel])
    return T_fun, omega_fun

def eq_meshing(flank = 'concave'):

    env_triplets = ca.SX.sym('uthetapsi', 3, 1)

    params = ca.SX.sym('par', 8, 1)
    kinematic_params = ca.SX.sym('kin', 6, 1) # delta_g, ratio_roll, z

    G_wg, omega = gear_wheel_kinematics()

    omega_rel = omega(kinematic_params, env_triplets[2])

    p, n = crown_wheel_surf(flank = flank)

    p = p(params, env_triplets[0:2])
    n = n(params, env_triplets[0:2])

    p_gear = sc.rigidInverse(G_wg(kinematic_params, env_triplets[2])) @ p
    n_gear = sc.rigidInverse(G_wg(kinematic_params, env_triplets[2])) @ n

    p_gear = p_gear[0:3]
    n_gear = n_gear[0:3]

    pg_fun = ca.Function('eq_meshing', [params, kinematic_params, env_triplets], [p_gear])
    ng_fun = ca.Function('eq_meshing', [params, kinematic_params, env_triplets], [n_gear])

    v_rel = ca.cross(omega_rel, p[0:3])

    eq_meshing  = v_rel.T @ n[0:3]
    eq_fun = ca.Function('eq_meshing', [params, kinematic_params, env_triplets], [eq_meshing])

    return eq_fun, pg_fun, ng_fun

def residuals_sampling_eqs(flank = 'concave'):

    env_triplets = ca.SX.sym('uthetapsi', 3, 1)
    params = ca.SX.sym('par', 8, 1)
    kinematic_params = ca.SX.sym('kin', 6, 1)

    # residual symbolic
    h = ca.SX.sym('h')

    # target point symbolic
    p_target = ca.SX.sym('target_point', 3, 1)

    eq_fun, pg_fun, ng_fun = eq_meshing(flank = flank)

    eq_meshing_expr = eq_fun(params, kinematic_params, env_triplets)
    pg_expr = pg_fun(params, kinematic_params, env_triplets)
    ng_expr = ng_fun(params, kinematic_params, env_triplets)

    eq_sol = ca.vertcat(pg_expr[0:3] - h* ng_expr[0:3] - p_target,
                        eq_meshing_expr)

    return ca.Function('eq_residuals', [params, kinematic_params, p_target, ca.vertcat(env_triplets, h)], [eq_sol])

def rz_sampling_solver(flank = 'concave', params = None, kinematic_params = None):

    triplet = ca.SX.sym('env', 3, 1)
    zR = ca.SX.sym('zR', 2, 1)
    sym_params = False
    if params is None:
        params = ca.SX.sym('par', 8, 1)
        kinematic_params = ca.SX.sym('kin', 6, 1)
        sym_params = True
    

    eq_fun, pg_fun, ng_fun = eq_meshing(flank = flank)
    eq_meshing_expr = eq_fun(params, kinematic_params, triplet)
    pg_expr = pg_fun(params, kinematic_params, triplet)

    eq_sol = ca.vertcat(pg_expr[2]**2 - zR[0]**2,
                        pg_expr[0]**2 + pg_expr[1]**2 - zR[1]**2,
                        eq_meshing_expr)

    if sym_params:
        problem = {'x': triplet, 'p': ca.vertcat(params, kinematic_params, zR), 'g': eq_sol}
    else:
        problem = {'x': triplet, 'p': zR, 'g': eq_sol}

    solver = ca.rootfinder('s', 'newton', problem, {'error_on_fail':False})
    return solver, pg_fun, ng_fun

def rz_sampling(flank, params, kinematic_params, z, R, guess):
    solver, pg_fun, ng_fun = rz_sampling_solver(flank = flank, params = params, kinematic_params = kinematic_params)
    shp = z.shape
    zR = np.array([z.flatten(order = 'F'), R.flatten(order = 'F')])
    sol = solver(x0 = guess, p = zR)['x']
    points = pg_fun(params, kinematic_params, sol).full()
    normals = ng_fun(params, kinematic_params, sol).full()
    return points, normals

def conesIntersection(cone1, cone2):
    """
    Python equivalent of MATLAB conesIntersection.

    Parameters
    ----------
    cone1 : array-like of length 3
        [Ar, Az, B] coefficients for the first cone.
    cone2 : array-like of length 3
        [Ar, Az, B] coefficients for the second cone.

    Returns
    -------
    zR : np.ndarray of shape (2,)
        [z, R] coordinates of the intersection.
    """
    Ar, Az, B = cone1
    Ar2, Az2, B2 = cone2

    R = (Az2 / Az * B - B2) / (Ar2 - Az2 / Az * Ar)
    z = -(B + Ar * R) / Az

    return np.array([z, R])

def cones_sampling(qCones, root_cone, face_cone, params, kinematic_params, flank = 'concave',  n_prof = 70, guess = ca.vertcat(4, -ca.pi/5, +0.6)):

    solver, p_fun, n_fun = rz_sampling_solver(flank = flank, params=params, kinematic_params=kinematic_params)

    s = np.linspace(0.025, 1.03, n_prof)

    z_mat = np.full((n_prof, qCones.shape[0]), 0.0)
    R_mat = np.full((n_prof, qCones.shape[0]), 0.0)

    for ii in range(0, qCones.shape[0]):
        zR1 = conesIntersection(root_cone, qCones[ii,:])
        zR2 = conesIntersection(face_cone, qCones[ii,:])
        z_mat[:, ii] = zR1[0] + s*(zR2[0] - zR1[0])
        R_mat[:, ii] = zR1[1] + s*(zR2[1] - zR1[1])

    z_sampling = z_mat.reshape(-1, 1, order='F')
    R_sampling = R_mat.reshape(-1, 1, order='F')
    zR = ca.horzcat(z_sampling, R_sampling).T
    zR = zR.full()

    sol = solver(x0 = guess, p = zR)['x']
    points = p_fun(params, kinematic_params, sol).full()
    normals = n_fun(params, kinematic_params, sol).full()
    return points, normals, z_mat, R_mat

def fit_to_nurbs(params, kinematic_params, zR_bounds, ease_off_cnv = None, ease_off_cvx = None):
    from hypoid.main.geometry import generate_rz_grid
    from computational_geometry import interp_arc
    z, R, U, V = generate_rz_grid(zR_bounds, n_prof=45, n_face=50, extend_tip=True, extend_heel=True, extend_toe=False, shrink_root= True)
    shp = z.shape
    zR = np.array([z.flatten(order = 'F'), R.flatten(order = 'F')])
    # gear guess
    # guess_cvx = ca.vertcat(params[0]*0.31, +ca.pi/7, -0.7)
    # guess_cnv = ca.vertcat(params[0]*0.31, +ca.pi/7, +0.7)
    # pinion guess
    guess_cvx = ca.vertcat(params[0]*0.35, -ca.pi/5, +0.8)
    guess_cnv = ca.vertcat(params[0]*0.35, -ca.pi/5, -0.8)
    pts_cvx, nrm_cvx = rz_sampling('convex', params, kinematic_params, z, R, guess_cvx)
    pts_cnv, nrm_cnv = rz_sampling('concave', params, kinematic_params, z, R, guess_cnv)
    nrm_cnv *=-1
    nrm_cvx *=-1
    from hypoid.main.ease_off import ease_off_9DoF, ease_off_fillet

    if ease_off_cvx is not None:
        E_fun = ease_off_9DoF(ease_off_cnv) 
        E_num = E_fun(U, V)

        # z = pts_cvx[2,:].reshape(45, 50, order = 'F')
        # R = np.sqrt(pts_cvx[0,:]**2 + pts_cvx[1,:]**2).reshape(45, 50, order = 'F')

        # F = ep.Figure()
        # s = ep.surface(F, z, R, E_num*100)
        # F.show()
        pts_cvx += E_num.reshape(1, -1, order='F')*nrm_cvx
    
    X_cvx = pts_cvx[0,:].reshape(shp, order = 'F'); Xn_cvx = nrm_cvx[0,:].reshape(shp, order = 'F')
    Y_cvx = pts_cvx[1,:].reshape(shp, order = 'F'); Yn_cvx = nrm_cvx[1,:].reshape(shp, order = 'F')
    Z_cvx = pts_cvx[2,:].reshape(shp, order = 'F'); Zn_cvx = nrm_cvx[2,:].reshape(shp, order = 'F')
    
    if ease_off_cnv is not None:
            
        E_fun = ease_off_9DoF(ease_off_cnv) 
        E_num = E_fun(U, V)
        pts_cnv += E_num.reshape(1, -1, order='F')*nrm_cnv

    X_cnv = pts_cnv[0,:].reshape(shp, order = 'F'); Xn_cnv = nrm_cnv[0,:].reshape(shp, order = 'F')
    Y_cnv = pts_cnv[1,:].reshape(shp, order = 'F'); Yn_cnv = nrm_cnv[1,:].reshape(shp, order = 'F')
    Z_cnv = pts_cnv[2,:].reshape(shp, order = 'F'); Zn_cnv = nrm_cnv[2,:].reshape(shp, order = 'F')

    p_root_cnv = np.vstack((X_cnv[:,0], Y_cnv[:,0], Z_cnv[:,0]))
    n_root_cnv = np.vstack((Xn_cnv[:,0], Yn_cnv[:,0], Zn_cnv[:,0]))

    p_root_cvx = np.vstack((X_cvx[:,0], Y_cvx[:,0], Z_cvx[:,0]))
    n_root_cvx = np.vstack((Xn_cvx[:,0], Yn_cvx[:,0], Zn_cvx[:,0]))

    p_mid = (p_root_cnv + p_root_cvx)*0.5
    n_mid = (n_root_cnv + n_root_cvx)*0.5

    # X_cnv[:,0] = p_mid[0,:]
    # Y_cnv[:,0] = p_mid[1,:]
    # Z_cnv[:,0] = p_mid[2,:]

    # Xn_cnv[:,0] = n_mid[0,:]
    # Yn_cnv[:,0] = n_mid[1,:]
    # Zn_cnv[:,0] = n_mid[2,:]

    X_cvx[:,0] = p_mid[0,:]
    Y_cvx[:,0] = p_mid[1,:]
    Z_cvx[:,0] = p_mid[2,:]

    Xn_cvx[:,0] = n_mid[0,:]
    Yn_cvx[:,0] = n_mid[1,:]
    Zn_cvx[:,0] = n_mid[2,:]
    
    X = np.hstack((np.fliplr(X_cvx), X_cnv))
    Y = np.hstack((np.fliplr(Y_cvx), Y_cnv))
    Z = np.hstack((np.fliplr(Z_cvx), Z_cnv))

    F = ep.Figure()
    s = ep.surface(F, X, Y, Z) 
    F.show()

    r, c = X.shape
    
    for ii in range(r):
        points = np.vstack((X[ii,:], Y[ii,:], Z[ii,:]))
        x, y, z = interp_arc(c, points[0,:], points[1,:], points[2,:])
        X[ii,:] = x
        Y[ii,:] = y
        Z[ii,:] = z

        
    r, c = X_cnv.shape
    for ii in range(r):
        points = np.vstack((X_cnv[ii,:], Y_cnv[ii,:], Z_cnv[ii,:]))
        x, y, z = interp_arc(c, points[0,:], points[1,:], points[2,:])
        X_cnv[ii,:] = x
        Y_cnv[ii,:] = y
        Z_cnv[ii,:] = z

    r, c = X_cvx.shape
    for ii in range(r):
        points = np.vstack((X_cvx[ii,:], Y_cvx[ii,:], Z_cvx[ii,:]))
        x, y, z = interp_arc(c, points[0,:], points[1,:], points[2,:])
        X_cvx[ii,:] = x
        Y_cvx[ii,:] = y
        Z_cvx[ii,:] = z

    Q = {
        'X': X,
        'Y': Y,
        'Z': Z
    }
    nurbs = Nurbs([],[],[],[],[])
    nurbs.fit(Q, 2, 2, (30,55))

    Q = {
        'X': X_cnv,
        'Y': Y_cnv,
        'Z': Z_cnv
    }
    nurbs_cnv = Nurbs([],[],[],[],[])
    nurbs_cnv.fit(Q, 2, 2, (30,35))

    Q = {
        'X': X_cvx,
        'Y': Y_cvx,
        'Z': Z_cvx
    }
    nurbs_cvx = Nurbs([],[],[],[],[])
    nurbs_cvx.fit(Q, 2, 2, (30,35))
    return nurbs, nurbs_cnv, nurbs_cvx

def zr_nurbs_sampling(nurbs:Nurbs, Z, R, UVguess=None):
    """
    Compute 3D surface points and normals corresponding to given (R, Z)
    coordinates using a NURBS surface and CasADi solvers.

    Parameters
    ----------
    R : array-like
        Array of cylindrical radii.
    Z : array-like
        Array of cylindrical heights (z-coordinates).
    nurbs : object
        NURBS surface object providing:
          - initCasadifun()
          - casadi_fun(u, v)
          - evalNormal(u, v)
    UVguess : (2, N) array, optional
        Initial guess for UV parameters for each point.

    Returns
    -------
    xyzbase : np.ndarray, shape (3, N)
        3D Cartesian coordinates of points on the surface.
    normalsbase : np.ndarray, shape (3, N)
        Surface normals at those points.
    UV : np.ndarray, shape (2, N)
        Parametric (u, v) coordinates on the NURBS surface.
    """

    # Define CasADi symbols
    u = ca.SX.sym('u')
    v = ca.SX.sym('v')
    R_sym = ca.SX.sym('R')
    Z_sym = ca.SX.sym('Z')

    # The system to solve: ||pg_xy||^2 - R^2 = 0, pg_z - Z = 0
    def system(x, Rv, Zv):
        pg = nurbs.casadi_eval(x[0], x[1])
        return ca.vertcat(pg[0] ** 2 + pg[1] ** 2 - Rv ** 2,
                          pg[2] - Zv)

    expr = system(ca.vertcat(u, v), R_sym, Z_sym)

    # --- Define CasADi solvers ---
    rf_root = {'x': ca.vertcat(u, v), 'p': ca.vertcat(R_sym, Z_sym), 'g': expr}
    SolverRoot = ca.rootfinder('solver', 'newton', rf_root, {'error_on_fail': False})

    opts = {
        'ipopt': {
            'nlp_scaling_method': 'gradient-based',
            'mu_strategy': 'adaptive',
            'linear_solver': 'ma57',
            'print_level': 5,
            'mu_oracle': 'probing',
            'alpha_for_y': 'min-dual-infeas',
            'adaptive_mu_globalization': 'never-monotone-mode'
        },
        'print_time': 0
    }

    rf_ipopt = {'x': ca.vertcat(u, v),
                'p': ca.vertcat(R_sym, Z_sym),
                'f': 0.5 * ca.dot(expr, expr)}
    SolverIpopt = ca.nlpsol('S', 'ipopt', rf_ipopt, opts)

    # --- Initialize outputs ---
    r, c = Z.shape
    # R = np.atleast_1d(np.array(R).flatten())
    # Z = np.atleast_1d(np.array(Z).flatten())
    N = r*c

    normalsbase = np.full((3, N), np.nan)
    U = np.full((1, r, c), np.nan)
    V = np.full((1, r, c), np.nan)

    # Default guess
    if UVguess is None:
        guess = np.array([0.2, 0.2])
    else:
        UVguess = np.array(UVguess)

    # --- Loop over each (R, Z) ---
    # for i in range(r):
    #     guess = np.array([0.5, 0.5])
    #     for j in range(c):
    #         if UVguess is not None and UVguess.size > 0:
    #             guess = UVguess[:, i]
    #             # Clip guesses to (0.15, 0.95)
    #             guess = np.clip(guess, 0.15, 0.95)

            # Attempt rootfinder first
            # try:
            #     res = SolverRoot(x0=guess, p=[R[i, j], Z[i, j]])
            #     res = np.array(res['x']).flatten()
            #     guess = res
            # except Exception:
            # print(f"rootfinder failed at index {i}, using IPOPT")
            # res = SolverIpopt(x0=guess, p=[R[i, j], Z[i, j]],
            #                 ubx=[1, 1], lbx=[0, 0])
            # res = np.array(res['x']).flatten()

            # U[0, i, j] = res[0]
            # V[0, i, j] = res[1]
    
    # UV = np.vstack((U.flatten().reshape(1, -1), V.flatten().reshape(1, -1)))
    # if UVguess is not None:
    #     guess = UVguess
    #     guess = np.clip(guess, 0.15, 0.95)
    # else:
    #     guess = np.array([0.2, 0.5])

    res = SolverRoot(x0=guess, p = ca.vertcat(R.reshape(1, -1, order = 'F'), Z.reshape(1, -1, order = 'F')))
    UV = np.array(res['x'])
    # res = SolverIpopt(x0=guess, p = ca.vertcat(R.reshape(1, -1, order = 'F'), Z.reshape(1, -1, order = 'F')))
    # UV = np.array(res['x'])
    # --- Evaluate surface points ---
    xyz_fun = nurbs.casadi_eval(UV[0, :].reshape(1,-1, order = 'F'), UV[1, :].reshape(1,-1, order = 'F')).full()
    normalsbase = nurbs.casadi_normal(UV[0, :].reshape(1,-1, order = 'F'), UV[1, :].reshape(1,-1, order = 'F')).full()
    xyzbase = np.array(xyz_fun)

    return xyzbase, normalsbase, UV

def main_pinion():

    from MultyxInterface.main.msh_generation import createBasicDataForMSH, readQuadratureCones, CALYX_MSH

    params = [4.25 ,        0.39097473 ,  0.36538241  , 1.52019059,   1.68748435,
        1.21903446,  -0.09377835,  53.89464916]
    kinematic_params = [1.08404173,   0.31415927,
        -0.08726647, -11.48031649  , 1.31864862 ,-22.40563389]
    
    # cones calculation: cones order: front -> back -> base -> face

    # points on front cone
    z = [75.7511, 77.5816]
    R = [19.0769, 25.5168]
    A = np.vstack((z, [1, 1])).T
    b = np.array(R).reshape(-1, 1)
    coeffs = np.linalg.solve(A, b); m = coeffs[0,0]; q = coeffs[1,0]
    tan_angle = m
    apex = q/tan_angle
    front_cone = [1, -tan_angle, -apex*tan_angle]

    # points on back cone
    z = [121.5, 121.5]
    R = [31.15, 35.642]

    back_cone = [0, 1, -121.5] # equation: z = 121.5

    # points on root cone: base cone will have the root apex shifted along +z
    z = [85.8116, 114.093]
    R = [18.754, 27.9597]

    A = np.vstack((z, [1, 1])).T
    b = np.array(R).reshape(-1, 1)
    coeffs = np.linalg.solve(A, b); m = coeffs[0,0]; q = coeffs[1,0]
    tan_angle = m
    apex = q/tan_angle
    base_cone = [1, -tan_angle, -(apex-10)*tan_angle]
    root_cone = [1, -tan_angle, -(apex)*tan_angle]

    print("Base cone parameters:")
    print(f"Apex: {apex-10}")
    print(f"Angle: {np.arctan(tan_angle)*180/np.pi}")
    # points on face cone
    z = [83.5894, 108.798]
    R = [29.0199, 37.2431]
    A = np.vstack((z, [1, 1])).T
    b = np.array(R).reshape(-1, 1)
    coeffs = np.linalg.solve(A, b); m = coeffs[0,0]; q = coeffs[1,0]
    tan_angle = m
    apex = q/tan_angle
    face_cone = [1, -tan_angle, -apex*tan_angle]

    zR_A = conesIntersection(base_cone, front_cone)
    zR_B = conesIntersection(base_cone, back_cone)

    print(f"Base cone A point: {zR_A}")
    print(f"Base cone B point: {zR_B}")
    print(f"OuterDia1: {zR_A[1]*2}")
    print(f"OuterDia2: {zR_B[1]*2}")
    print(f"InnerDia1: {zR_A[1]*2-10}")
    print(f"InnerDia2: {zR_B[1]*2-10}")
    print(f"Shaft length: {zR_B[0] - zR_A[0]}")

    secondary_face_back = [1, 0, -37.5]
    cones = np.array([front_cone, back_cone, base_cone, face_cone, face_cone, secondary_face_back])

    print("Cones generated:")
    print(cones)
    
    path  = r'C:\Users\egrab\Desktop\Ferrari\GT_2025_reverse_N2\T3D'
    member = 'pinion'
    mshname = member + '.msh'
    n_teeth = 8
    createBasicDataForMSH(member, path, mshname, n_teeth, cones, secondary=True)
    print('Basic data for MSH generation provided...')

    # read quadrature cones
    q_cones = readQuadratureCones(path, member, write_to_file = False)
    print("Quadrature cones reading...")


    zR1 = conesIntersection(root_cone, front_cone)
    zR2 = conesIntersection(root_cone, back_cone)
    zR3 = conesIntersection(face_cone, back_cone)
    zR4 = conesIntersection(face_cone, front_cone)

    zR_bounds = np.array([zR1, zR2, zR3, zR4])
     
    EO_cnv = [-0.02915025,-0.03232543, -0.09838886, -0.05397777 , 0.00284426, -0.03830321,
        0.02900724, -0.01947911 , 0.02599234]
    EO_cvx = [[0.01038767, -0.28870755,  0.11674772,  0.01589604,  0.01033635,  0.1058825,
        -0.08487564,  0.11182238, -0.02592621]]
    nurbs, nurbs_cnv, nurbs_cvx = fit_to_nurbs(params, kinematic_params, zR_bounds, ease_off_cnv=EO_cnv, ease_off_cvx=EO_cvx)

    guess = ca.vertcat(params[0]*0.35, -ca.pi/5, -0.8)
    points_cvx, normals_cvx, z_mat, R_mat = cones_sampling(q_cones, root_cone, face_cone, params, kinematic_params, flank = 'concave', guess = guess)
    # guess = ca.vertcat(params[0]*0.35, -ca.pi/5, 0.8)
    # points_cnv, normals_cnv, _, _ = cones_sampling(q_cones, root_cone, face_cone, params, kinematic_params, flank = 'convex', guess = guess)
    # normals_cvx *= -1
    # normals_cnv *= -1

    # guess = ca.vertcat(params[0]*0.35, -ca.pi/5, -0.8)
    # points_cvx, normals_cvx, z_mat, R_mat = cones_sampling(q_cones, root_cone, face_cone, params, kinematic_params, flank = 'convex', guess = guess)

    # guess = ca.vertcat(params[0]*0.32, +ca.pi/7, +0.5)
    # points_cnv, normals_cnv, _, _ = cones_sampling(q_cones, root_cone, face_cone, params, kinematic_params, flank = 'concave', guess = guess)
    # normals_cvx *= -1
    # normals_cnv *= -1

    points_cnv, normals_cnv, UV = zr_nurbs_sampling(nurbs_cvx, z_mat, R_mat)
    points_cvx, normals_cvx, UV = zr_nurbs_sampling(nurbs_cnv, z_mat, R_mat)
    normals_cvx*=-1

    X_cnv = points_cnv[0,:].reshape(70, -1, order = 'F')
    Y_cnv = points_cnv[1,:].reshape(70, -1, order = 'F')
    Z_cnv = points_cnv[2,:].reshape(70, -1, order = 'F')
    Xn_cnv = normals_cnv[0,:].reshape(70, -1, order = 'F')
    Yn_cnv = normals_cnv[1,:].reshape(70, -1, order = 'F')
    Zn_cnv = normals_cnv[2,:].reshape(70, -1, order = 'F')

    X_cvx = points_cvx[0,:].reshape(70, -1, order = 'F')
    Y_cvx = points_cvx[1,:].reshape(70, -1, order = 'F')
    Z_cvx = points_cvx[2,:].reshape(70, -1, order = 'F')
    Xn_cvx = normals_cvx[0,:].reshape(70, -1, order = 'F')
    Yn_cvx = normals_cvx[1,:].reshape(70, -1, order = 'F')
    Zn_cvx = normals_cvx[2,:].reshape(70, -1, order = 'F')

    p_root_cnv = np.vstack((X_cnv[0,:], Y_cnv[0,:], Z_cnv[0,:]))
    n_root_cnv = np.vstack((Xn_cnv[0,:], Yn_cnv[0,:], Zn_cnv[0,:]))

    p_root_cvx = np.vstack((X_cvx[0,:], Y_cvx[0,:], Z_cvx[0,:]))
    n_root_cvx = np.vstack((Xn_cvx[0,:], Yn_cvx[0,:], Zn_cvx[0,:]))

    p_mid = (p_root_cnv + p_root_cvx)*0.5
    n_mid = (n_root_cnv + n_root_cvx)*0.5

    X_cnv[0,:] = p_mid[0,:]
    Y_cnv[0,:] = p_mid[1,:]
    Z_cnv[0,:] = p_mid[2,:]

    Xn_cnv[0,:] = n_mid[0,:]
    Yn_cnv[0,:] = n_mid[1,:]
    Zn_cnv[0,:] = n_mid[2,:]

    X_cvx[0,:] = p_mid[0,:]
    Y_cvx[0,:] = p_mid[1,:]
    Z_cvx[0,:] = p_mid[2,:]

    Xn_cvx[0,:] = n_mid[0,:]
    Yn_cvx[0,:] = n_mid[1,:]
    Zn_cvx[0,:] = n_mid[2,:]

    points_cnv = np.vstack((X_cnv.reshape(-1, 1, order = 'F').T, Y_cnv.reshape(-1, 1, order = 'F').T, Z_cnv.reshape(-1, 1, order = 'F').T))
    normals_cnv = np.vstack((Xn_cnv.reshape(-1, 1, order = 'F').T, Yn_cnv.reshape(-1, 1, order = 'F').T, Zn_cnv.reshape(-1, 1, order = 'F').T))
    points_cvx = np.vstack((X_cvx.reshape(-1, 1, order = 'F').T, Y_cvx.reshape(-1, 1, order = 'F').T, Z_cvx.reshape(-1, 1, order = 'F').T))
    normals_cvx = np.vstack((Xn_cvx.reshape(-1, 1, order = 'F').T, Yn_cvx.reshape(-1, 1, order = 'F').T, Zn_cvx.reshape(-1, 1, order = 'F').T))

    points_cnv = sc.rotZ(2*np.pi/n_teeth)@points_cnv
    normals_cnv = sc.rotZ(2*np.pi/n_teeth)@normals_cnv


#     E_fun = ease_off_9DoF([-0.02915025,-0.03232543, -0.09838886, -0.05397777 , 0.00284426, -0.03830321,
#         0.02900724, -0.01947911 , 0.02599234]) 
#     E_num = E_fun(U, V)
#     points_cnv += E_num.reshape(1, -1, order='F')*normals_cnv
    
#     E_fun = ease_off_9DoF([0.01038767, -0.28870755,  0.11674772,  0.01589604,  0.01033635,  0.1058825,
#  -0.08487564,  0.11182238, -0.02592621]) 
#     E_num = E_fun(U, V)
    # points_cvx += E_num.reshape(1, -1, order='F')*normals_cvx
    # F = ep.Figure()
    # pts_cvx = ep.quiver(F,points_cvx[0,:], points_cvx[1,:], points_cvx[2,:], normals_cvx[0,:], normals_cvx[1,:], normals_cvx[2,:])
    # pts_cnv = ep.quiver(F, points_cnv[0,:],points_cnv[1,:],points_cnv[2,:], normals_cnv[0, :], normals_cnv[1,:], normals_cnv[2,:])
    # F.show()

    # F = ep.Figure()
    # ep.scatter(F, z_mat.flatten(), R_mat.flatten(), E_num.flatten()*100)
    # F.show()

    p = np.hstack((points_cvx, points_cnv))
    n = np.hstack((normals_cvx, normals_cnv))
    CALYX_MSH(path, p, n, member)
    return

def main_gear():

    from MultyxInterface.main.msh_generation import createBasicDataForMSH, readQuadratureCones, CALYX_MSH

    params = [ 4.25   ,    0.38269816,  0.39975603,  1.2,  1.2, -0.0,
  0.15911157, 82.20861284]
    kinematic_params = [1.01852415, 1.25663706,  0.0 ,         0.48355136,
        0.99999999,  4.85827962]
    
    # cones calculation: cones order: front -> back -> base -> face

    # points on front cone
    z = [19.2189, 24.8753]
    R = [78.9236, 77.21]
    A = np.vstack((z, [1, 1])).T
    b = np.array(R).reshape(-1, 1)
    coeffs = np.linalg.solve(A, b); m = coeffs[0,0]; q = coeffs[1,0]
    tan_angle = m
    apex = q/tan_angle
    front_cone = [1, -tan_angle, -apex*tan_angle]

    # points on back cone
    z = [29.9592, 36.7553]
    R = [115.883, 113.827]
    b = np.array(R).reshape(-1, 1)
    coeffs = np.linalg.solve(A, b); m = coeffs[0,0]; q = coeffs[1,0]
    tan_angle = m
    apex = q/tan_angle
    back_cone = [1, -tan_angle, -apex*tan_angle]

    # points on root cone: base cone will have the root apex shifted along +z
    z = [27.5989, 37.1319]
    R = [80.825, 110.28]

    A = np.vstack((z, [1, 1])).T
    print(A)
    b = np.array(R).reshape(-1, 1)
    print(b)
    coeffs = np.linalg.solve(A, b); m = coeffs[0,0]; q = coeffs[1,0]
    tan_angle = m
    apex = q/tan_angle
    base_cone = [1, -tan_angle, -(apex-10)*tan_angle]
    root_cone = [1, -tan_angle, -(apex)*tan_angle]

    print("Base cone parameters:")
    print(f"Apex: {apex-10}")
    print(f"Angle: {np.arctan(tan_angle)*180/np.pi}")

    zR_A = conesIntersection(base_cone, front_cone)
    zR_B = conesIntersection(base_cone, back_cone)

    print(f"Base cone A point: {zR_A}")
    print(f"Base cone B point: {zR_B}")
    print(f"OuterDia1: {zR_A[1]*2}")
    print(f"OuterDia2: {zR_B[1]*2}")
    print(f"InnerDia1: {zR_A[1]*2-10}")
    print(f"InnerDia2: {zR_B[1]*2-10}")
    print(f"Shaft length: {zR_B[0] - zR_A[0]}")
    print(tan_angle)
    print(q)
    # points on face cone
    z = [19.2916, 27.1734]
    R = [88.7204, 113.329]
    A = np.vstack((z, [1, 1])).T
    b = np.array(R).reshape(-1, 1)
    coeffs = np.linalg.solve(A, b); m = coeffs[0,0]; q = coeffs[1,0]
    tan_angle = m
    apex = q/tan_angle
    face_cone = [1, -tan_angle, -apex*tan_angle]

    secondary_front = [0, 1, -18.4]
    cones = np.array([front_cone, back_cone, base_cone, face_cone, secondary_front, face_cone])

    print("Cones generated:")
    print(cones)
    
    path  = r'C:\Users\egrab\Desktop\Ferrari\GT_2025_reverse_N2\T3D'
    member = 'gear'
    mshname = member + '.msh'
    n_teeth = 35
    createBasicDataForMSH(member, path, mshname, n_teeth, cones, secondary=True)
    print('Basic data for MSH generation provided...')

    # read quadrature cones
    q_cones = readQuadratureCones(path, member, write_to_file = False)
    print("Quadrature cones reading...")

    zR1 = conesIntersection(root_cone, front_cone)
    zR2 = conesIntersection(root_cone, back_cone)
    zR3 = conesIntersection(face_cone, back_cone)
    zR4 = conesIntersection(face_cone, front_cone)

    zR_bounds = np.array([zR1, zR2, zR3, zR4])
     
    EO_cnv = [-0.02948026, -0.08643483,  0.01652208, -0.05488231, -0.00165687, -0.0055977,
        -0.03737456,  0.02332042,  0.01930011]
    EO_cvx = [[0.04906429, -0.08249135, -0.04131732, -0.06202457, -0.00261753, -0.00164868,
        0.02240734, -0.01709827,  0.01353732]]
    nurbs, nurbs_cnv, nurbs_cvx = fit_to_nurbs(params, kinematic_params, zR_bounds, ease_off_cnv=EO_cnv, ease_off_cvx=EO_cvx)

    guess = ca.vertcat(params[0]*0.32, +ca.pi/7, -0.5)
    points_cvx, normals_cvx, z_mat, R_mat = cones_sampling(q_cones, root_cone, face_cone, params, kinematic_params, flank = 'convex', guess = guess)

    # guess = ca.vertcat(params[0]*0.32, +ca.pi/7, +0.5)
    # points_cnv, normals_cnv, _, _ = cones_sampling(q_cones, root_cone, face_cone, params, kinematic_params, flank = 'concave', guess = guess)
    # normals_cvx *= -1
    # normals_cnv *= -1

    points_cnv, normals_cnv, UV = zr_nurbs_sampling(nurbs_cnv, z_mat, R_mat)
    points_cvx, normals_cvx, UV = zr_nurbs_sampling(nurbs_cvx, z_mat, R_mat)
    normals_cnv*=-1
    X_cnv = points_cnv[0,:].reshape(70, -1, order = 'F')
    Y_cnv = points_cnv[1,:].reshape(70, -1, order = 'F')
    Z_cnv = points_cnv[2,:].reshape(70, -1, order = 'F')
    Xn_cnv = normals_cnv[0,:].reshape(70, -1, order = 'F')
    Yn_cnv = normals_cnv[1,:].reshape(70, -1, order = 'F')
    Zn_cnv = normals_cnv[2,:].reshape(70, -1, order = 'F')

    X_cvx = points_cvx[0,:].reshape(70, -1, order = 'F')
    Y_cvx = points_cvx[1,:].reshape(70, -1, order = 'F')
    Z_cvx = points_cvx[2,:].reshape(70, -1, order = 'F')
    Xn_cvx = normals_cvx[0,:].reshape(70, -1, order = 'F')
    Yn_cvx = normals_cvx[1,:].reshape(70, -1, order = 'F')
    Zn_cvx = normals_cvx[2,:].reshape(70, -1, order = 'F')


    p_root_cnv = np.vstack((X_cnv[0,:], Y_cnv[0,:], Z_cnv[0,:]))
    n_root_cnv = np.vstack((Xn_cnv[0,:], Yn_cnv[0,:], Zn_cnv[0,:]))

    p_root_cvx = np.vstack((X_cvx[0,:], Y_cvx[0,:], Z_cvx[0,:]))
    n_root_cvx = np.vstack((Xn_cvx[0,:], Yn_cvx[0,:], Zn_cvx[0,:]))

    p_mid = (p_root_cnv + p_root_cvx)*0.5
    n_mid = (n_root_cnv + n_root_cvx)*0.5

    X_cnv[0,:] = p_mid[0,:]
    Y_cnv[0,:] = p_mid[1,:]
    Z_cnv[0,:] = p_mid[2,:]

    Xn_cnv[0,:] = n_mid[0,:]
    Yn_cnv[0,:] = n_mid[1,:]
    Zn_cnv[0,:] = n_mid[2,:]

    X_cvx[0,:] = p_mid[0,:]
    Y_cvx[0,:] = p_mid[1,:]
    Z_cvx[0,:] = p_mid[2,:]

    Xn_cvx[0,:] = n_mid[0,:]
    Yn_cvx[0,:] = n_mid[1,:]
    Zn_cvx[0,:] = n_mid[2,:]




    points_cnv = np.vstack((X_cnv.reshape(-1, 1, order = 'F').T, Y_cnv.reshape(-1, 1, order = 'F').T, Z_cnv.reshape(-1, 1, order = 'F').T))
    normals_cnv = np.vstack((Xn_cnv.reshape(-1, 1, order = 'F').T, Yn_cnv.reshape(-1, 1, order = 'F').T, Zn_cnv.reshape(-1, 1, order = 'F').T))
    points_cvx = np.vstack((X_cvx.reshape(-1, 1, order = 'F').T, Y_cvx.reshape(-1, 1, order = 'F').T, Z_cvx.reshape(-1, 1, order = 'F').T))
    normals_cvx = np.vstack((Xn_cvx.reshape(-1, 1, order = 'F').T, Yn_cvx.reshape(-1, 1, order = 'F').T, Zn_cvx.reshape(-1, 1, order = 'F').T))

    points_cnv = sc.rotZ(-2*np.pi/n_teeth)@points_cnv
    normals_cnv = sc.rotZ(-2*np.pi/n_teeth)@normals_cnv


    # F = ep.Figure()
    # ep.scatter(F, z_mat.flatten(), R_mat.flatten(), E_num.flatten()*100)
    # F.show()

    F = ep.Figure()
    pts_cvx = ep.quiver(F, points_cvx[0,:], points_cvx[1,:], points_cvx[2,:], normals_cvx[0,:], normals_cvx[1,:], normals_cvx[2,:])
    pts_cnv = ep.quiver(F, points_cnv[0,:],points_cnv[1,:],points_cnv[2,:], normals_cnv[0, :], normals_cnv[1,:], normals_cnv[2,:])
    F.show()


    p = np.hstack((points_cnv, points_cvx))
    n = np.hstack((normals_cnv, normals_cvx))
    CALYX_MSH(path, p, n, member)
    return



if __name__ == '__main__':
    main_pinion()
    # main_gear()