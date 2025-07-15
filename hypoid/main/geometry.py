from casadi.casadi import exp
import numpy as np
import casadi as ca
import screwCalculus as sc
from scipy.optimize import fsolve # fsolve(f, x0=x0, fprime=jacobian)
from math import sqrt, pi, atan, cos, sin, acos, asin, tan
from general_utils import *

from .kinematics import *
from .utils import *
from solvers import fsolve_casadi



def surface_sampling_casadi(data: DesignData, member, flank, sampling_size, triplet_guess = None, spreadblade = False, FW_vec = None):
    n_face = sampling_size[0]
    n_prof = sampling_size[1]
    n_fillet = sampling_size[2]
    HAND = data.system_data.HAND

    blank_settings = list(data.extract_blank_settings(member))
    tool_settings = data.extract_tool_settings(member, flank)

    raw_machine_settings = data.extract_machine_settings_matrix(member, flank)
    if spreadblade:
        raw_machine_settings = data.extract_machine_settings_matrix(member, 'concave')

    if triplet_guess is None or not triplet_guess:
        triplet_guess = initial_guess_from_data(data, member, flank)
    
    if member.lower() == 'gear' and data.gear_common_data.gen_type.lower() == 'formate':
        raise Exception("Formate sampling not yet implemented")
        return # TO DO: tooth_sampling_casadi_formate()
    else:
        surfVars, filletVars, points, normals, pointsFillet, normalsFillet, pointsRoot, normalsRoot, rootVars, pointsBounds, normalsBounds =\
        tooth_sampling_casadi(raw_machine_settings, tool_settings, blank_settings, member, flank, HAND, triplet_guess, n_face, n_prof, n_fillet)
    
    p_tool_fun, n_tool_fun, _ = casadi_tool_fun(flank, toprem=True, flankrem=True)

    p_tool = p_tool_fun(tool_settings, reduce_2d(surfVars[0:2])).full().reshape((3, surfVars.shape[1], surfVars.shape[2]))
    n_tool = n_tool_fun(tool_settings, reduce_2d(surfVars[0:2])).full().reshape((3, surfVars.shape[1], surfVars.shape[2]))

    z_tool = p_tool[2,:,:].reshape(-1,).min()

    return points, normals, p_tool, n_tool, surfVars, z_tool, pointsFillet, pointsRoot, normalsRoot, rootVars, pointsBounds, normalsBounds

def tooth_sampling_casadi(machine_settings, tool_settings, blank_settings, member, flank, HAND, triplet_guess, n_face, n_prof, n_fillet):

    # extract relevant blank parameters
    A0 = blank_settings[0]; Fw = blank_settings[1]
    front_angle = blank_settings[3]; back_angle = blank_settings[4]
    pitch_angle = blank_settings[5]; pitch_apex = blank_settings[6]
    face_angle = blank_settings[7]; face_apex = blank_settings[8]
    root_angle = blank_settings[9]; root_apex = blank_settings[10]

    # compute z-R coordinates of the face cone at the toe
    R_head = (A0 - Fw)*sin(face_angle)
    z_head = (A0 - Fw)*cos(face_angle) - face_apex

    # machine kinematics and tool geometry
    ggt, Vgt, Vgt_spatial = casadi_machine_kinematics(member, HAND)
    p_tool, n_tool, csi_edge_blade = casadi_tool_fun(flank, toprem=True, flankrem=True)
    csi_edge_blade = csi_edge_blade(tool_settings).full()

    # initialize casadi variables and expressions
    c = ca.SX.sym('c')
    csi = ca.SX.sym('csi')
    theta = ca.SX.sym('csi')
    phi = ca.SX.sym('csi')
    x = ca.SX.sym('csi')
    y = ca.SX.sym('csi')
    z = ca.SX.sym('csi')
    enveloping_triplet = ca.vertcat(csi, theta, phi)
    surface_point = ca.vertcat(x, y, z)

    # symbolic tool equation
    p_tool_expr = ca.vertcat(p_tool(tool_settings, ca.vertcat(csi, theta)), 1)
    n_tool_expr = ca.vertcat(n_tool(tool_settings, ca.vertcat(csi, theta)), 0)

    # symbolic family of tool surfaces
    G = ggt(machine_settings, phi)
    p_gear_expr = G @ p_tool_expr
    n_gear_expr = G @ n_tool_expr
    p_gear_fun = ca.Function('pg', [enveloping_triplet], [p_gear_expr[0:3]])
    n_gear_fun = ca.Function('ng', [enveloping_triplet], [n_gear_expr[0:3]])

    # congruence equations
    eq_congruence = p_gear_expr[0:3] - surface_point
    eq_congruence_fun = ca.Function('eq_congr', [ca.vertcat(enveloping_triplet, surface_point)], [eq_congruence])

    # transversal cone equation
    lc = A0 - Fw + c
    transversal_angle = front_angle + (back_angle - front_angle)*(c/Fw)
    sc = ca.tan(transversal_angle)*lc*ca.sin(pitch_angle)
    pc = - pitch_apex + lc*ca.cos(pitch_angle) + sc
    eq_transversal = ( x**2 + y**2 - ((-z + pc)*ca.tan(pi/2 - transversal_angle))**2 )*(ca.norm_1(transversal_angle) > 1e-5) + (-z + pc)*(ca.norm_1(transversal_angle) <= 1e-5)
    
    # equation of meshing
    eq_meshing = n_tool_expr.T @ Vgt(machine_settings, phi) @ p_tool_expr
    eq_meshing_fun = ca.Function('eq_meshing', [enveloping_triplet], [eq_meshing])

    # head cone equation
    eq_head = x**2 + y**2 - ((z + face_apex)*ca.tan(face_angle))**2

    # root cone equation
    eq_root = x**2 + y**2 - ((z + root_apex)*ca.tan(root_angle))**2

    # root sample system
    root_sys = ca.vertcat(eq_congruence, eq_meshing, eq_transversal)# eq_root)
    root_sys_jacobian = ca.jacobian(root_sys, ca.vertcat(theta, phi, surface_point))
    root_sys_fun = ca.Function('root_sys', [ca.vertcat(enveloping_triplet, surface_point), c], [root_sys])
    root_sys_jacobian_fun = ca.Function('root_sys_jac', [ca.vertcat(enveloping_triplet, surface_point), c], [root_sys_jacobian])

    # head sample system
    head_sys = ca.vertcat(eq_congruence, eq_meshing, eq_transversal, eq_head)
    head_sys_jacobian = ca.jacobian(head_sys, ca.vertcat(enveloping_triplet, surface_point))
    head_sys_fun = ca.Function('head_sys', [ca.vertcat(enveloping_triplet, surface_point), c], [head_sys])
    head_sys_jacobian_fun = ca.Function('head_sys_jac', [ca.vertcat(enveloping_triplet, surface_point), c], [head_sys_jacobian])

    # flank sample system
    flank_sys = ca.vertcat(eq_congruence, eq_meshing, eq_transversal)
    flank_sys_jacobian = ca.jacobian(flank_sys, ca.vertcat(theta, phi, surface_point))
    flank_sys_fun = ca.Function('flank_sys', [ca.vertcat(theta, phi, surface_point), csi, c], [flank_sys])
    flank_sys_jacobian_fun = ca.Function('flank_sys_jac', [ca.vertcat(theta, phi, surface_point), csi, c], [flank_sys_jacobian])

    # the flank sampling employs casadi rootfinder, 
    # root and head sampling will be carried out by the more accurate (hopefully) scipy's fsolve
    # problem = {'x': ca.vertcat(theta, phi, surface_point), 'p': ca.vertcat(csi, c), 'g': root_sys}
    # solver_root = ca.rootfinder('solver_flank', 'newton', problem, {'error_on_fail' : False})

    problem = {'x': ca.vertcat(theta, phi, surface_point), 'p': ca.vertcat(csi, c), 'g': flank_sys}
    solver_flank = ca.rootfinder('solver_flank', 'newton', problem, {'error_on_fail' : False})
    problem = {'x': ca.vertcat(csi, theta, phi, surface_point), 'p': c, 'g': head_sys}
    solver_head = ca.rootfinder('solver_head', 'newton', problem, {'error_on_fail' : False})

    # initialize solutions
    surface_sol = np.zeros((6, n_face, n_fillet + n_prof -1))
    flank_fillet_sol = np.zeros((6, n_face))
    root_sol = np.zeros((6, n_face))
    head_sol = np.zeros((6, n_face))
    toe_sol = np.zeros((6, n_prof - 2))
    heel_sol = np.zeros((6, n_prof - 2))

    guess = ca.reshape(ca.DM(triplet_guess), 3, 1) # it should be a column array
    guess[0] = 0.3
    point_guess = p_gear_fun(guess)
    guess = ca.vertcat(guess, point_guess[0:3])

    # root cone sampling
    for ii in range (0, n_face):
        c_value = Fw*(ii)/(n_face - 1)
        sol = fsolve(lambda x, c: root_sys_fun(ca.vertcat(0, x), c).full().squeeze(), x0 = guess[1:], args=(c_value), xtol = 1e-5, col_deriv=False, fprime = lambda x, c: root_sys_jacobian_fun(ca.vertcat(0, x), c).full().squeeze())
        surface_sol[:, ii, 0] = np.r_[0, sol]
        root_sol[:, ii] = np.r_[0, sol]
        guess = np.r_[0, sol]

    # fillet sampling
    for ii in range(0, n_face):
        c_value = Fw*ii/(n_face - 1)
        guess = surface_sol[1:, ii, 0]

        for kk in range(1, n_fillet): # first row is the root line
            csi_value = csi_edge_blade*(kk)/(n_fillet)
            result = solver_flank(x0 = guess, p = ca.vertcat(csi_value[0], c_value))
            sol = result['x'].full()
            surface_sol[:, ii, kk] = np.r_[csi_value, sol].flatten()
            guess = sol
            if kk == n_fillet-1:   # flank-fillet transition line
                flank_fillet_sol[:, ii] = surface_sol[:, ii, kk]

        if ii == 1: # the first profile line will brute force the sampling to obtain the head guess solution
            p =  p_gear_fun(flank_fillet_sol[0:3, ii])
            jj = n_fillet
            guess_head = guess
            guess_head[1] = 0
            R = ca.sqrt(p[0]**2 + p[1]**2).full()
            z = p[2].full()
            # we check if either we pass the z value or the radial value. It depends on the value of the face angle
            while (front_angle*180/np.pi>45)*(z >= z_head) or (front_angle*180/np.pi<=45)*(R <= R_head):
                csi_value = csi_edge_blade*(jj)/(20)
                sol = fsolve(lambda x, csi, c: flank_sys_fun(x, csi, c).full().squeeze(),\
                               fprime = lambda x, csi, c: flank_sys_jacobian_fun(x, csi, c).full().squeeze(),\
                               x0 = guess_head,\
                               args=(csi_value, 0), xtol = 1e-5, col_deriv=False
                               )
                # res = solver_flank(x0 = guess, p = np.r_[csi_value[0], c_value])
                # sol = res['x'].full().squeeze()
                guess_head = sol
                p = p_gear_fun(np.r_[csi_value[0], guess_head[0:2]])
                R = ca.sqrt(p[0]**2 + p[1]**2).full()
                z = p[2].full()
                jj += 1
            guess_head = np.r_[csi_value[0], guess_head.squeeze()]

    guess = guess_head

    # head cone sampling

    for ii in range(0, n_face):
        c_value = Fw*ii/(n_face-1)
        res = solver_head(x0 = guess, p = c_value)
        sol = res['x'].full().reshape(-1,)
        surface_sol[:, ii, -1] = sol
        guess = sol
        head_sol[:, ii] = sol

    tip_csi_values = surface_sol[0,:,-1]
    csi_theta_phi_guesses = surface_sol[:, :, -1]

    # flank sampling
    for ii in range(0, n_face):
        c_value = Fw*ii/(n_face-1)
        guess = csi_theta_phi_guesses[1:, ii]

        # active flank
        for jj in reversed(range(1, n_prof-1)): # last flank points are the head points, first flank points are the last fillet points
            csi_value = csi_edge_blade + (tip_csi_values[ii] - csi_edge_blade)*jj/(n_prof-1)
            res = solver_flank(x0 = guess, p = ca.vertcat(csi_value[0], c_value))
            sol = res['x'].full()
            surface_sol[:, ii, jj + n_fillet-1] = np.r_[csi_value, sol].squeeze()
            guess = sol
            if ii == 0:        # we are sampling the toe profile
                toe_sol[:, jj-1] = surface_sol[:, ii, jj + n_fillet-1]
            if ii == n_face-1: # we are sampling the heel profile
                heel_sol[:, jj-1] = surface_sol[:, ii, jj + n_fillet-1]

    # matrix structure of the solution
    #            Face   toe   ->   heel
    # surfVars = MAT : [ [x;y;z] [x;y;z] ... ]  root    ; [x;y;z] = [csi;theta;phi]
    #                  [ [x;y;z] [x;y;z] ... ]   |
    #                  [ [x;y;z] [x;y;z] ... ]   |
    #                  [ ...     ...     ... ]  tip

    # extract and save values
    surfVars   = surface_sol[0:3, :, :]
    filletVars = flank_fillet_sol[0:3, :]
    rootVars   = root_sol[0:3, :]
    headVars   = head_sol[0:3, :]
    toeVars    = toe_sol[0:3, :]
    heelVars   = heel_sol[0:3, :]

    points = p_gear_fun(reduce_2d(surfVars)).full()
    normals = n_gear_fun(reduce_2d(surfVars)).full()

    pointsFillet = p_gear_fun(reduce_2d(filletVars)).full()
    normalsFillet = n_gear_fun(reduce_2d(filletVars)).full()

    pointsRoot = p_gear_fun(reduce_2d(rootVars)).full()
    normalsRoot = n_gear_fun(reduce_2d(rootVars)).full()

    pointsHead = np.fliplr(p_gear_fun(reduce_2d(headVars)).full())
    pointsToe = np.fliplr(p_gear_fun(reduce_2d(toeVars)).full())
    pointsHeel = p_gear_fun(reduce_2d(heelVars)).full()

    normalsHead = np.fliplr(n_gear_fun(reduce_2d(headVars)).full())
    normalsToe = np.fliplr(n_gear_fun(reduce_2d(toeVars)).full())
    normalsHeel = n_gear_fun(reduce_2d(heelVars)).full()

    pointsBounds = [pointsFillet, pointsHeel, pointsHead, pointsToe]
    normalsBounds = [normalsFillet, normalsHeel, normalsHead, normalsToe]

    return surfVars, filletVars, points, normals, pointsFillet, normalsFillet, pointsRoot, normalsRoot, rootVars, pointsBounds, normalsBounds

def rz_sampling_casadi_formate(R, Z, data: DesignData, member, flank, triplet_guess):

    R = np.atleast_2d(R)
    Z = np.atleast_2d(Z)

    system_hand = data.system_data.HAND

    # Kinematics definition 
    machine_par_matrix = data.extract_machine_settings_matrix(member, flank)

    ggt, Vgt, Vgt_spatial = casadi_machine_kinematics(member, system_hand)
    G = ca.full(ggt(machine_par_matrix, 0))

    # Tool definition
    toolvec = data.extract_tool_settings(member, flank)
    toolvec = np.concatenate(toolvec)  # Convert list of arrays to a single array
    pT, nT = casadi_tool_fun(flank)

    # Define the system function
    def system(x, R, Z):
        p_tool = np.concatenate([pT(toolvec, [x[0], x[1]]), [1]])
        pg = G @ p_tool  # Matrix multiplication
        out = ca.vertcat(
            pg[0]**2 + pg[1]**2 - R**2,
            pg[2]**2 - Z**2,
        )
        return out
    
    # Define symbolic variables
    csi = ca.SX.sym('csi')
    theta = ca.SX.sym('theta')
    z_sym = ca.SX.sym('z')
    R_sym = ca.SX.sym('R')

    expr = system(ca.SX.vertcat(csi, theta), R_sym, z_sym)

    problem_rootfinder = {'x': ca.SX.vertcat(csi, theta), 'p': ca.SX.vertcat(R_sym, z_sym), 'g': expr}
    SolverRoot = ca.rootfinder('solver', 'newton', problem_rootfinder, {'error_on_fail': False})

    # Instantiate Ipopt solver in case rootfinder fails
    options_ipopt = IPOPT_global_options()
    options_ipopt['ipopt']['nlp_scaling_method'] = 'gradient-based'

    problem_ipopt = {'x': ca.SX.vertcat(csi, theta), 'p': ca.SX.vertcat(R_sym, z_sym), 'f': 0.5 * expr.T @ expr}
    SolverIpopt = ca.nlpsol('S', 'ipopt', problem_ipopt, options_ipopt)

    r, c = Z.shape
    xyzbase = np.full((4, r, c), np.nan)
    normalsbase = np.full((4, r, c), np.nan)
    triplets = np.full((3, r, c), 0)

    if triplet_guess is None or len(triplet_guess) == 0: # If no initial guess is provided
        guess = initial_guess_from_data(data, member, flank)

    for ii in range(r):
        for jj in range(c):
            if triplet_guess:
                if isinstance(triplet_guess, list):  # Check if it's a list
                    triplet_interp = triplet_guess
                    csi = triplet_interp[0](Z[ii, jj], R[ii, jj])
                    theta = triplet_interp[1](Z[ii, jj], R[ii, jj])
                    phi = triplet_interp[2](Z[ii, jj], R[ii, jj])
                    guess = np.array([csi, theta, phi])
                elif triplet_guess.ndim == 1:
                    guess = triplet_guess[:, 0]
                else:
                    guess = triplet_guess[:, ii, jj]

            guess = guess[0:2]
            # Trying with the findroot Newton solver
            try:
                res = SolverRoot(x0=guess[0:2], p=np.array([R[ii, jj], Z[ii, jj]]))
                res = ca.full(res['x']).T
            except Exception:
                res = SolverIpopt(x0=guess[0:2], p=np.array([R[ii, jj], Z[ii, jj]]),
                                  ubx=guess + np.array([5, np.pi, np.pi]),
                                  lbx=guess * np.array([0, 1, 1]) - np.array([0, np.pi, np.pi]))
                res = ca.full(res['x']).T

            triplets[0:2, ii, jj] = res[0:2]
            xyzbase[:, ii, jj] = G @ ca.full(np.concatenate([pT(toolvec, [res[0], res[1]]), [1]]))
            normalsbase[:, ii, jj] = G @ ca.full(np.concatenate([nT(toolvec, [res[0], res[1]]), [0]]))

            if not triplet_guess:
                guess = triplets[0:2, ii, jj]

    return xyzbase, normalsbase, triplets
    return
    
def rz_sampling_casadi(R, Z, data: DesignData, member, flank, triplet_guess = None, sb_machine = False):
    """
    This function samples the surface of a gear or a blade using the R-Z coordinates of the surface.
    
    Parameters
    ----------
    R : np.array
        Array of the radial coordinates of the surface.
        Z : np.array
        Array of the axial coordinates of the surface.
        
        data : dict
        Dictionary containing the data of the system.
        
        member : str
        Member of the system to be sampled. It can be 'gear' or 'blade'.
        
        flank : str
        Flank of the member to be sampled. It can be 'concave' or 'convex'.
        
        triplet_guess : np.array, optional
        Initial guess for the triplet of parameters. The default is None.
        
        sb_machine : bool, optional
        Check if the completing method is active (both flanks genereated at the same time). The default is False.
        
        Returns
        -------
        xyzbase : np.array
        Array of the coordinates of the sampled points.
        
        normalsbase : np.array
        Array of the normals of the sampled points.
        
        triplets : np.array
        Array of the triplets of parameters used to sample the points.
        
        """
    # Check if z and R are 2D arrays
    R = np.atleast_2d(R)
    Z = np.atleast_2d(Z)

    if data.gear_common_data.gen_type.lower() == 'formate' and member == 'gear':
        return rz_sampling_casadi_formate(R, Z, data, member, flank, triplet_guess)

    system_hand = data.system_data.HAND

    # Kinematics definition
    if sb_machine:
        machine_par_matrix = data.extract_machine_settings_matrix(member, 'concave')
    else:
        machine_par_matrix = data.extract_machine_settings_matrix(member, flank)

    ggt, Vgt, Vgt_spatial = casadi_machine_kinematics(member, system_hand)

    # Tool definition
    toolvec = data.extract_tool_settings(member, flank)
    toolvec = np.array(toolvec) 
    pT, nT, _ = casadi_tool_fun(flank)

    # Define symbolic variables
    csi = ca.SX.sym('csi')
    theta = ca.SX.sym('theta')
    phi = ca.SX.sym('phi')
    z_sym = ca.SX.sym('z')
    R_sym = ca.SX.sym('R')
    p_sym = ca.SX.sym('p', 3, 1)
    n_sym = ca.SX.sym('n', 3, 1)

    # Define the rootfinidng system
    p_tool = ca.vertcat(pT(toolvec, ca.vertcat(csi, theta)), 1)
    n_tool = ca.vertcat(nT(toolvec, ca.vertcat(csi, theta)), 0)
    pg = ggt(machine_par_matrix, phi) @ p_tool
    ng = ggt(machine_par_matrix, phi) @ n_tool
    expr = ca.vertcat(
        pg[0]**2 + pg[1]**2 - R_sym**2,
        pg[2]**2 - z_sym**2,
        ng.T @ Vgt_spatial(machine_par_matrix, phi) @ pg, # Meshing condition
        pg[0:3] - p_sym, # Congruence condition
        ng[0:3] - n_sym  # Congruence condition
    )

    problem_rootfinder = {'x': ca.vertcat(csi, theta, phi, p_sym, n_sym), 'p': ca.vertcat(R_sym, z_sym), 'g': expr}
    SolverRoot = ca.rootfinder('solver', 'newton', problem_rootfinder, {'error_on_fail': False})

    # Instantiate Ipopt solver in case rootfinder fails
    options_ipopt = IPOPT_global_options()
    options_ipopt['ipopt']['nlp_scaling_method'] = 'gradient-based'

    problem_ipopt = {'x': ca.vertcat(csi, theta, phi, p_sym, n_sym), 'p': ca.vertcat(R_sym, z_sym), 'f': 0.5 * expr.T @ expr}
    SolverIpopt = ca.nlpsol('S', 'ipopt', problem_ipopt, options_ipopt)

    r, c = Z.shape
    xyzbase = np.full((4, r, c), np.nan)
    normalsbase = np.full((4, r, c), np.nan)
    triplets = np.full((3, r, c), np.nan)

    if triplet_guess is None or len(triplet_guess) == 0: # If no initial guess is provided
        guess = initial_guess_from_data(data, member, flank)

    for ii in range(r):
        for jj in range(c):
            if triplet_guess is not None or len(triplet_guess)>0:
                if isinstance(triplet_guess, list):  # Check if it's a list
                    triplet_interp = triplet_guess
                    csi = triplet_interp[0](Z[ii, jj], R[ii, jj])
                    theta = triplet_interp[1](Z[ii, jj], R[ii, jj])
                    phi = triplet_interp[2](Z[ii, jj], R[ii, jj])
                    guess = np.array([csi, theta, phi])
                elif triplet_guess.ndim == 1:
                    guess = triplet_guess
                else:
                    guess = triplet_guess[:, ii, jj]

            # Trying with the findroot Newton solver
            G_num = ggt(machine_par_matrix, guess[2])
            p_num = (G_num @ ca.vertcat(pT(toolvec, [guess[0], guess[1]]), 1)).full()
            n_num = (G_num @ ca.vertcat(nT(toolvec, [guess[0], guess[1]]), 0)).full()

            guess_sparse = np.hstack([guess.flatten(), p_num[0:3].flatten(), n_num[0:3].flatten()])
            try:
                res = SolverRoot(x0=guess_sparse, p=np.array([R[ii, jj], Z[ii, jj]]))
                res = res['x'].full()
            except Exception:
                res = SolverIpopt(x0=guess_sparse, p=np.array([R[ii, jj], Z[ii, jj]]),
                                  ubx=guess + np.array([5, np.pi, np.pi]),
                                  lbx=guess * np.array([0, 1, 1]) - np.array([0, np.pi, np.pi]))
                res = res['x'].full()

            triplets[:, ii, jj] = res[0:3,0]
            xyzbase[:, ii, jj] = (ggt(machine_par_matrix, res[2]) @ca.vertcat(pT(toolvec, [res[0], res[1]]), 1)).full().flatten()
            normalsbase[:, ii, jj] = (ggt(machine_par_matrix, res[2]) @ ca.vertcat(nT(toolvec, [res[0], res[1]]), 0)).full().flatten()

            if triplet_guess is None or len(triplet_guess) == 0:
                guess[:] = triplets[:, ii, jj]

    return xyzbase, normalsbase, triplets

def rz_sampling_NURBS_casadi(data: DesignData, member, flank, z, R, triplets):


    raise Exception("NURBS sampling not yet implemented")
    return 

def pinion_conjugate_to_gear(data: DesignData, flank, zRgear, EPGalpha, triplets_gear, interpolated_triplets_pin, offset_psi):

    hypoid_offset = data.system_data.hypoid_offset
    HAND = data.system_data.HAND
    ratio = data.system_data.ratio
    shaft_angle = data.system_data.shaft_angle*np.pi/180

    if flank.lower() == 'concave':
        gear_flank = 'convex'
        pinion_flank = 'concave'
    else:
        gear_flank = 'concave'
        pinion_flank = 'convex'

    # tool geometry and machine kinematics
    tool_settings = data.extract_tool_settings('gear', gear_flank)
    p_tool, n_tool, _ = casadi_tool_fun(gear_flank, toprem=False, flankrem=False)
    raw_machine_settings = data.extract_machine_settings_matrix('gear', gear_flank)
    ggt, Vgt, Vgt_spatial = casadi_machine_kinematics('gear', HAND)

    # gear-pinion kinematics
    Tpg, Vpg_g, Tfp, Tfg, Vpg_p = gear_to_pinion_kinematics(hypoid_offset, shaft_angle, HAND, EPGalpha)

    # extract data points
    z = zRgear[:, 0]
    R = zRgear[:, 1]
    psi0 = 0
    psi_G0 = 0

    num_points = z.shape[0]
    psi_G = np.zeros(num_points)
    psi_P = np.zeros(num_points)
    gear_points = np.zeros((3, num_points))
    gear_normals = np.zeros((3, num_points))
    conjugate_points = np.zeros((4, num_points))
    conjugate_normals = np.zeros((4, num_points))
    csithetaphi = np.zeros((3, num_points))
    Vpg_g_num = np.zeros((4, num_points))
    v_pg_p = np.zeros((4, num_points))
    omega = np.zeros((3, num_points))

    # casadi variables
    csi = ca.SX.sym('csi')
    theta = ca.SX.sym('theta')
    phi = ca.SX.sym('phi')
    R_sym = ca.SX.sym('R')
    z_sym = ca.SX.sym('z')
    psi = ca.SX.sym('psi')
    pG_sym = ca.SX.sym('pG', 3, 1)
    nG_sym = ca.SX.sym('nG', 3, 1)

    G = ggt(raw_machine_settings, phi)
    p_tool_expr = ca.vertcat(p_tool(tool_settings, ca.vertcat(csi, theta)), 1)
    n_tool_expr = ca.vertcat(n_tool(tool_settings, ca.vertcat(csi, theta)), 0)
    pG_expr = G @ p_tool_expr
    nG_expr = G @ n_tool_expr
    pG_fun = ca.Function('pg', [ca.vertcat(csi, theta, phi)], [pG_expr[0:3]])
    nG_fun = ca.Function('ng', [ca.vertcat(csi, theta, phi)], [nG_expr[0:3]])
    pg_num = pG_fun(ca.vertcat(triplets_gear[0, :].reshape(1, -1), triplets_gear[1, :].reshape(1, -1), triplets_gear[2, :].reshape(1, -1))).full()
    ng_num = nG_fun(ca.vertcat(triplets_gear[0, :].reshape(1, -1), triplets_gear[1, :].reshape(1, -1), triplets_gear[2, :].reshape(1, -1))).full()


    equations = ca.vertcat(
        pG_sym[0]**2 + pG_sym[1]**2 - R_sym**2,
        pG_sym[2]**2 - z_sym**2,
        ca.vertcat(nG_sym, 0).T @ Vgt_spatial(raw_machine_settings, phi) @ ca.vertcat(pG_sym, 1),
        ca.vertcat(nG_sym, 0).T @ Vpg_g(psi - psi0, ratio*(psi - psi0), 1, ratio) @ ca.vertcat(pG_sym, 1)/100,
        pG_sym - pG_expr[0:3],
        nG_sym - nG_expr[0:3]
    )

    # define the rootfinder problem
    problem = {'x': ca.vertcat(csi, theta, phi, psi, pG_sym, nG_sym), 'p': ca.vertcat(R_sym, z_sym), 'g': equations}
    solver = ca.rootfinder('solver', 'newton', problem, {'error_on_fail': False})
    
    phi_P0 = 0
    triplets_gear[0, :] = triplets_gear[0, :] + 0.2
    for ii in range(num_points):
        x0 = np.vstack([triplets_gear[:, ii].reshape(-1, 1), psi_G[ii].reshape(-1, 1), pg_num[0:3, ii].reshape(-1, 1), ng_num[0:3, ii].reshape(-1, 1)])
        p = np.array([R[ii], z[ii]]).reshape(-1, 1)
        sol = solver(x0=x0, p=p)

        # if solver.stats()['success'] == False:
        #     raise Exception(f'Solver did not converge at point number {ii}')
        
        res = sol['x'].full()
        csithetaphi[:, ii] = res[0:3].flatten()
        gear_points[:, ii] = res[4:7].flatten()
        gear_normals[:, ii] = res[7:10].flatten()

        psi_G[ii] = res[3]
        psi_P[ii] = ratio * (psi_G[ii] - psi_G0)
        T = Tpg(psi_G[ii] - psi_G0, psi_P[ii])
        conjugate_points[:, ii] = (T @ np.hstack((gear_points[:, ii], 1)).reshape(-1, 1)).squeeze()
        conjugate_normals[:, ii] = (T @ np.hstack([gear_normals[:, ii], 0]).reshape(-1, 1)).squeeze()
        V = Vpg_g(psi_G[ii] - psi_G0, ratio*(psi_G[ii] - psi_G0), 1, ratio).full()
        omega[:, ii] = sc.vecForm(V[0:3, 0:3])
        Vpg_g_num[:, ii] = V @ np.hstack([gear_points[:, ii], 1])
        v_pg_p[:, ii] = T@(-Vpg_g_num[:, ii])

    Rpin = np.sqrt(np.sum(conjugate_points[0:2, :]**2, axis=0))
    Zpin = conjugate_points[2, :]
    zRpin = np.vstack([Zpin, Rpin])

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(gear_points[0, :], gear_points[1, :], gear_points[2, :])
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(conjugate_points[0, :], conjugate_points[1, :], conjugate_points[2, :])
    # plt.show()

    interp = [interpolated_triplets_pin['csi'], interpolated_triplets_pin['theta'], interpolated_triplets_pin['phi']]
    points_base, normals_base, triplets_base = rz_sampling_casadi(Rpin, Zpin, data, 'pinion', pinion_flank, interp)

    points_base = points_base.squeeze()
    normals_base = normals_base.squeeze()
    triplets_base = triplets_base.squeeze()

    s = -1
    if HAND.lower() == 'left':
        s = +1

    id_pt = int(np.floor(num_points/2))
    offset_psi = s*(np.arctan2(points_base[1, id_pt], points_base[0, id_pt]) - np.arctan2(conjugate_points[1, id_pt], conjugate_points[0, id_pt]))
    p = sc.TrotZ(offset_psi)@conjugate_points
    n = sc.TrotZ(offset_psi)@conjugate_normals


    angular_ease_off = s*(np.arctan2(points_base[1, :], points_base[0, :]) - np.arctan2(p[1, :], p[0, :]))
    
    return p, n, zRpin, triplets_base, psi_P, psi_G, angular_ease_off, v_pg_p, omega

def shaft_segment_computation(data:DesignData):
    
    gear_data = data.gear_common_data
    pinion_data = data.pinion_common_data

    gear_data = data.gear_common_data

    # compute gear data
    O_g = gear_data.OUTERCONEDIST
    gamma_g = gear_data.PITCHANGLE
    pA_g = gear_data.PITCHAPEX
    bA_g = gear_data.BASECONEAPEX
    gammab_g = gear_data.BASECONEANGLE
    Fw_g = gear_data.FACEWIDTH
    backA_g = gear_data.BACKANGLE
    frontA_g = gear_data.FRONTANGLE
    
    PB_g = np.array([O_g * np.cos(np.deg2rad(gamma_g)) - pA_g, 
                     O_g * np.sin(np.deg2rad(gamma_g))])  # [z, R]
    PF_g = np.array([(O_g - Fw_g) * np.cos(np.deg2rad(gamma_g)) - pA_g, 
                     (O_g - Fw_g) * np.sin(np.deg2rad(gamma_g))])  # [z, R]
    
    tangent_front = np.tan(np.deg2rad(90 - frontA_g))
    tangent_back = np.tan(np.deg2rad(90 - frontA_g))

    zB_g = (-np.tan(np.deg2rad(gammab_g)) * bA_g + PB_g[1] + tangent_back * PB_g[0]) / \
            (np.tan(np.deg2rad(gammab_g)) + tangent_back)
    zA_g = (-np.tan(np.deg2rad(gammab_g)) * bA_g + PF_g[1] + tangent_front * PF_g[0]) / \
            (np.tan(np.deg2rad(gammab_g)) + tangent_front)
    
    # check if front and back angle rise singularities
    if tangent_back == np.inf:
        zB_g =  PB_g[0]
    if tangent_front == np.inf:
        zA_g = PF_g[0]

    RB_g = np.tan(np.deg2rad(gammab_g)) * zB_g + np.tan(np.deg2rad(gammab_g)) * bA_g
    RA_g = np.tan(np.deg2rad(gammab_g)) * zA_g + np.tan(np.deg2rad(gammab_g)) * bA_g
    
    gear_data.ShaftzB = zB_g
    gear_data.ShaftRB = RB_g
    gear_data.ShaftzA = zA_g
    gear_data.ShaftRA = RA_g
    gear_data.ShaftDiA = RA_g * (2 - 0.85)  # diametro interno a ridosso del punto A
    gear_data.ShaftDiB = RB_g * (2 - 0.85)  # diametro interno a ridosso del punto B

    # compute pinion data
    O_p = pinion_data.OUTERCONEDIST
    gamma_p = pinion_data.PITCHANGLE
    pA_p = pinion_data.PITCHAPEX
    bA_p = pinion_data.BASECONEAPEX
    gammab_p = pinion_data.BASECONEANGLE
    Fw_p = pinion_data.FACEWIDTH
    backA_p = pinion_data.BACKANGLE
    frontA_p = pinion_data.FRONTANGLE
    
    # pitch cone-back cone intersection
    PB_p = np.array([O_p * np.cos(np.deg2rad(gamma_p)) - pA_p, 
                     O_p * np.sin(np.deg2rad(gamma_p))])  # [z, R]
    # pitch cone- face cone intersection
    PF_p = np.array([(O_p - Fw_p) * np.cos(np.deg2rad(gamma_p)) - pA_p, 
                     (O_p - Fw_p) * np.sin(np.deg2rad(gamma_p))])  # [z, R]
    
    tangent_front = np.tan(np.deg2rad(90 - frontA_p))
    tangent_back = np.tan(np.deg2rad(90 - backA_p))
    
    zB_p = (-np.tan(np.deg2rad(gammab_p)) * bA_p + PB_p[1] + tangent_back * PB_p[0]) / \
            (np.tan(np.deg2rad(gammab_p)) + tangent_back)
 
    zA_p = (-np.tan(np.deg2rad(gammab_p)) * bA_p + PF_p[1] + tangent_front * PF_p[0]) / \
            (np.tan(np.deg2rad(gammab_p)) + tangent_front)
    
    # check if front and back angle rise singularities
    if tangent_back == np.inf:
        zB_p =  PB_p[0]
    if tangent_front == np.inf:
        zA_p = PF_p[0]

    RB_p = np.tan(np.deg2rad(gammab_p)) * zB_p + np.tan(np.deg2rad(gammab_p)) * bA_p
    RA_p = np.tan(np.deg2rad(gammab_p)) * zA_p + np.tan(np.deg2rad(gammab_p)) * bA_p
    
    pinion_data.ShaftzB = zB_p
    pinion_data.ShaftRB = RB_p
    pinion_data.ShaftzA = zA_p
    pinion_data.ShaftRA = RA_p
    pinion_data.ShaftDiA = RA_p * (2 - 0.85)  # diametro interno a ridosso del punto A
    pinion_data.ShaftDiB = RB_p * (2 - 0.85)  # diametro interno a ridosso del punto B
    
    return data

def rz_boundaries_computation(data: DesignData, member):

    common_data = data.gear_common_data
    if member.lower() == 'pinion':
        common_data = data.pinion_common_data

    O = common_data.OUTERCONEDIST
    Fw = common_data.FACEWIDTH

    if common_data.MEANCONEDIST is None:
        common_data.MEANCONEDIST = O - Fw/2
        common_data.INNERCONEDIST = O - Fw

    face_angle = common_data.FACEANGLE
    pitch_angle = common_data.PITCHANGLE
    root_angle = common_data.ROOTANGLE  
    front_angle = common_data.FRONTANGLE
    back_angle = common_data.BACKANGLE
    base_angle = common_data.BASECONEANGLE

    pitch_apex = common_data.PITCHAPEX
    face_apex = common_data.FACEAPEX
    root_apex = common_data.ROOTAPEX
 
    # pitch-back cones intersection
    PB = np.array(
        [O*np.cos(np.deg2rad(pitch_angle)) - pitch_apex, # z
        O*np.sin(np.deg2rad(pitch_angle))]               # R
    )

    # pitch-face cones intersection
    PF = np.array(
        [(O-Fw)*np.cos(np.deg2rad(pitch_angle)) - pitch_apex, # z
        (O- Fw)*np.sin(np.deg2rad(pitch_angle))]              # R
    )

    P_center = (PB + PF)/2

    tangent_front = np.tan(np.deg2rad(90 - front_angle))
    tangent_back = np.tan(np.deg2rad(90 - back_angle))

    z_root_heel = (-np.tan(np.deg2rad(root_angle)) * root_apex + PB[1] + tangent_back * PB[0]) / \
            (np.tan(np.deg2rad(root_angle)) + tangent_back)
    z_root_toe = (-np.tan(np.deg2rad(root_angle)) * root_apex + PF[1] + tangent_front * PF[0]) / \
            (np.tan(np.deg2rad(root_angle)) + tangent_front)

    z_face_heel = (-np.tan(np.deg2rad(face_angle)) * face_apex + PB[1] + tangent_back * PB[0]) / \
            (np.tan(np.deg2rad(face_angle)) + tangent_back)
    z_face_toe = (-np.tan(np.deg2rad(face_angle)) * face_apex + PF[1] + tangent_front * PF[0]) / \
            (np.tan(np.deg2rad(face_angle)) + tangent_front)

    
    if tangent_front == np.inf:
        zA = PF[0]
    if tangent_back == np.inf:
        zB = PB[0]

    R_root_heel = np.tan(np.deg2rad(root_angle)) * z_root_heel + np.tan(np.deg2rad(root_angle)) * root_apex
    R_root_toe = np.tan(np.deg2rad(root_angle)) * z_root_toe + np.tan(np.deg2rad(root_angle)) * root_apex

    R_face_heel = np.tan(np.deg2rad(face_angle)) * z_face_heel + np.tan(np.deg2rad(face_angle)) * face_apex
    R_face_toe = np.tan(np.deg2rad(face_angle)) * z_face_toe + np.tan(np.deg2rad(face_angle)) * face_apex

    common_data.zROOTHEEL = z_root_heel
    common_data.RROOTHEEL = R_root_heel
    common_data.zROOTTOE = z_root_toe
    common_data.RROOTTOE = R_root_toe

    common_data.zFACEHEEL = z_face_heel
    common_data.RFACEHEEL = R_face_heel
    common_data.zFACETOE = z_face_toe
    common_data.RFACETOE = R_face_toe

    zR = np.array([
        [z_root_toe, R_root_toe],
        [z_root_heel, R_root_heel],
        [z_face_heel, R_face_heel],
        [z_face_toe, R_face_toe]]
    )
    return data, P_center, zR

def PCA_computation(data: DesignData, member, side, EPGalpha, boundary_points, boundary_points_other, n_face):
    """
    Perform PCA (Potential Contact Area) computation for hypoid gears.
    Parameters:
    data (dict): Dictionary containing system data and gear parameters.
    member (str): Specifies whether the member is 'gear' or 'pinion'.
    side (str): Specifies the side of the gear, either 'drive' or 'coast'.
    EPGalpha (float): Angle parameter for the PCA computation.
    boundary_points (tuple): Tuple containing boundary points and normals for the member.
    boundary_points_other (tuple): Tuple containing boundary points and normals for the other member.
    n_face (int): Number of face points.
    n_prof (int): Number of profile points.
    Returns:
    np.ndarray: Transformed points in the other member's coordinate system.
    """

    offset = data.system_data.hypoid_offset
    SIGMA = np.deg2rad(data.system_data.shaft_angle)
    u = data.system_data.ratio
    hand = data.system_data.hand
    pinion_common_data = data.pinion_common_data
    gear_common_data = data.gear_common_data
    nP = pinion_common_data.NTEETH
    nG = gear_common_data.NTEETH

    points_other = np.hstack(boundary_points_other[0]) # extracting the edge points and collecting them in a single array
    points_other = np.vstack((points_other, np.ones(points_other.shape[1])))  # Add row of ones
    normals_other = np.hstack(boundary_points_other[1])
    normals_other = np.vstack((normals_other, np.zeros(normals_other.shape[1])))  # Add row of zeros

    if member.lower() == 'gear':
        otherFlank = 'concave' if side.lower() == 'drive' else 'convex'
        otherMember = 'pinion'


        zR = pin_to_gear_rz(data, points_other, normals_other, EPGalpha, guess = 2 * np.pi / nP / 3)

        # check if the sampled face points are inside the gear, otherwise it means we sampled the wrong branch of the envelope
        """
                     corner point
                        |
          midpoint (A)  | 
                |       |
                v       v
        *-------*-------* <- face points
        |               |
        |               |
        """
        points = boundary_points[0][2] # extracting the face points
        counter = 0
        midface = n_face // 2
        corner = 0
        C = zR[midface, :]
        A3D = points[:3, midface]
        B3D = points[:3, corner]
        A = [A3D[2], np.sqrt(A3D[0]**2 + A3D[1]**2)]
        B = [B3D[2], np.sqrt(B3D[0]**2 + B3D[1]**2)]
        AB = np.array(B) - np.array(A)
        AC = np.array(C) - np.array(A)

        check = AC[0] * AB[1] - AC[1] * AB[0]

        while check < 0:
            zR = pin_to_gear_rz(data, points_other, normals_other, EPGalpha, guess = -2 * np.pi / nP / 4 - 2 * np.pi / nP / 8 * counter)
            C = zR[34, :]
            AC = np.array(C) - np.array(A)
            check = AC[0] * AB[1] - AC[1] * AB[0]
            counter += 1
            if counter == 100:
                break

    else:  # member is 'pinion'
        otherFlank = 'convex' if side.lower() == 'drive' else 'concave'
        otherMember = 'gear'

        points = boundary_points[0][2] # extracting the face points
        zR = gear_to_pin_rz(data, points_other, normals_other, EPGalpha, guess = 2 * np.pi / 2 / nG)

        if np.min(zR[:, 0]) <= np.min(points[2, :]) * 0.9:
            zR = gear_to_pin_rz(data, points_other, normals_other, EPGalpha, guess = -2 * np.pi / 2 / nG)

    zRinOther = zR
    return zRinOther

def pin_to_gear_rz(data: DesignData, points_pinion, normals_pinion, EPGalpha, guess = 0):
    offset = data.system_data.hypoid_offset
    SIGMA = np.deg2rad(data.system_data.shaft_angle)
    u = data.system_data.ratio
    hand = data.system_data.hand
    

    if not EPGalpha or EPGalpha is None:
        EPGalpha = np.array([0, 0, 0, 0])
    
    Tpg, Vpg, _, _, Vgp = gear_to_pinion_kinematics(offset, SIGMA, hand, EPGalpha)
    Tgp = lambda phiP, phiG: sc.rigidInverse(Tpg(phiP, phiG))

    phiG = ca.SX.sym('phiG')
    p_sym = ca.SX.sym('p', 3, 1)
    n_sym = ca.SX.sym('n', 3, 1)

    sys_expr = ca.vertcat(n_sym, 0).T @ Vgp(phiG, u*phiG, 1, u) @ ca.vertcat(p_sym, 1)
    f = sys_expr
    jacobian = None

    p_gear = np.full((4, points_pinion.shape[1]), np.nan)
    for ii in range(points_pinion.shape[1]):
        p = points_pinion[:, ii]
        n = normals_pinion[:, ii]
        sol, jacobian, f = fsolve_casadi(f, phiG, ca.vertcat(p_sym, n_sym), guess, ca.vertcat(p[0:3], n[0:3]), jac_fun = jacobian)
        guess = sol
        p_gear[:, ii] = Tgp(sol[0], u*sol[0]) @ p
    
    Rg = np.sqrt(p_gear[0, :]**2 + p_gear[1, :]**2)
    zg = p_gear[2, :]
    
    return np.c_[zg, Rg]

def gear_to_pin_rz(data: DesignData, points_gear, normals_gear, EPGalpha, guess = 0):
    offset = data.system_data.hypoid_offset
    SIGMA = np.deg2rad(data.system_data.shaft_angle)
    hand = data.system_data.hand
    u = data.system_data.ratio

    if not EPGalpha or EPGalpha is None:
        EPGalpha = np.array([0, 0, 0, 0])
    
    Tpg, Vpg, _, _, _ = gear_to_pinion_kinematics(offset, SIGMA, hand, EPGalpha)

    phiP = ca.SX.sym('phiP')
    p_sym = ca.SX.sym('p', 3, 1)
    n_sym = ca.SX.sym('n', 3, 1)

    sys_expr = ca.vertcat(n_sym, 0).T @ Vpg(phiP, u*phiP, 1, u) @ ca.vertcat(p_sym, 1)
    f = sys_expr
    jacobian = None

    p_pinion = np.full((4, points_gear.shape[1]), np.nan)
    for ii in range(points_gear.shape[1]):
        p = points_gear[:, ii]
        n = normals_gear[:, ii]
        sol, jacobian, f = fsolve_casadi(f, phiP, ca.vertcat(p_sym, n_sym), guess, ca.vertcat(p[0:3], n[0:3]), jac_fun = jacobian)
        guess = sol
        p_pinion[:, ii] = Tpg(sol[0], u*sol[0]) @ p
    
    Rp = np.sqrt(p_pinion[0, :]**2 + p_pinion[1, :]**2)
    zp = p_pinion[2, :]

    return np.c_[zp, Rp]

def zr_activeflank_bounds(data: DesignData, member, flank, zr_fillet):

    common_data = data.gear_common_data
    if member.lower() == 'pinion':
        common_data = data.pinion_common_data

    # root points
    zr = np.vstack((zr_fillet[0, :], zr_fillet[-1, :]))

    #append face points
    zr = np.vstack((zr, common_data.zFACEHEEL, common_data.RFACEHEEL))
    zr = np.vstack((zr, common_data.zFACETOE, common_data.RFACETOE))

    return zr
