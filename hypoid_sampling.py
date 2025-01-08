from casadi.casadi import exp
import numpy as np
import casadi as ca
import screwCalculus as sc
from scipy.optimize import fsolve # fsolve(f, x0=x0, fprime=jacobian)
from math import sqrt, pi, atan, cos, sin, acos, asin, tan
from utils import *

from hypoid_kinematics import *
from hypoid_functions import *
from hypoid_utils import *
from solvers import fsolve_casadi



def surface_sampling_casadi(data, member, flank, sampling_size, triplet_guess = None, spreadblade = False, FW_vec = None):
    n_face = sampling_size[0]
    n_prof = sampling_size[1]
    n_fillet = sampling_size[2]
    HAND = data['SystemData']['HAND']
    blank_settings = list(assign_Blank_Par(data, member))

    tool_settings = assign_tool_par(data, member, flank)

    raw_machine_settings = assign_machine_par(data, member, flank)
    if spreadblade:
        raw_machine_settings = assign_machine_par(data, member, 'concave')

    if triplet_guess is None:
        triplet_guess = initial_guess_from_data(data, member, flank)
    
    common_field_name, sub_common_field_name = get_data_field_names(member, flank, fields='common')
    if member.lower() == 'gear' and data[common_field_name][f'{sub_common_field_name}GenType'].lower() == 'formate':
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

def surface_from_rz_casadi_formate(R, Z, data, member, flank, triplet_guess):
    system_hand = data['SystemData']['HAND']

    # Kinematics definition 
    machine_par_matrix = assign_machine_par(data, member, 'concave')

    ggt, Vgt, Vgt_spatial = casadi_machine_kinematics(member, system_hand)
    G = ca.full(ggt(machine_par_matrix, 0))

    # Tool definition
    toolvec = assign_tool_par(data, member, flank)
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

    if not triplet_guess:
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
    
def surface_from_rz_casadi(R, Z, data, member, flank, triplet_guess = None, sb_machine = False):

    if data['GearCommonData']['GenType'].lower() == 'formate' and member == 'gear':
        return surface_from_rz_casadi_formate(R, Z, data, member, flank, triplet_guess)

    system_hand = data['SystemData']['HAND']

    # Kinematics definition
    if sb_machine:
        machine_par_matrix = assign_machine_par(data, member, 'concave')
    else:
        machine_par_matrix = assign_machine_par(data, member, flank)

    ggt, Vgt, Vgt_spatial = casadi_machine_kinematics(member, system_hand)

    # Tool definition
    toolvec = assign_tool_par(data, member, flank)
    toolvec = np.concatenate(toolvec)  # Convert list of arrays to a single array
    pT, nT = casadi_tool_fun(flank)

    # Define the system function
    def system(x, R, Z):
        p_tool = np.concatenate([pT(toolvec, [x[0], x[1]]), [1]])
        n_tool = np.concatenate([nT(toolvec, [x[0], x[1]]), [0]])
        px = x[3:6]
        nx = x[6:9]
        pg = ggt(machine_par_matrix, x[2]) @ p_tool  # Matrix multiplication
        ng = ggt(machine_par_matrix, x[2]) @ n_tool  # Matrix multiplication
        out = ca.vertcat(
            pg[0]**2 + pg[1]**2 - R**2,
            pg[2]**2 - Z**2,
            ng.T @ Vgt_spatial(machine_par_matrix, x[2]) @ pg,
            pg[0:3] - px,
            ng[0:3] - nx
        )
        return out

    # Define symbolic variables
    csi = ca.SX.sym('csi')
    theta = ca.SX.sym('theta')
    phi = ca.SX.sym('phi')
    z_sym = ca.SX.sym('z')
    R_sym = ca.SX.sym('R')
    p_sym = ca.SX.sym('p', 3, 1)
    n_sym = ca.SX.sym('n', 3, 1)

    expr = system(ca.SX.vertcat(csi, theta, phi, p_sym, n_sym), R_sym, z_sym)

    problem_rootfinder = {'x': ca.SX.vertcat(csi, theta, phi, p_sym, n_sym), 'p': ca.SX.vertcat(R_sym, z_sym), 'g': expr}
    SolverRoot = ca.rootfinder('solver', 'newton', problem_rootfinder, {'error_on_fail': False})

    # Instantiate Ipopt solver in case rootfinder fails
    options_ipopt = IPOPT_global_options()
    options_ipopt['ipopt']['nlp_scaling_method'] = 'gradient-based'

    problem_ipopt = {'x': ca.SX.vertcat(csi, theta, phi, p_sym, n_sym), 'p': ca.SX.vertcat(R_sym, z_sym), 'f': 0.5 * expr.T @ expr}
    SolverIpopt = ca.nlpsol('S', 'ipopt', problem_ipopt, options_ipopt)

    r, c = Z.shape
    xyzbase = np.full((4, r, c), np.nan)
    normalsbase = np.full((4, r, c), np.nan)
    triplets = np.full((3, r, c), np.nan)

    if not triplet_guess:
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

            # Trying with the findroot Newton solver
            G_num = ggt(guess[2])
            p_num = G_num @ np.concatenate([pT(toolvec, [guess[0], guess[1]]), [1]])
            n_num = G_num @ np.concatenate([nT(toolvec, [guess[0], guess[1]]), [0]])

            guess_sparse = np.concatenate([guess, p_num[0:3], n_num[0:3]])
            try:
                res = SolverRoot(x0=guess_sparse, p=np.array([R[ii, jj], Z[ii, jj]]))
                res = ca.full(res['x']).T
            except Exception:
                res = SolverIpopt(x0=guess_sparse, p=np.array([R[ii, jj], Z[ii, jj]]),
                                  ubx=guess + np.array([5, np.pi, np.pi]),
                                  lbx=guess * np.array([0, 1, 1]) - np.array([0, np.pi, np.pi]))
                res = ca.full(res['x']).T

            triplets[:, ii, jj] = res[0:3]
            xyzbase[:, ii, jj] = ca.full(ggt(res[2]) @ np.concatenate([pT(toolvec, [res[0], res[1]]), [1]]))
            normalsbase[:, ii, jj] = ca.full(ggt(res[2]) @ np.concatenate([nT(toolvec, [res[0], res[1]]), [0]]))

            if not triplet_guess:
                guess[:] = triplets[:, ii, jj]

    return xyzbase, normalsbase, triplets