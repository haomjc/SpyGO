import casadi as ca
import screwCalculus as sc
import numpy as np
from hypoid.main.utils import *
from hypoid.main.kinematics import *
from hypoid.main.data_structs import DesignData, MachineField, CutterField
from copy import deepcopy
from hypoid.main.ease_off import plot_ease_off
SPARSITY_FLAG = 'dense' # for the 'sparse' flag we need to rebuild the identification problem sparsifying the machine-tool settings for each grid point. Not worth
PI = np.pi

def split_machine_tool_index(x_index): 
    """
    Split the selction index array for machine-tool settings into machine slection settings, tool selection settings as well as
    the diff set (indices of unchanged machine-tool parameters)
    """
    index_machine = x_index[x_index < 72]
    index_tool = x_index[x_index >= 72] - 72
    parameters_machine = np.setdiff1d(np.arange(0, 72), index_machine)
    parameters_tool = np.setdiff1d(np.arange(0, 10), index_tool)
    return index_machine, index_tool, parameters_machine, parameters_tool

def extract_bounds_from_data(design_data:DesignData, x_index, member, flank):
    mach_UB, mach_LB, cut_UB, cut_LB = identification_bounds(design_data, member, flank)
    lb = data_from_settings_index(mach_LB, cut_LB, x_index)
    ub = data_from_settings_index(mach_UB, cut_UB, x_index)
    return lb, ub

def data_from_settings_index(machine_settings:MachineField, tool_settings:CutterField, x_index):

    if isinstance(x_index, list) or isinstance(x_index, tuple):
        x_index = np.array(x_index)
    
    # split machine and tool indices
    machine_index = x_index[x_index < 72]
    tool_index = x_index[x_index >= 72] - 72


    selected_machine_settings = machine_settings.extract_settings().reshape(-1, 1, order = 'F')[machine_index]
    selected_tool_settings = np.array(tool_settings.extract_settings(bounds_data=True))[tool_index]
    #(POINTRADIUS, RHO, EDGERADIUS, topremRADIUS, flankremRADIUS, BLADEANGLE, flankremDEPTH, topremDEPTH, topremANGLE, flankremANGLE)

    # concatenate into a single array
    selected_settings = np.vstack((selected_machine_settings, selected_tool_settings.reshape(-1, 1, order = 'F')))
    return selected_settings
 
def identification_bounds(designData:DesignData, member, flank):
    
    scaling = designData.system_data.NORMALMODULE

    if scaling is None: # we may miss the normal module if the data is loaded from T3D report
        scaling = normal_module_from_data(designData)

    machine_field = designData.get_machine_field(member, flank)

    machine_field_LB = deepcopy(machine_field)
    machine_field_UB = deepcopy(machine_field)

    # bounds for radial joint
    machine_field_LB.RADIALSETTING = machine_field.RADIALSETTING * 0.7
    machine_field_LB.R1 = -10
    machine_field_LB.R2 = -50
    machine_field_LB.R3 = -1000
    machine_field_LB.R4 = -5000
    machine_field_LB.R5 = -100000
    machine_field_LB.R6 = -10000000

    machine_field_UB.RADIALSETTING = machine_field.RADIALSETTING * 1.3
    machine_field_UB.R1 = 10
    machine_field_UB.R2 = 50
    machine_field_UB.R3 = 1000
    machine_field_UB.R4 = 5000
    machine_field_UB.R5 = 100000
    machine_field_UB.R6 = 10000000
    
    # bounds for tilt joint
    machine_field_LB.TILTANGLE = machine_field.TILTANGLE - 8
    machine_field_LB.TLT1 = -0.02
    machine_field_LB.TLT2 = -1
    machine_field_LB.TLT3 = -5
    machine_field_LB.TLT4 = -100
    machine_field_LB.TLT5 = -10000
    machine_field_LB.TLT6 = -500000

    machine_field_UB.TILTANGLE = machine_field.TILTANGLE + 8
    machine_field_UB.TLT1 = 0.02
    machine_field_UB.TLT2 = 1
    machine_field_UB.TLT3 = 5
    machine_field_UB.TLT4 = 100
    machine_field_UB.TLT5 = 10000
    machine_field_UB.TLT6 = 500000

    # bounds for swivel joint
    machine_field_LB.SWIVELANGLE = -70
    machine_field_LB.SW1 = -2
    machine_field_LB.SW2 = -20
    machine_field_LB.SW3 = -100
    machine_field_LB.SW4 = -20000
    machine_field_LB.SW5 = -400000
    machine_field_LB.SW6 = -8000000

    machine_field_UB.SWIVELANGLE = +70
    machine_field_UB.SW1 = 2
    machine_field_UB.SW2 = 20
    machine_field_UB.SW3 = 100
    machine_field_UB.SW4 = 20000
    machine_field_UB.SW5 = 400000
    machine_field_UB.SW6 = 8000000

    # bounds for vertical joint (OFFSET)
    machine_field_LB.BLANKOFFSET = machine_field.BLANKOFFSET - 5 
    machine_field_LB.V1 = -1
    machine_field_LB.V2 = -10
    machine_field_LB.V3 = -100
    machine_field_LB.V4 = -5000
    machine_field_LB.V5 = -100000
    machine_field_LB.V6 = -1000000

    machine_field_UB.BLANKOFFSET = machine_field.BLANKOFFSET + 5 
    machine_field_UB.V1 = 1
    machine_field_UB.V2 = 10
    machine_field_UB.V3 = 100
    machine_field_UB.V4 = 5000
    machine_field_UB.V5 = 100000
    machine_field_UB.V6 = 1000000

    # helical joint
    machine_field_LB.SLIDINGBASE = machine_field.SLIDINGBASE - 5
    machine_field_LB.H1 = -1
    machine_field_LB.H2 = -10
    machine_field_LB.H3 = -100
    machine_field_LB.H4 = -5000
    machine_field_LB.H5 = -1000000000
    machine_field_LB.H6 = -10000000000
    
    machine_field_UB.SLIDINGBASE = machine_field.SLIDINGBASE + 5
    machine_field_UB.H1 = +1
    machine_field_UB.H2 = +10
    machine_field_UB.H3 = +100
    machine_field_UB.H4 = +5000
    machine_field_UB.H5 = +1000000000
    machine_field_UB.H6 = +10000000000

    # cradle angle
    machine_field_LB.CRADLEANGLE = max(machine_field.CRADLEANGLE - 60, 2)
    machine_field_UB.CRADLEANGLE = min(machine_field.CRADLEANGLE + 60, 120)

    # blank roll joint
    machine_field_LB.RATIOROLL = machine_field.RATIOROLL * 0.7
    machine_field_LB.C2        = -0.1
    machine_field_LB.D6        = -5
    machine_field_LB.E24       = -50
    machine_field_LB.F120      = -3000
    machine_field_LB.G720      = -10000000
    machine_field_LB.H5040     = -100000000
    
    machine_field_UB.RATIOROLL = machine_field.RATIOROLL * 1.4
    machine_field_UB.C2        = +0.1
    machine_field_UB.D6        = +5
    machine_field_UB.E24       = +50
    machine_field_UB.F120      = +3000
    machine_field_UB.G720      = +10000000
    machine_field_UB.H5040     = +100000000

    # machine center to back
    machine_field_LB.MACHCTRBACK = machine_field.MACHCTRBACK - 5
    machine_field_UB.MACHCTRBACK = machine_field.MACHCTRBACK + 5

    # machine root angle
    machine_field_LB.ROOTANGLE = machine_field.ROOTANGLE - 1
    machine_field_UB.ROOTANGLE = machine_field.ROOTANGLE + 1

    cutter_field = designData.get_tool_field(member, flank)

    cutter_field_LB = deepcopy(cutter_field)
    cutter_field_UB = deepcopy(cutter_field)

    cutter_field_LB.POINTRADIUS = cutter_field.POINTRADIUS * 0.66
    cutter_field_UB.POINTRADIUS = cutter_field.POINTRADIUS * 1.33

    cutter_field_LB.RHO = 5*scaling
    cutter_field_UB.RHO = 1000*scaling

    cutter_field_LB.BLADEANGLE = 5
    cutter_field_UB.BLADEANGLE = 30

    cutter_field_LB.EDGERADIUS = 0.2
    cutter_field_UB.EDGERADIUS = 1.5

    cutter_field_LB.topremRADIUS = 5*scaling
    cutter_field_UB.topremRADIUS = 100*scaling

    cutter_field_LB.topremDEPTH = 1.5
    cutter_field_UB.topremDEPTH = scaling

    cutter_field_LB.topremANGLE = -5
    cutter_field_UB.topremANGLE = +5

    cutter_field_LB.flankremRADIUS = 5*scaling
    cutter_field_UB.flankremRADIUS = 100*scaling

    cutter_field_LB.flankremDEPTH = scaling
    cutter_field_UB.flankremDEPTH = 20

    cutter_field_LB.flankremANGLE = -5
    cutter_field_UB.flankremANGLE = +5


    return machine_field_UB, machine_field_LB, cutter_field_UB, cutter_field_LB

def machine_identification_problem(triplets, x_index, lb, ub, lb_scaling, ub_scaling, 
                                    designData: DesignData, member, flank, root_constraint = None,
                                    bound_points_tol = 1
                                    ):
    """
    """
    HAND = designData.system_data.hand
    machine_settings = designData.extract_machine_settings_matrix(member, flank)
    tool_settings = np.array(designData.extract_tool_settings(member, flank))

    if isinstance(x_index, list):
        x_index = np.array(x_index)
    
    # we need to differentiate the machine settings used as decision variables and the ones used as fixed parameters
    index_machine, index_tool, parameters_machine, parameters_tool = split_machine_tool_index(x_index)

    if triplets.ndim == 3:
        triplets = triplets.squeeze()

    num_points = max(triplets.shape) # this may be misleading if we have less than 3 points

    # check if toprem is enabled. I need to chek if 75 is included in the x_index
    toprem_check = False
    if 76 in x_index:
        toprem_check = True
    
    # check if flankrem is enabled. I need to check if 76 is included in the x_index
    flankrem_check = False
    if 79 in x_index:
        flankrem_check = True
        
    p_tool, n_tool, _ = casadi_tool_fun(flank, toprem = toprem_check, flankrem = flankrem_check)
    ggt, Vgt, Vgt_spatial = casadi_machine_kinematics(member, HAND)

    # define casadi symbols just to vreate the kinematic functions
    csi = ca.MX.sym('csi', 1)
    theta = ca.MX.sym('theta', 1)
    phi = ca.MX.sym('phi', 1)

    pg_expr = ggt(machine_settings, phi) @ ca.vertcat(p_tool(tool_settings, ca.vertcat(csi, theta)), 1)
    ng_expr = ggt(machine_settings, phi) @ ca.vertcat(n_tool(tool_settings, ca.vertcat(csi, theta)), 0)

    pg_fun = ca.Function('pg_fun', [ca.vertcat(csi, theta, phi)], [pg_expr[0:3]])
    ng_fun = ca.Function('ng_fun', [ca.vertcat(csi, theta, phi)], [ng_expr[0:3]])

    # numerical values extraction and scaling bounds definition
    machine_settings_flatten = machine_settings.flatten(order = 'F')
    x_machine_values = machine_settings_flatten[index_machine]
    p_machine_values = machine_settings_flatten[parameters_machine]
    x_tool_values = tool_settings[index_tool]
    p_tool_values = tool_settings[parameters_tool]
    csi_num = triplets[0,:].reshape(1, -1, order = 'F')
    theta_num = triplets[1,:].reshape(1, -1, order = 'F')
    phi_num = triplets[2,:].reshape(1, -1, order = 'F')
    h_num = np.ones((1, num_points)) * 0.0

    h_ub = h_num + 0.05
    h_lb = h_num - 0.05

    num_points_flank = num_points
    if root_constraint is not None:
        triplets_root = root_constraint['triplets']
        num_points_root = max(triplets_root.shape)
        csi_num = np.hstack([csi_num, triplets_root[0,:].reshape(1,-1, order = 'F')])
        theta_num = np.hstack([theta_num, triplets_root[1,:].reshape(1,-1, order = 'F')])
        phi_num = np.hstack([phi_num, triplets_root[2,:].reshape(1,-1, order = 'F')])
        num_points = num_points + num_points_root
    else:
        num_points_root = 0

    # numerical values for the scaling bounds
    csi_lb = np.maximum(csi_num - 0.1*csi_num.max(), 0)
    csi_ub = csi_num + 0.1*csi_num.max()
    theta_lb = theta_num - np.pi/10
    theta_ub = theta_num + np.pi/10
    phi_lb = phi_num - np.pi/10
    phi_ub = phi_num + np.pi/10

    pg_num = pg_fun(np.r_[csi_num, theta_num, phi_num]).full()
    pg_lb = pg_num - 0.1
    pg_ub = pg_num + 0.1
    pg_lb[:, -num_points_root:] = pg_num[:, -num_points_root:] - 0.1
    pg_ub[:, -num_points_root:] = pg_num[:, -num_points_root:] + 0.1

    ng_num = ng_fun(np.r_[csi_num, theta_num, phi_num]).full()
    ng_lb = ng_num - 0.1
    ng_ub = ng_num + 0.1
    ng_lb[:, -num_points_root:] = ng_num[:, -num_points_root:] - 0.05
    ng_ub[:, -num_points_root:] = ng_num[:, -num_points_root:] + 0.05

    Vgt_num = Vgt_spatial(machine_settings, phi_num).full().reshape((4, 4, num_points), order = 'F')
    Vgt_scaling = np.max(Vgt_num, axis = 2)
    Vgt_num = Vgt_num[0:3, :, :].reshape(12, num_points, order = 'F')
    Vgt_lb = Vgt_num - 8
    Vgt_ub = Vgt_num + 8

    # Define symbolic variables
    # scaled variables
    csi_n = ca.SX.sym('csi_n')
    theta_n = ca.SX.sym('theta_n')
    phi_n = ca.SX.sym('phi_n')
    h_n = ca.SX.sym('h_n')
    p_g_n = ca.SX.sym('p_g_n', 3)
    n_g_n = ca.SX.sym('n_g_n', 3)
    Vgt_n = ca.SX.sym('Vgt_n', 12, 1)

    # symbolic variables for the tool settings
    tool_settings_sym = ca.SX.sym('Ts', 10, 1)
    machine_settings_sym = ca.SX.sym('Ms', 9*8, 1)
    xM_n = machine_settings_sym[index_machine]
    xT_n = tool_settings_sym[index_tool]

    # symbolic parameters (symbolic upper and lower bounds for parametric scaling)
    csi_ub_sym = ca.SX.sym('csi_ub')
    csi_lb_sym = ca.SX.sym('csi_lb')
    theta_ub_sym = ca.SX.sym('theta_ub')
    theta_lb_sym = ca.SX.sym('theta_lb')
    phi_ub_sym = ca.SX.sym('phi_ub')
    phi_lb_sym = ca.SX.sym('phi_lb')
    h_ub_sym = ca.SX.sym('h_ub')
    h_lb_sym = ca.SX.sym('h_lb')
    xM_ub_sym = ca.SX.sym('xM_ub', len(index_machine))
    xM_lb_sym = ca.SX.sym('xM_lb', len(index_machine))
    xT_ub_sym = ca.SX.sym('xT_ub', len(index_tool))
    xT_lb_sym = ca.SX.sym('xT_lb', len(index_tool))
    pg_ub_sym = ca.SX.sym('pg_ub', 3)
    pg_lb_sym = ca.SX.sym('pg_lb', 3)
    ng_ub_sym = ca.SX.sym('ng_ub', 3)
    ng_lb_sym = ca.SX.sym('ng_lb', 3)
    Vgt_ub_sym = ca.SX.sym('Vgt_ub', 12)
    Vgt_lb_sym = ca.SX.sym('Vgt_lb', 12)
    
    p_target_sym = ca.SX.sym('p_target', 3)

    # unscale the variables
    csi_sym = csi_lb_sym + (csi_ub_sym - csi_lb_sym) * csi_n
    theta_sym = theta_lb_sym + (theta_ub_sym - theta_lb_sym) * theta_n
    phi_sym = phi_lb_sym + (phi_ub_sym - phi_lb_sym) * phi_n
    h_sym = h_lb_sym + (h_ub_sym - h_lb_sym) * h_n
    xM_sym = xM_lb_sym + (xM_ub_sym - xM_lb_sym) * xM_n
    xT_sym = xT_lb_sym + (xT_ub_sym - xT_lb_sym) * xT_n
    pg_sym = pg_lb_sym + (pg_ub_sym - pg_lb_sym) * p_g_n
    ng_sym = ng_lb_sym + (ng_ub_sym - ng_lb_sym) * n_g_n
    Vgt_sym = Vgt_lb_sym + (Vgt_ub_sym - Vgt_lb_sym) * Vgt_n

    # use non scaled variables to compute the kinematics
    machine_settings_sym[index_machine] = xM_sym
    tool_settings_sym[index_tool] = xT_sym

    pg_expr = ggt(machine_settings_sym.reshape((9, 8)), phi_sym) @ ca.vertcat(p_tool(tool_settings_sym, ca.vertcat(csi_sym, theta_sym)), 1)
    ng_expr = ggt(machine_settings_sym.reshape((9, 8)), phi_sym) @ ca.vertcat(n_tool(tool_settings_sym, ca.vertcat(csi_sym, theta_sym)), 0)

    Vg_expr = Vgt_spatial(machine_settings_sym.reshape((9, 8)), phi_sym)
    Vg_expr = Vg_expr[0:3, :]

    X_scale = np.abs(pg_num[0,:]).max()
    Y_scale = np.abs(pg_num[1,:]).max()
    Z_scale = np.abs(pg_num[2,:]).max()

    eq_meshing_scale = ca.fabs(np.array([0.8, 0.8, 0.8, 0])@Vgt_scaling@np.array([X_scale, Y_scale, Z_scale, 1]))

    # constraints for the generic point
    E_eq = (pg_sym - p_target_sym + h_sym * ng_sym)/0.5 # ease-off equation
    f_eq = ca.vertcat(ng_sym, 0).T @ ca.vertcat(Vgt_sym.reshape((3, 4)), ca.horzcat(0,0,0,0)) @ ca.vertcat(pg_sym, 1)/eq_meshing_scale # meshing equation
    point_eq = (pg_sym - pg_expr[0:3])/0.5  # congruence equation
    normal_eq = (ng_sym - ng_expr[0:3])  # normals congruence
    twist_eq = (Vgt_sym - Vg_expr.reshape((-1, 1)))/(ca.fabs(Vgt_scaling[0:3,:].flatten(order = 'F')) + 0.1) # twist equation

    # constraints for the i-th point of the rootline
    R_scale = ca.sqrt(X_scale**2 + Y_scale**2)
    R_eq = (pg_sym - p_target_sym).T @ ng_sym
    z_eq = (pg_sym[2] - p_target_sym[2])
    circ_eq = (ca.atan2(pg_sym[1], pg_sym[0]) - ca.atan2(p_target_sym[1], p_target_sym[0]))*ca.sqrt(pg_sym[0]**2 + pg_sym[1]**2)
    root_constraints_eq = ca.vertcat(z_eq, circ_eq, R_eq)

    # reassign scaled settings as variable
    machine_settings_sym[index_machine] = xM_n
    tool_settings_sym[index_tool] = xT_n

    # build the constraint functions
    W_p = ca.vertcat(
        csi_n, theta_n, phi_n, h_n, p_g_n, n_g_n, Vgt_n,
        csi_lb_sym, csi_ub_sym, theta_lb_sym, theta_ub_sym, phi_lb_sym, phi_ub_sym, h_lb_sym, h_ub_sym,
        pg_lb_sym, pg_ub_sym, ng_lb_sym, ng_ub_sym, Vgt_lb_sym, Vgt_ub_sym, p_target_sym
    )

    W_x = ca.vertcat(
        machine_settings_sym, tool_settings_sym, xM_lb_sym, xT_lb_sym, xM_ub_sym, xT_ub_sym
    )

    # flank ease off constraints
    con_fun_1 = ca.Function('con1', [W_p, W_x],
                                [ca.vertcat(E_eq, f_eq, point_eq, normal_eq, twist_eq)],
                                {'cse': True})
    con_map_1 = con_fun_1.map(num_points_flank)

    W_p = ca.vertcat(
        csi_n, theta_n, phi_n, p_g_n, n_g_n, Vgt_n,
        csi_lb_sym, csi_ub_sym, theta_lb_sym, theta_ub_sym, phi_lb_sym, phi_ub_sym,
        pg_lb_sym, pg_ub_sym, ng_lb_sym, ng_ub_sym, Vgt_lb_sym, Vgt_ub_sym, p_target_sym
    )

    # rootline constraints
    con_fun_2 = ca.Function('con2', [W_p, W_x],
                                [ca.vertcat(f_eq, point_eq, normal_eq, twist_eq, root_constraints_eq)],
                                {'cse': True})

    con_map_2 = con_fun_2.map(num_points_root)

    num_constraints_root = ca.vertcat(f_eq, point_eq, normal_eq, twist_eq, root_constraints_eq).shape[0]
    
    ## Mapped problem building

    casadi_sym = ca.SX.sym

    # variables initialization
    csi_var = casadi_sym('csi_var', 1, num_points)
    theta_var = casadi_sym('theta_var', 1, num_points)
    phi_var = casadi_sym('phi_var', 1, num_points)
    h_var = casadi_sym('h_var', 1, num_points_flank)
    pg_var = casadi_sym('pg_var', 3, num_points)
    ng_var = casadi_sym('ng_var', 3, num_points)
    Vgt_var = casadi_sym('Vgt_var', 12, num_points)
    p_target = casadi_sym('p_target', 3, num_points)

    # collect variables associated with the grid points
    W_points_con1 = [csi_var, theta_var, phi_var, h_var, pg_var, ng_var, Vgt_var,
            csi_lb, csi_ub, theta_lb, theta_ub, phi_lb, phi_ub, h_lb, h_ub,
            pg_lb, pg_ub, ng_lb, ng_ub, Vgt_lb, Vgt_ub, p_target]
    
    W_points_1 = [] # collect points on the active flank for constraint 1 (ease-off)
    for item in W_points_con1:
        W_points_1.append(item[:, 0:num_points_flank])
    
    # group variables used in the rooline shift constraints
    W_points_con2 = [csi_var, theta_var, phi_var, pg_var, ng_var, Vgt_var,
        csi_lb, csi_ub, theta_lb, theta_ub, phi_lb, phi_ub,
        pg_lb, pg_ub, ng_lb, ng_ub, Vgt_lb, Vgt_ub, p_target]
    
    W_points_2 = [] # collect points on the rootline for rootlineshift constraint
    for item in W_points_con2:
        W_points_2.append(item[:, -(num_points_root):])


    xM_var = casadi_sym('xM', max(index_machine.shape))
    xT_var = casadi_sym('xT', max(index_tool.shape))
    pM_var = casadi_sym('pM', max(parameters_machine.shape))
    pT_var = casadi_sym('pT', max(parameters_tool.shape))
    machine_settings_var = ca.SX(9*8, 1)
    machine_settings_var[index_machine] = xM_var
    machine_settings_var[parameters_machine] = pM_var
    tool_settings_var = ca.SX(10, 1)
    tool_settings_var[index_tool] = xT_var
    tool_settings_var[parameters_tool] = pT_var

    # collect variables associated with machine-tool settings
    W_settings = [machine_settings_var, tool_settings_var, lb_scaling, ub_scaling]

    # objective function: in the future we may also want to minimize some machine settings excursion w.r.t. to a desired configuration
    h = h_var*(h_ub - h_lb) + h_lb
    obj_expr = h @ h.T

    assert ca.vertcat(*W_points_1).shape[1] == num_points_flank
    constraints_expr_1 = con_map_1(ca.vertcat(*W_points_1), ca.vertcat(*W_settings))
    constraints_expr_2 = con_map_2(ca.vertcat(*W_points_2), ca.vertcat(*W_settings))

    constraint_expr = ca.reshape(constraints_expr_1, -1, 1)
    if root_constraint is not None:
        constraint_expr = ca.vertcat(constraint_expr, ca.reshape(constraints_expr_2, -1 ,1))

    X = reorder_identification_variables(ca.vertcat(xM_var, xT_var), csi_var, theta_var, phi_var, h_var, pg_var, ng_var, Vgt_var, SPARSITY_FLAG)

    problem = {'x': X, 
               'p': ca.vertcat(pM_var, pT_var, ca.reshape(p_target,-1,1)),
               'f': obj_expr,
               'g': constraint_expr}
    
    len_g = max(constraint_expr.shape)

    # casadi and IPOPT options
    options = IPOPT_global_options()

    # solver
    solver = ca.nlpsol('solver', 'ipopt', problem, options)

    # parameters collection
    P = ca.vertcat(ca.reshape(pM_var, -1, 1),
                   ca.reshape(pT_var, -1, 1),
                   ca.reshape(p_target, -1, 1))
    
    objective_function = ca.Function('obj', [X, P], [obj_expr, ca.gradient(obj_expr, X)])
    constraint_function = ca.Function('con', [X, P], [constraint_expr, ca.jacobian(constraint_expr, X)])

    # inspect jacobiab structure
    # import matplotlib.pyplot as plt
    # J = ca.DM(ca.jacobian(constraint_expr, X))
    # plt.spy(J, markersize=10)
    # plt.show()

    # scaling bounds for all variables (those will just scale the properly the problem and are not the real bounds)
    lb_scaling = reorder_identification_variables(lb_scaling, csi_lb, theta_lb, phi_lb, h_lb, pg_lb, ng_lb, Vgt_lb, SPARSITY_FLAG).full().reshape(-1,1, order = 'F')
    ub_scaling = reorder_identification_variables(ub_scaling, csi_ub, theta_ub, phi_ub, h_ub, pg_ub, ng_ub, Vgt_ub, SPARSITY_FLAG).full().reshape(-1,1, order = 'F')

    # actual bounds calculation
    csi_lb = np.maximum(csi_num - csi_num.max(), 0.000+0.05)
    csi_ub = csi_num + csi_num.max()
    csi_lb[0, -num_points_root:] = 0
    csi_ub[0, -num_points_root:] = 0.001
    theta_lb = theta_num - PI/2
    theta_ub = theta_num + PI/2
    phi_lb = phi_num - PI/2
    phi_ub = phi_num + PI/2
    h_lb = h_lb*0 - 10
    h_ub = h_lb*0 + 10

    # twist bounds
    Vgt_lb = Vgt_num - 50
    Vgt_ub = Vgt_num + 50

    # point bounds
    pg_lb = pg_num - bound_points_tol*ca.vertcat(1,1,1)
    pg_ub = pg_num + bound_points_tol*ca.vertcat(1,1,1)
    pg_lb[:, -num_points_root:] = pg_num[:, -num_points_root:] - bound_points_tol*ca.vertcat(1,1,1)
    pg_ub[:, -num_points_root:] = pg_num[:, -num_points_root:] + bound_points_tol*ca.vertcat(1,1,1)

    # normals bounds
    ng_lb = ng_num - 0.8*ca.vertcat(1,1,1)
    ng_ub = ng_num + 0.8*ca.vertcat(1,1,1)

    # actual bounds order restructuring
    lb = reorder_identification_variables(lb, csi_lb, theta_lb, phi_lb, h_lb, pg_lb, ng_lb, Vgt_lb, SPARSITY_FLAG).full().reshape(-1,1, order = 'F')
    ub = reorder_identification_variables(ub, csi_ub, theta_ub, phi_ub, h_ub, pg_ub, ng_ub, Vgt_ub, SPARSITY_FLAG).full().reshape(-1,1, order = 'F')
    values = reorder_identification_variables(ca.vertcat(x_machine_values, x_tool_values), csi_num, theta_num, phi_num, h_num, pg_num, ng_num, Vgt_num, SPARSITY_FLAG).full().reshape(-1, 1, order = 'F') 

    constraints_bounds = np.zeros(len_g)
    lbg = constraints_bounds.copy()

    # slice indices (Python stop is exclusive, so subtract an extra step)
    lbg[-1 : -( (num_points_root-1)*num_constraints_root + 2 ) : -num_constraints_root] = -1
    lbg[-2 : -( (num_points_root-1)*num_constraints_root + 3 ) : -num_constraints_root] = -1  # circular shift bound
    lbg[-3 : -( (num_points_root-1)*num_constraints_root + 4 ) : -num_constraints_root] = -1  # z bound

    ubg = lbg.copy()
    ubg[-1 : -( (num_points_root-1)*num_constraints_root + 2 ) : -num_constraints_root] = 1
    ubg[-2 : -( (num_points_root-1)*num_constraints_root + 3 ) : -num_constraints_root] = 1  # circular shift bound
    ubg[-3 : -( (num_points_root-1)*num_constraints_root + 4 ) : -num_constraints_root] = 1  # z bound

    # managing bound locked variables (upper bound == lower bound)
    index_mask = ub - lb == 0
    x0 = (values - lb_scaling)/(ub_scaling - lb_scaling)
    lbx = (lb - lb_scaling) / (ub_scaling - lb_scaling)
    ubx = (ub - lb_scaling) / (ub_scaling - lb_scaling)
    x0[index_mask] = lb_scaling[index_mask]
    lbx[index_mask] = lb_scaling[index_mask]
    ubx[index_mask] = lb_scaling[index_mask]

    # problem settings
    settings = {
        'x0_not_normalized': values,
        'x0' : x0,
        'p'   : ca.vertcat(p_machine_values, p_tool_values), # the only parameter missing are the target points to provide at evaluation time
        'lb' : lb,
        'ub' : ub,
        'lb_scaling' : lb_scaling,
        'ub_scaling' : ub_scaling,
        'lbx' : lbx,
        'ubx' : ubx,
        'lbg' : lbg,
        'ubg' : ubg,
        'x_index': x_index,
        'triplets': np.concatenate((triplets, triplets_root), 1),
        'ng': len_g,
        'ng_root' : num_constraints_root,
        'num_points_root'     : num_points_root,
        'num_points_flank'     : num_points_flank,
        'nx'    : 22,
        'x_sym': X,
        'p_sym' :P,
        'obj_fun': objective_function,
        'con_fun': constraint_function
    }

    return solver, settings

def evaluate_identification_problem(solver:ca.nlpsol, settings, target_points:np.array, root_tol = 0.025, circ_tol = 0.15, warm_x0 = None):
    """
    Evaluates the pre-built identification problem
    """

    if target_points.shape[0] == 4:
        target_points = target_points[0:3, :]

    # extract solver settings and parameters
    x0  = settings['x0']
    lbx = settings['lbx']
    ubx = settings['ubx']
    lbN = settings['lb_scaling'].squeeze()
    ubN = settings['ub_scaling'].squeeze()
    lbg = settings['lbg']
    ubg = settings['ubg']

    ng_root = settings['ng_root']
    num_points_root = settings['num_points_root']
    num_points_given = target_points.shape[1]
    num_points_flank = settings['num_points_flank']
    num_index = max(settings['x_index'].shape)
    nx = settings['nx']

    # scale the rootline shift and circular shoft constraints to the actual desired value

    # If toor_tol has two elements, we have asymmetric rootline constraint bounds
    if isinstance(root_tol, list):
        # rootline shift
        lbg[-1 : -( (num_points_root-1)*ng_root + 2 ) : -ng_root] = -root_tol[0]
        ubg[-1 : -( (num_points_root-1)*ng_root + 2 ) : -ng_root] = root_tol[1]

        # axial shift of points (we do not want them to shift too much)
        lbg[-3 : -( (num_points_root-1)*ng_root + 4 ) : -ng_root] = -root_tol[0]
        ubg[-3 : -( (num_points_root-1)*ng_root + 4 ) : -ng_root] = root_tol[1]

    else:
        # rootline shift
        lbg[-1 : -( (num_points_root-1)*ng_root + 2 ) : -ng_root] = -root_tol
        ubg[-1 : -( (num_points_root-1)*ng_root + 2 ) : -ng_root] = root_tol

        # axial shift of points (we do not want them to shift too much)
        lbg[-3 : -( (num_points_root-1)*ng_root + 4 ) : -ng_root] = -root_tol * 0.1
        ubg[-3 : -( (num_points_root-1)*ng_root + 4 ) : -ng_root] = root_tol * 0.1

    # circular shift
    lbg[-2 : -( (num_points_root-1)*ng_root + 3 ) : -ng_root] = -circ_tol
    ubg[-2 : -( (num_points_root-1)*ng_root + 3 ) : -ng_root] = circ_tol

    p = np.concatenate((settings['p'].full().squeeze(), target_points.flatten(order = 'F')), 0)

    if warm_x0 is not None:
        x0 = warm_x0
    
    result = solver(x0 = x0, p = p, lbx = lbx, ubx = ubx, lbg = lbg, ubg = ubg)
    stats = solver.stats()
    return_status = stats['return_status']

    x_n = result['x'].full().squeeze()
    # active = (np.isclose(x, lbx, atol=1e-5) | np.isclose(x, ubx, atol=1e-5))
    # print("Active bounds at solution:", np.where(active)[0])

    # post-processing
    x = (ubN - lbN)*x_n + lbN

    new_settings = x[0:num_index]
    x = x[num_index:]
    states = x[0:nx*(num_points_flank)].reshape(nx, num_points_flank, order = 'F')
    triplets = states[0:3,:]
    residuals = states[3,:]
    points = states[4:7,:]
    normals = states[7:10,:]
    # points = points[:, :-num_points_root]
    # Z = points[2,:].reshape(11,22,order = 'F')
    # R = np.sqrt(points[0,:]**2 + points[1,:]**2).reshape(11,22,order = 'F')
    # res = residuals.reshape(11,22,order = 'F')
    # U, V = np.meshgrid(np.linspace(-1, 1, 22), np.linspace(-1, 1, 11))
    # plot_ease_off(res*1000, Z, R, aspect_ratio=[1,1,np.max(np.abs(res))*1000/5], labels=['z (mm)', 'R (mm)', 'E ($\mu$ m)'])
    # print(np.abs(res).max())

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[0,:], points[1,:], points[2,:])
    # ax.set_aspect('equal', adjustable='box', anchor='C')
    # plt.show()
    return new_settings, residuals
 
def reorder_identification_variables(x, csi_var, theta_var, phi_var, h_var,
                                         p_g_var, n_g_var, V_g_var, sparsity_flag):
    num_points_flank = max(h_var.shape)  # length in MATLAB → number of rows

    if sparsity_flag.lower() == 'sparse':
        # Sparse machine-tool settings
        c, r = x.shape

        # MATLAB's csi_var(:)' → row vector
        csi_var   = ca.reshape(csi_var,   1, -1)
        theta_var = ca.reshape(theta_var, 1, -1)
        phi_var   = ca.reshape(phi_var,   1, -1)
        h_var     = ca.reshape(h_var,     1, -1)

        xx = ca.vertcat(
            x[:, :num_points_flank],
            csi_var[:, :num_points_flank],
            theta_var[:, :num_points_flank],
            phi_var[:, :num_points_flank],
            h_var,
            p_g_var[:, :num_points_flank],
            n_g_var[:, :num_points_flank],
            V_g_var[:, :num_points_flank]
        )

        xx2 = ca.vertcat(
            x[:, num_points_flank:r-1],
            csi_var[:, num_points_flank:],
            theta_var[:, num_points_flank:],
            phi_var[:, num_points_flank:],
            p_g_var[:, num_points_flank:],
            n_g_var[:, num_points_flank:],
            V_g_var[:, num_points_flank:]
        )

        # Flatten: MATLAB xx(:) → reshape to column
        x = ca.vertcat(
            ca.reshape(xx,  -1, 1),
            ca.reshape(xx2, -1, 1),
            x[:, r-1]  # last column
        )

    elif sparsity_flag.lower() == 'dense':
        # Dense machine-tool settings
        csi_var   = ca.reshape(csi_var,   1, -1)
        theta_var = ca.reshape(theta_var, 1, -1)
        phi_var   = ca.reshape(phi_var,   1, -1)
        h_var     = ca.reshape(h_var,     1, -1)

        xx = ca.vertcat(
            csi_var[:, :num_points_flank],
            theta_var[:, :num_points_flank],
            phi_var[:, :num_points_flank],
            h_var,
            p_g_var[:, :num_points_flank],
            n_g_var[:, :num_points_flank],
            V_g_var[:, :num_points_flank]
        )

        xx2 = ca.vertcat(
            csi_var[:, num_points_flank:],
            theta_var[:, num_points_flank:],
            phi_var[:, num_points_flank:],
            p_g_var[:, num_points_flank:],
            n_g_var[:, num_points_flank:],
            V_g_var[:, num_points_flank:]
        )

        x = ca.vertcat(
            ca.reshape(x,   -1, 1),
            ca.reshape(xx,  -1, 1),
            ca.reshape(xx2, -1, 1)
        )

    return x

def approxToolIdentification_casadi(data: DesignData, member, RHO = None):

    RHOinput = RHO
    system_data = data.system_data
    if member.lower() == 'pinion':
        common_data = data.pinion_common_data
        cutter_data_CNV = data.pinion_cutter_data.concave
        cutter_data_CVX = data.pinion_cutter_data.convex
    else:
        common_data = data.gear_common_data
        cutter_data_CNV = data.gear_cutter_data.concave
        cutter_data_CVX = data.gear_cutter_data.convex

    # extract data from struct
    nT         = common_data.NTEETH
    edgeRadius = cutter_data_CNV.EDGERADIUS
    rc0        = common_data.MEANCUTTERRAIDUS            # mean cutter radius
    alphanD    = system_data.NOMINALDRIVEPRESSUREANGLE   # drive side tooth pressure angle
    alphanC    = system_data.NOMINALCOASTPRESSUREANGLE   # coast side tooth pressure angle
    hamc       = common_data.MEANCHORDALADDENDUM         # tooth mean chordal addendum
    ham        = common_data.MEANADDENDUM                # tooth mean addendum
    t          = common_data.MEANNORMALCHORDALTHICKNESS  # tooth normal chordal thickness
    Rm         = common_data.MEANCONEDIST                # mean cone distance
    pitchapex  = common_data.PITCHAPEX
    delta      = common_data.PITCHANGLE
    betam      = common_data.SPIRALANGLE
    RHO        = cutter_data_CNV.RHO
    hand       = system_data.hand
    mmn        = system_data.NORMALMODULE

    if cutter_data_CNV.TYPE.lower() == 'straight':
        RHO = 6000

    if RHOinput is not None:
        RHO = RHOinput
        cutter_data_CNV.RHO = RHO

    if (hand.lower() == 'right' and member.lower() == 'gear') or (hand.lower() == 'left' and member.lower() == 'pinion'):
        Tng = lambda phi2: sc.TrotZ(phi2)@sc.TtZ(-pitchapex)@sc.TrotY(delta*pi/180)@sc.TtZ(Rm)@sc.TrotX(betam*pi/180)
        signThick = +1
        rotguess = pi/nT
    else:
        Tng = lambda phi2: sc.TrotZ(phi2)@sc.TtZ(-pitchapex)@sc.TrotY(delta*pi/180)@sc.TtZ(Rm)@sc.TrotX(-betam*pi/180)
        signThick = -1
        rotguess = -pi/nT

    if member.lower() == 'gear':
        pressAngCvx = alphanD # drive should always be convex for gear and concave for pinion, but it may happen to have inverse situations
        pressAngCnv = alphanC 
    else:
        pressAngCnv = alphanD
        pressAngCvx = alphanC

    machinePar = data.extract_machine_settings_matrix(member, 'concave')
    cMat, sMat = DesignData.manage_machine_settings(member, hand)
    machineParMatrix = cMat*sMat*machinePar

    # Kinematic computation
    ggt, Vgt, Vgt_spatial = machine_kinematics(machineParMatrix)

    triplet = initial_guess_from_data(data, member, 'convex')
    csiguessCVX = mmn*6
    thetaguessCVX = triplet[1]
    phiguessCVX = triplet[2]
    triplet = initial_guess_from_data(data, member, 'concave')
    csiguessCNV = mmn*6
    thetaguessCNV = triplet[1]
    phiguessCNV = triplet[2]

    x0 = [rc0-t, pressAngCvx*1.2, pressAngCnv, csiguessCNV, thetaguessCNV, csiguessCVX, thetaguessCVX, csiguessCNV, thetaguessCNV, csiguessCVX, thetaguessCVX,\
    rotguess, phiguessCNV, phiguessCVX, phiguessCNV, phiguessCVX, rc0]
    csiLow = mmn/50
    csiMax = mmn*10
    lb = [x0[0]-3*t, x0[1] - 10, x0[2] - 10, csiLow, x0[4] - pi, csiLow, x0[6] - pi, csiLow, x0[8] - pi, csiLow, x0[10] - pi, x0[11] - 3*pi/nT, x0[12] - pi, x0[13] - pi, x0[14] - pi, x0[15] - pi, x0[16]-rc0/4]
    ub = [x0[0]+3*t, x0[1] + 10, x0[2] + 10, x0[3] + csiMax, x0[4] + pi, x0[5] + csiMax , x0[6] + pi, x0[7] + csiMax, x0[8] + pi, x0[9] + csiMax, x0[10] + pi, x0[11] + 3*pi/nT, x0[12] + pi, x0[13] + pi, x0[14] + pi, x0[15] + pi, x0[16]+rc0/4]

    # simplified system variables
    x0 = [rc0-t, csiguessCNV, thetaguessCNV, csiguessCVX, thetaguessCVX, rotguess, phiguessCNV, phiguessCVX, rc0 + t]
    lb = [rc0-3*t, csiLow, thetaguessCNV-np.pi, csiLow, thetaguessCVX-np.pi, rotguess-np.pi/nT, phiguessCNV-np.pi/2, phiguessCVX-np.pi/2, rc0-2*t]
    ub = [rc0+3*t, csiguessCNV + csiMax, thetaguessCNV+np.pi, csiguessCVX + csiMax, thetaguessCVX+np.pi, rotguess+np.pi/nT, phiguessCNV+np.pi/2, phiguessCVX+np.pi/2, rc0+4*t]

    x0 = np.array(x0)
    lb = np.array(lb)
    ub = np.array(ub)

    x_unscaled = ca.SX.sym('x', 9, 1)

    x = x_unscaled * (ub - lb) + lb

    Rpcvx = x[0]
    csiI = x[1]  # concave
    thetaI = x[2]
    csiO = x[3]  # convex
    thetaO = x[4]
    phi2 = x[5]
    phiEnvI = x[6]
    phiEnvO = x[7]
    Rpcnv = x[8]

    alphacnv = pressAngCnv
    alphacvx = pressAngCvx
    toolO, toolNO = parametric_tool_casadi('convex', Rpcvx, RHO, alphacvx, edgeRadius, csiO, thetaO)
    toolI, toolNI = parametric_tool_casadi('concave', Rpcnv, RHO, alphacnv, edgeRadius, csiI, thetaI)

    T = sc.rigidInverse(Tng(phi2))
    pointO = T @ sc.TrotZ(signThick * 2 * np.pi / nT) @ ggt(phiEnvO) @ ca.vertcat(toolO, 1)
    pointI = T @ ggt(phiEnvI) @ ca.vertcat(toolI, 1)

    equations = ca.vertcat(
        (pointO[0:3] - np.array([-(hamc - ham)*0, +signThick * t / 2, 0])), # point cnv flank
        (pointI[0:3] - np.array([-(hamc - ham)*0, -signThick * t / 2, 0])), # point cvx flank
        ca.vertcat(toolNO, 0).T @ Vgt(phiEnvO) @ ca.vertcat(toolO, 1),      # eq meshing cnv
        ca.vertcat(toolNI, 0).T @ Vgt(phiEnvI) @ ca.vertcat(toolI, 1),      # eq meshing cvx
        Rpcnv - (2 * rc0 - Rpcvx)                                           # meancutter radius and point radii congruence
    )


    fun_test = ca.Function('ft', [x_unscaled], [equations])
    opts = IPOPT_global_options()
    opts['ipopt']['nlp_scaling_method'] = 'gradient-based'
    opts['ipopt']['max_iter'] = 4000
    problem = {'x': x_unscaled, 'g': equations, 'f': 0}
    solver = ca.nlpsol('S', 'ipopt', problem, opts)

    x0 = (x0 - lb)/(ub - lb)

    solution  = solver(x0 = x0, lbx = lb*0-0.0, ubx = lb*0+1.0, ubg = 0, lbg = 0)
    res = solution['x'].full().squeeze()
    res = res*(ub-lb) + lb

    edge_radius = (res[-1] - res[0])/2.5

    cutter_data_CNV.RHO = RHO
    cutter_data_CVX.RHO = RHO
    cutter_data_CVX.POINTRADIUS = res[0]
    cutter_data_CNV.POINTRADIUS = res[-1]
    cutter_data_CNV.EDGERADIUS = edge_radius
    cutter_data_CVX.EDGERADIUS = edge_radius

    triplet_concave = [res[1], res[2], res[6]]
    triplet_convex = [res[3], res[4], res[7]]

    return data, triplet_concave, triplet_convex
