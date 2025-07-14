import casadi as ca
import screwCalculus as sc
import numpy as np
from hypoid.main.utils import *
from hypoid.main.kinematics import *

def machine_identification_problem(triplets, x_index, lb, ub, lb_scaling, ub_scaling, 
                                    designData, member, flank, root_constraint = None,
                                    bound_points_tol = 2
                                    ):
    """
    """
    HAND = designData['SystemData']['HAND']
    machine_settings = assign_machine_par(designData, member, flank)
    tool_settings = assign_tool_par(designData, member, flank)

    if isinstance(x_index, list):
        x_index = np.array(x_index)
    
    # we need to differentiate the machine settings used as decision variables and the ones used as fixed parameters
    index_machine = x_index[x_index <= 72]
    index_tool = x_index[x_index > 72] - 72
    parameters_machine = np.setdiff1d(np.arange(1, 73), index_machine)
    parameters_tool = np.setdiff1d(np.arange(1, 11), index_tool)

    num_points = max(triplets.shape) # this may be misleading if we have less than 3 points

    # check if toprem is enabled. I need to chek if 76 is included in the x_index
    toprem_check = False
    if 76 in x_index:
        toprem_check = True
        

    # cehck if flankrem is enabled. I need to check if 77 is included in the x_index
    flankrem_check = False
    if 77 in x_index:
        flankflankrem_checkrem = True
        
    p_tool, n_tool, _ = casadi_tool_fun(flank, toprem = toprem_check, flankrem = flankrem_check)
    ggt, Vgt, Vgt_spatial = casadi_machine_kinematics(member, HAND)

    # define casadi symbols
    csi = ca.MX.sym('csi', 1)
    theta = ca.MX.sym('theta', 1)
    phi = ca.MX.sym('phi', 1)

    pg_expr = ggt(machine_settings, phi) @ ca.vertcat(p_tool(tool_settings, ca.vertcat(csi, theta)), 1)
    ng_expr = ggt(machine_settings, phi) @ ca.vertcat(n_tool(tool_settings, ca.vertcat(csi, theta)), 0)

    pg_fun = ca.Function('pg_fun', ca.vertcat(csi, theta, phi), [pg_expr[0:3]])
    ng_fun = ca.Function('ng_fun', ca.vertcat(csi, theta, phi), [ng_expr[0:3]])

    # numerical values extraction and scaling bounds definition
    machine_settings_flatten = machine_settings.flatten(order = 'F')
    x_machine_values = machine_settings_flatten[index_machine - 1]
    p_machine_values = machine_settings_flatten[parameters_machine - 1]
    x_tool_values = tool_settings[index_tool - 1]
    p_tool_values = tool_settings[parameters_tool - 1]
    csi_values = triplets[0,:]
    theta_values = triplets[1,:]
    phi_values = triplets[2,:]
    h_values = np.ones(num_points) * 0.0

    ub_h = h_values + 0.05
    lb_h = h_values - 0.05

    num_points_flank = num_points
    if root_constraint is not None:
        triplets_root = root_constraint['triplets']
        num_points_root = max(triplets_root.shape)
        csi_values = np.hstack([csi_values, triplets_root[0,:]])
        theta_values = np.hstack([theta_values, triplets_root[1,:]])
        phi_values = np.hstack([phi_values, triplets_root[2,:]])
        h_values = np.hstack([h_values, np.ones(num_points_root) * 0.0])
        ub_h = np.hstack([ub_h, np.ones(num_points_root) + 0.05])
        lb_h = np.hstack([lb_h, np.ones(num_points_root) - 0.05])
        num_points = num_points + num_points_root
    else:
        num_points_root = 0

    lb_machine_tool = lb
    ub_machine_tool = ub

    # numerical values for the scaling bounds
    csi_lb = np.maximum(csi_values - 0.1*csi_values.max(), 0)
    csi_ub = csi_values + 0.1*csi_values.max()
    theta_lb = theta_values - np.pi/10
    theta_ub = theta_values + np.pi/10
    phi_lb = phi_values - np.pi/10
    phi_ub = phi_values + np.pi/10

    pg_num = pg_fun(np.r_[csi_values, theta_values, phi_values]).full()
    pg_lb = pg_num - 0.1
    pg_ub = pg_num + 0.1
    pg_lb[:, -num_points_root:] = pg_num - 0.05
    pg_ub[:, -num_points_root:] = pg_num + 0.05

    ng_num = ng_fun(np.r_[csi_values, theta_values, phi_values]).full()
    ng_lb = ng_num - 0.1
    ng_ub = ng_num + 0.1
    ng_lb[:, -num_points_root:] = ng_num - 0.05
    ng_ub[:, -num_points_root:] = ng_num + 0.05

    Vgt_num = Vgt(machine_settings, phi_values).full().reshape((4, 4, num_points), order = 'F')
    Vgt_scaling = np.max(Vgt_num, axis = 2)
    Vgt_num = Vgt_num[0:3, :, :].reshape(12, num_points, order = 'F')
    Vgt_lb = Vgt_num - 8
    Vgt_ub = Vgt_num + 8

    # Define symbolic variables
    # normalized variables
    csi_n = ca.SX.sym('csi_n')
    theta_n = ca.SX.sym('theta_n')
    phi_n = ca.SX.sym('phi_n')
    h_n = ca.SX.sym('h_n')
    p_g_n = ca.SX.sym('p_g_n', 3)
    n_g_n = ca.SX.sym('n_g_n', 3)
    Vgt_n = ca.SX.sym('Vgt_n', 3, 4)

    # symbolic variables for the tool settings
    tool_settings_sym = ca.SX.sym('Ts', 10, 1)
    machine_settings_sym = ca.SX.sym('Ms', 72, 1)
    xM_n = machine_settings_sym[index_machine - 1]
    xT_n = tool_settings_sym[index_tool - 1]

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
    machine_settings[index_machine - 1] = xM_sym
    tool_settings[index_tool - 1] = xT_sym

    pg_expr = ggt(machine_settings, phi_sym) @ ca.vertcat(p_tool(tool_settings, ca.vertcat(csi_sym, theta_sym)), 1)
    ng_expr = ggt(machine_settings, phi_sym) @ ca.vertcat(n_tool(tool_settings, ca.vertcat(csi_sym, theta_sym)), 0)

    Vg_expr = Vgt_spatial(machine_settings, phi_sym)
    Vg_expr = Vg_expr[0:3, :]

    X_scale = pg_num[0,:].max()
    Y_scale = pg_num[1,:].max()
    Z_scale = pg_num[2,:].max()

    eq_meshing_scale = np.array([0.8, 0.8, 0.8, 0])@Vgt_scaling@np.array([X_scale, Y_scale, Z_scale, 1])

    # constraints for the generic point
    E_eq = (pg_sym - p_target_sym + h_sym * ng_sym)/0.5 # ease-off equation
    f_eq = ca.vertcat(ng_sym, 0).T @ Vgt_sym @ ca.vertcat(pg_sym, 1) # meshing equation
    point_eq = (pg_sym - pg_expr[0:3])/0.5  # congruence equation
    normal_eq = (ng_sym - ng_expr[0:3])/0.5  # normals congruence
    twist_eq = (Vgt_sym.vec() - Vg_expr.vec())/(ca.fabs(Vgt_scaling.flatten()) + 0.1) # twist equation


    return

def approxToolIdentification_casadi(data, member, RHO = None):

    RHOinput = RHO
    
    if member.lower() == 'pinion':
        cutterFieldName, subCutterFieldName, commonFieldName, subCommonFieldName, _, _ =\
            get_data_field_names('pinion', 'concave')
        cutterConvexFieldName, subConvexCutterFieldName, _, _, _, _ =\
            get_data_field_names('pinion', 'convex')
    else:
        cutterFieldName, subCutterFieldName, commonFieldName, subCommonFieldName, _, _ =\
              get_data_field_names('gear', 'concave')
        cutterConvexFieldName, subConvexCutterFieldName, _, _, _, _ =\
              get_data_field_names('gear', 'convex')
    
    # extract data from struct
    nT         = data[commonFieldName][f'{subCommonFieldName}NTEETH']
    edgeRadius = data[cutterFieldName][f'{subCutterFieldName}EDGERADIUS']
    rc0        = data[commonFieldName][f'{subCommonFieldName}MEANCUTTERRAIDUS']            # mean cutter radius
    alphanD    = data['SystemData']['NOMINALDRIVEPRESSUREANGLE']                             # drive side tooth pressure angle
    alphanC    = data['SystemData']['NOMINALCOASTPRESSUREANGLE']                             # coast side tooth pressure angle
    hamc       = data[commonFieldName][f'{subCommonFieldName}MEANCHORDALADDENDUM']         # tooth mean chordal addendum
    ham        = data[commonFieldName][f'{subCommonFieldName}MEANADDENDUM']               # tooth mean addendum
    t          = data[commonFieldName][f'{subCommonFieldName}MEANNORMALCHORDALTHICKNESS']   # tooth normal chordal thickness
    Rm         = data[commonFieldName][f'{subCommonFieldName}MEANCONEDIST']                # mean cone distance
    pitchapex  = data[commonFieldName][f'{subCommonFieldName}PITCHAPEX']
    delta      = data[commonFieldName][f'{subCommonFieldName}PITCHANGLE']
    betam      = data[commonFieldName][f'{subCommonFieldName}SPIRALANGLE']
    RHO        = data[cutterFieldName][f'{subCutterFieldName}RHO']
    hand       = data['SystemData']['HAND']
    mmn        = data['SystemData']['NORMALMODULE']

    if data[cutterFieldName][f'{subCutterFieldName}TYPE'].lower() == 'straight':
        RHO = 6000

    if RHOinput is not None:
        RHO = RHOinput
        data[cutterFieldName][f'{subCutterFieldName}RHO'] = RHO

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

    machinePar = assign_machine_par(data, member, 'concave')
    cMat, sMat = manage_machine_par(member, hand)
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

    """    # x_unscaled = ca.SX.sym('x', 17, 1) #+3*4+3*4
    # x = x_unscaled * (ub - lb) + lb
    # Rpcvx = x[0]
    # alphacvx = x[1]
    # alphacnv = x[2]
    # csiI = x[3]  # concave
    # thetaI = x[4]
    # csiO = x[5]  # convex
    # thetaO = x[6]
    # csiIprime = x[7]
    # thetaIprime = x[8]
    # csiOprime = x[9]
    # thetaOprime = x[10]
    # phi2 = x[11]
    # phiEnvIprime = x[12]
    # phiEnvOprime = x[13]
    # phiEnvI = x[14]
    # phiEnvO = x[15]
    # Rpcnv = x[16]
    # pO = x[17:20]
    # pI = x[20:23]
    # pOp = x[23:26]
    # pIp = x[26:29]
    # nO = x[17:20]
    # nI = x[20:23]
    # nOp = x[23:26]
    # nIp = x[26:29]"""

    alphacnv = pressAngCnv
    alphacvx = pressAngCvx
    toolO, toolNO = parametric_tool_casadi('convex', Rpcvx, RHO, alphacvx, edgeRadius, csiO, thetaO)
    toolI, toolNI = parametric_tool_casadi('concave', Rpcnv, RHO, alphacnv, edgeRadius, csiI, thetaI)

    T = sc.rigidInverse(Tng(phi2))
    pointO = T @ sc.TrotZ(signThick * 2 * np.pi / nT) @ ggt(phiEnvO) @ ca.vertcat(toolO, 1)
    pointI = T @ ggt(phiEnvI) @ ca.vertcat(toolI, 1)

    """    # toolOprime, toolNOprime = parametric_tool_casadi('convex', Rpcvx, RHO, alphacvx, edgeRadius, csiOprime, thetaOprime)
    # toolIprime, toolNIprime = parametric_tool_casadi('concave', Rpcnv, RHO, alphacnv, edgeRadius, csiIprime, thetaIprime)
    # pointOprime = ggt(phiEnvOprime) @ ca.vertcat(toolOprime, 1)
    # normalOprime = ggt(phiEnvOprime) @ ca.vertcat(toolNOprime, 0)
    # pointIprime = ggt(phiEnvIprime) @ ca.vertcat(toolIprime, 1)
    # normalIprime = ggt(phiEnvIprime) @ ca.vertcat(toolNIprime, 0)

    # pointOprime = T @ sc.TrotZ(signThick * 2 * np.pi / nT) @ pointOprime
    # normalOprime = T @ sc.TrotZ(signThick * 2 * np.pi / nT) @ normalOprime
    # pointIprime = T @ pointIprime
    # normalIprime = T @ normalIprime

    # out = ca.SX(17, 1)
    # out[0:3] = (pointO[0:3] - np.array([-(hamc - ham), +signThick * t / 2, 0]))
    # out[3:6] = (pointI[0:3] - np.array([-(hamc - ham), -signThick * t / 2, 0]))
    # out[6] = pointOprime[2]
    # out[7] = pointOprime[0] * normalOprime[1] - pointOprime[1] * normalOprime[0]
    # out[8] = ca.cos(pressAngCvx*pi/180) * ca.sqrt(normalOprime[0]**2 + normalOprime[1]**2) + signThick * normalOprime[1]
    # out[9] = pointIprime[2]
    # out[10] = pointIprime[0] * normalIprime[1] - pointIprime[1] * normalIprime[0]
    # out[11] = ca.cos(pressAngCnv*pi/180) * ca.sqrt(normalIprime[0]**2 + normalIprime[1]**2) - signThick * normalIprime[1]
    # out[12] = ca.vertcat(toolNOprime, 1).T @ Vgt(phiEnvOprime) @ ca.vertcat(toolOprime, 1)
    # out[13] = ca.vertcat(toolNIprime, 1).T @ Vgt(phiEnvIprime) @ ca.vertcat(toolIprime, 1)
    # out[14] = ca.vertcat(toolNO, 1).T @ Vgt(phiEnvO) @ ca.vertcat(toolO, 1)
    # out[15] = ca.vertcat(toolNI, 1).T @ Vgt(phiEnvI) @ ca.vertcat(toolI, 1)
    # out[16] = Rpcnv - (2 * rc0 - Rpcvx)

    # out[17:20] = pO-pointO[0:3]
    # out[17:20] = pI-pointI[0:3]
    # out[17:20] = pOp-pointOprime[0:3]
    # out[17:20] = pIp-pointIprime[0:3]
    # out[17:20] = nOp-pointOprime[0:3]
    # out[17:20] = nIp-pointIprime[0:3]"""

    equations = ca.vertcat(
        (pointO[0:3] - np.array([-(hamc - ham)*0, +signThick * t / 2, 0])),
        (pointI[0:3] - np.array([-(hamc - ham)*0, -signThick * t / 2, 0])),
        ca.vertcat(toolNO, 0).T @ Vgt(phiEnvO) @ ca.vertcat(toolO, 1),
        ca.vertcat(toolNI, 0).T @ Vgt(phiEnvI) @ ca.vertcat(toolI, 1),
        Rpcnv - (2 * rc0 - Rpcvx)
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

    data[cutterFieldName][f'{subCutterFieldName}RHO'] = RHO
    data[cutterConvexFieldName][f'{subConvexCutterFieldName}RHO'] = RHO
    data[cutterConvexFieldName][f'{subConvexCutterFieldName}POINTRADIUS'] = res[0]
    # data[cutterConvexFieldName][f'{subConvexCutterFieldName}BLADEANGLE'] = res[1]
    # data[cutterFieldName][f'{subCutterFieldName}BLADEANGLE'] = res[2]
    data[cutterFieldName][f'{subCutterFieldName}POINTRADIUS'] = res[-1]
    edge_radius = (res[-1] - res[0])/2.5
    data[cutterFieldName][f'{subCutterFieldName}EDGERADIUS'] = edge_radius
    data[cutterConvexFieldName][f'{subConvexCutterFieldName}EDGERADIUS'] = edge_radius
    triplet_concave = [res[1], res[2], res[6]]
    triplet_convex = [res[3], res[4], res[7]]

    return data, triplet_concave, triplet_convex
