# -*- coding: utf-8 -*-
"""
Screw calculus package for robotic chains and kinematic trees

This package is intended to be used within the casadi framework complemented by numpy

Created on Thu Jul 29 15:27:49 2021

@author: Eugeniu Grabovic

original implementation from Prof. Marco Gabiccini in a Mathematica package
"""

from casadi.casadi import exp
import numpy as np
import casadi as ca
#from math import sin, cos

# notes:
#  1) by default a numpy array is a column vector
#  2)
#  

"""
rotations and homogeneous transformations
""" 

def rotX(x):
    if isinstance(x, ca.SX) or isinstance(x, ca.MX):
        cx = ca.cos(x)
        sx = ca.sin(x)
    else:
        cx = np.cos(x)
        sx = np.sin(x)
    return np.array([
        [1,  0,   0],
        [0, cx, -sx],
        [0, sx,  cx]
        ])

def rotY(x):
    if isinstance(x, ca.SX) or isinstance(x, ca.MX):
        cx = ca.cos(x)
        sx = ca.sin(x)
    else:
        cx = np.cos(x)
        sx = np.sin(x)
    return np.array([
        [cx,  0, sx],
        [ 0,  1, 0],
        [-sx, 0, cx]
        ])

def rotZ(x):
    if isinstance(x, ca.SX) or isinstance(x, ca.MX):
        cx = ca.cos(x)
        sx = ca.sin(x)
    else:
        cx = np.cos(x)
        sx = np.sin(x)
    return np.array([
        [cx,-sx, 0],
        [sx, cx, 0],
        [ 0,  0, 1]
        ])

def rotZ2D(x):
    if isinstance(x, ca.SX) or isinstance(x, ca.MX):
        cx = ca.cos(x)
        sx = ca.sin(x)
    else:
        cx = np.cos(x)
        sx = np.sin(x)
    return np.array([
        [cx, -sx],
        [sx,  cx]
        ])

def rotToAxisAngle(R):
    theta = np.arccos((np.trace(R) - 1)/2)
    n = 1/(2*np.sin(theta))*(vecForm(R) - vecForm(np.transpose(R)))
    return (n, theta)

def rotNtheta(n, theta):
    if isinstance(n, ca.SX) or isinstance(n, ca.MX):
        N = n@ca.transpose(n)
    else:
        N = np.outer(n, n)
    return N + (np.eye(3) - N)*np.cos(theta) + hat(n)*np.sin(theta)

def rodriguezAxisAngle(ax, x):
    axisHat = hat(ax)
    R = np.eye(3) + axisHat*np.sin(x) + axisHat@axisHat*(1 - np.cos(x))
    return R

def RPTohomogeneous(R, p):
    p = p.reshape((3,1))
    if isinstance(R, ca.SX) or isinstance(R, ca.MX) or isinstance(p, ca.SX) or isinstance(p, ca.MX):
        return ca.vertcat(
            ca.horzcat(R, p),
            np.array([[0,0,0,1]])
        )
    return\
            np.concatenate(
            (np.concatenate((R, p), axis = 1),
            np.array([[0,0,0,1]])),
            axis = 0)
            
def TrotX(x):
    cx = np.cos(x)
    sx = np.sin(x)
    return \
        np.array([
            [1,  0,  0, 0],
            [0, cx,-sx, 0],
            [0, sx, cx, 0],
            [0,  0,  0, 1]
            ])

def TrotY(x):
    cx = np.cos(x)
    sx = np.sin(x)
    return np.array([
            [ cx, 0, sx, 0],
            [  0, 1,  0, 0],
            [-sx, 0, cx, 0],
            [  0, 0,  0, 1]
            ])

def TrotZ(x):
    cx = np.cos(x)
    sx = np.sin(x)
    return np.array([
            [cx,-sx, 0, 0],
            [sx, cx, 0, 0],
            [ 0,  0, 1, 0],
            [ 0,  0, 0, 1]
            ])

def TtX(x):
    return np.array([
            [1,0,0,x],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
            ])

def TtY(x):
    return np.array([
            [1,0,0,0],
            [0,1,0,x],
            [0,0,1,0],
            [0,0,0,1]
            ])

def TtZ(x):
    return np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,x],
            [0,0,0,1]
            ])

def TtP(x):
    return np.array([
        [1,0,0,x[0]],
        [0,1,0,x[1]],
        [0,0,1,x[2]],
        [0,0,0,1]
        ])

def DH_transformation(DH_parameters, q, joint_type):
    """
    Denavit-Hartemberg template transofmation and its differentiation
    DH_parameters = [a, alpha, d, theta]
    """
    a = DH_parameters[0]
    alpha = DH_parameters[1]
    d = DH_parameters[2]
    theta = DH_parameters[3]
    d_dq = 0
    dtheta_dq = 0
    # check joint type to include the joint variable q
    if joint_type.lower() == 'p': # prismatic
        d += q
        d_dq = 1
    elif joint_type.lower() == 'r': # revolute
        theta += q
        dtheta_dq = 1

    # trig expressionis
    cth = np.cos(theta)
    sth = np.sin(theta)
    calpha = np.cos(alpha)
    salpha = np.sin(alpha)
        
    T = np.array([
        [cth, -calpha*sth, salpha*sth, a*cth],
        [sth, calpha*cth, -salpha*cth, a*sth],
        [0, salpha, calpha, d],
        [0,0,0,1]
        ])
    
    # T = ca.vertcat(ca.horzcat(cth, -calpha*sth, salpha*sth, a*cth),
    #                ca.horzcat(sth, calpha*cth, -salpha*cth, a*sth),
    #                ca.horzcat(0, salpha, calpha, d),
    #                 ca.horzcat(0,0,0,1))
    
    dT_dtheta = np.array([
        [-sth, -cth*calpha, cth*salpha, -a*sth],
        [cth, -sth*calpha, sth*salpha, a*cth],
        [0,0,0,0],
        [0,0,0,0]
    ])
    dT_dd = np.zeros((4,4))
    dT_dd[2,3] = d_dq

    dT_dq = dT_dtheta + dT_dd

    return T, dT_dq

"""
Lie algebra operators
"""

def hat(x):
    if isinstance(x, ca.SX) or isinstance(x, ca.MX):
        if x.shape[0] == 3 or x.shape[1] == 3:
            return np.array([
                [0, - x[2], x[1]],
                [x[2], 0, -x[0]],
                [-x[1], x[0], 0]
                ])
        elif x.shape[0] == 6 or x.shape[1] == 6:
            return ca.vertcat(
            ca.horzcat(hat(x[3:]), x[0:3]),
            np.array([[0,0,0,0]])
            )
            
    if x.size == 3:
        x = x.reshape(3,)
        return np.array([
            [0, - x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
            ])
    elif x.size == 6:
        x = x.reshape(6,)
        return ca.vertcat(
            ca.horzcat(hat(x[3:]), x[0:3]),
            np.array([[0,0,0,0]])
            )
    else:
        raise ValueError('wrong shape of the input in hat(x) function')
        
def ad(x):
    """
    ad function 
    """
    w = x[3:6]
    v = x[0:3]
    what = hat(w)
    return np.concatenate((
        np.concatenate((what, hat(v)), axis=1),
        np.concatenate((np.zeros((3,3)), what), axis=1)),
        axis=0)

def adStar(x):
    """
    adStar function  helper
    """
    w = x[3:6]
    v = x[0:3]
    what = hat(w)
    return np.concatenate((
        np.concatenate((what, np.zeros((3,3))), axis=1),
        np.concatenate((hat(v), what), axis=1)),
        axis=0)

def adjoint(x):
    R = x[0:3,0:3]
    d = x[0:3,3]
    if isinstance(x, ca.SX):
        return ca.vertcat(ca.horzcat(R, hat(d)@R),
            ca.horzcat(np.zeros((3,3)), R))
    else:
        return np.r_[np.c_[R, hat(d)@R],
                    np.c_[np.zeros((3,3)), R]]

def adjointInv(x):
    d = x[0:3,3]
    R = (x[0:3,0:3]).T
    if isinstance(x, ca.SX):
        return ca.vertcat(ca.horzcat(R, -R@hat(d)),
            ca.horzcat(np.zeros((3,3)), R))
    else:
        return np.r_[np.c_[R, -R@hat(d)],
                np.c_[np.zeros((3,3)), R]
                ]

def adjointStar(x):
    d = x[0:3,3]
    R = (x[0:3,0:3])
    if isinstance(x, ca.SX):
        return ca.vertcat(ca.horzcat(R, np.zeros((3,3))),
            ca.horzcat(hat(d)@R, R))
    return np.r_[np.c_[R, np.zeros((3,3))],
                  np.c_[hat(d)@R, R]]
    
def rigidInverse(x):
    R = x[0:3,0:3]
    d = x[0:3,3]
    return RPTohomogeneous(R.T, (-R.T)@d)

"""twist exponentials"""

def expSkew(axis, x):
    """
    computes the exponential matrix associated to the infinitesimal rotation about axis of an angle x
    simplifications are made when the axis concides with one of the main axes (X, Y, Z)
        which holds true when the axis is given in numeric form
        cannot compute with generic symbolic axis
    """
    if abs(axis[0]) <= 1e-8 and abs(axis[1])  <= 1e-8: #simple rotation about Z axis
        return rotZ(x*np.sign(axis[2]))
    elif abs(axis[1]) <= 1e-8 and abs(axis[2]) <= 1e-8: #simple rotation about X axis
        return rotX(x*np.sign(axis[0]))
    elif abs(axis[0]) <= 1e-8 and abs(axis[2]) <= 1e-8: #simple rotation about Y axis
        return rotY(x*np.sign(axis[1]))
    else:
        return rotNtheta(axis, x)

def unitTwist(h, axis, q):
    """
    returns unitary twist starting from joint axis and distance from origin represented by vector q
    the joint type is defined by the helix lead h (h=Inf -> prismatic, h=0 -> revolute, h = finite -> helical)
    """
    axis = axis.reshape(3,)
    q = q.reshape(3,)
    if h == np.Inf: # prismatic
        uT = np.concatenate((axis, 0, 0, 0), axis = 0)
    else:
        uT = np.concatenate(
            (-np.cross(axis, q) + h*axis, axis),
        axis = 0)

def vecForm(x):
    if x.size == 9: # 3 by 3 skew sym matrix
        return np.array([x[2,1], x[0,2], x[1,0]])
    elif x.size == 16:
        return np.concatenate(
            (x[0:3,3],
            vecForm(x[0:3, 0:3])),
            axis = 0
        )

def expTw(unitTwist, x, helix):
    """
    exponential matrix of a twist
    - unitTwist is the unitary twist direction vector
    - x is the joint value that represents the twist
    - helix is a numerical value between 0 and Inf representing the helical value of the joint:
        * 0 -> revolute joint
        * Inf -> prismatic joint
        * finite number =/= 0 -> generic helical joint
    """
    if helix == np.Inf: # it is a prismatic joint
        axis = unitTwist[0:3]
        R = np.eye(3)
        d = axis*x
    else: # it is either revolute or helical joint (depending on helix value)
        axis = unitTwist[3:]
        v = unitTwist[0:3]
        q = -np.cross(v,axis)
        R = expSkew(axis, x)
        d = (np.eye(3) - R)@q + helix*axis*x
    return RPTohomogeneous(R, d)

"""kinematic chains with different parametrizations"""

def DHFWkin(DH_table, q, joint_type):
    """
    forward kinematics calculation with DH parameterization
    """
    n = np.max(q.shape)
    Tj = [0]*n
    T0j = [0]*n
    Tj[0], _ = DH_transformation(DH_table[0,:], q[0], joint_type[0])
    T0E = Tj[0]
    T0j[0] = Tj[0]

    for j in range(1,n):
        Tj[j], _ = DH_transformation(DH_table[j,:], q[j], joint_type[j])
        T0E @=Tj[j]
        T0j[j] = T0E

    return T0E, Tj, T0j

def FWkin_globalPOE_v2(gst0, *joints):
    """
    Forward kinematics computation with global POE parametrization
    gst0: initial offset matrix
    joints: joints definition as a tuple of 3 entries:
        1 unitary twist
        2 joint value
        3 helix value (for helical rototranslation)
    """
    n = len(joints)
    g = expTw(*(joints[0])) # unpacking the joint triplet value directly into expTw funciton
    for i in range(1,n):
        g = g@expTw(*(joints[i]))
    g = g@gst0
    return g

def FWkin_globalPOE(gst0, twists, q, helix):
    """
    Forward kinematics computation with global POE parametrization
    gst0: initial offset matrix
    joints: joints definition as a tuple of 3 entries:
        1 unitary twist
        2 joint value
        3 helix value (for helical rototranslation)
    """
    n = len(helix)
    g = expTw(twists[:, 0], q[0], helix[0]) # unpacking the joint triplet value directly into expTw funciton
    for i in range(1,n):
        g = g@expTw(twists[:, i], q[i], helix[i])
    g = g@gst0
    return g

def FWkin_localPOE(G_offset, *joints, jstart = 0, jend = []):
    n = len(joints)
    if isinstance(jend, list):
        jend = n
    Glocals = [None]*n
    Gglobal = np.eye(4)
    for i in range(jstart, jend):
        Glocals[i] = G_offset[i]@expTw(*(joints[i]))
        Gglobal = Gglobal@Glocals[i]
    return Gglobal, Glocals

def bodyJac_globalPOE_v2(gst0, *joints):
    """
    body jacobian of a serial manipulator with global POE formulation
    """
    n = len(joints)
    g = gst0
    J = ca.SX.zeros((6, n))
    for i in range(n-1, -1, -1):
        g = expTw(*(joints[i]))@g
        J[:, i] = adjointInv(g)@joints[i][0]
    return J

def bodyJac_globalPOE(gst0, twists, q, helix):
    """
    body jacobian of a serial manipulator with global POE formulation
    """
    n = len(helix)
    g = gst0

    if isinstance(q[0], ca.SX):
        J = ca.SX(6, n)
    elif isinstance(q[0], ca.MX):
        J = ca.MX(6, n)
    else:
        J = np.zeros((6, n))
    
    for i in range(n-1, -1, -1):
        g = expTw(twists[:, i], q[i], helix[i])@g
        J[:, i] = adjointInv(g)@(twists[:,i])
    return J

def spatialJac_globalPOE_v2(*joints):
    """
    spatial jacobian of a serial manipulator with global POE formulation
    """
    n = len(joints)
    g = expTw(*(joints[0]))
    J = np.zeros((6,n), dtype = object)
    J[:, 0] = joints[0][0]
    for i in range(1, n):
        J[:, i] = adjoint(g)@joints[i][0]
        g = g @expTw(*(joints[i]))
    return J

def spatialJac_globalPOE(twists, q, helix):
    """
    spatial jacobian of a serial manipulator with global POE formulation
    """
    n = q.shape[0]
    g = expTw(twists[:,0], q[0], helix[0])

    if isinstance(q[0], ca.SX):
        J = ca.SX(6, n)
    elif isinstance(q[0], ca.MX):
        J = ca.MX(6, n)
    else:
        J = np.zeros((6, n))

    J[:, 0] = twists[:,0]
    for i in range(1, n):
        J[:, i] = adjoint(g)@(twists[:,i])
        g = g @expTw(twists[:,i], q[i], helix[i])
    return J

def bodyJac_localPOE(G_offset, *joints, EE_offset = np.eye(4)):
    n = len(joints)
    J = np.zeros((6, n), dtype = object)
    Binv = rigidInverse(EE_offset)
    for i in range(n-1, -1, -1):
        Binv = Binv@expTw(-joints[i][0], joints[i][1], joints[i][2])
        J[:, i] = adjoint(Binv)@joints[i][0]
        Binv = Binv@rigidInverse(G_offset[i])
    return J

def spatialJac_localPOE(G_offset, *joints, EE_offset = np.eye(4)):
    G_offset[-1] = G_offset[-1]@EE_offset
    n = len(joints)
    J = np.zeros((6,n), dtype=object)
    B = np.eye(4)
    for i in range(0,n-1):
        B = B@G_offset[i]
        J[:, i] = adjoint(B)@joints[i][0]
        B = B@expTw(*(joints[i]))
    # final joint and offset
    B = B@G_offset[-1]
    J[:, -1] = adjoint(B)@adjoint(EE_offset)@joints[-1][0]
    return J

def DHJac(T0j, joint_type, joint_index = None, base_offset = np.eye(4), end_offset = np.eye(4)):
    """
    joint_index: number of jonit w.r.t. jacibian is computed; None = end_effector
    base_offet: homogeneous matrix transform defining the base offset
    end_offset: homogeneous matrix transform defining the end-effector offset
    """
    n = len(T0j) # using len cause T0j is a tuple

    if joint_index is not None:
        n = joint_index
    
    O_0n = T0j[n-1][0:3, 3] # extracting origin point of the last frame

    Jac = np.zeros((6, len(T0j)), dtype=T0j[0].dtype)
    z_axis = np.array([0,0,1])
    if joint_type[0].lower() == 'P':
        Jac[2,0] = 1
    elif joint_type[0].lower() == 'R':
        Jac[:, 0] = np.concatenate((np.cross(z_axis, O_0n), z_axis), axis = 0)

    for jj in range (1, n):
        T0j_1 = T0j[jj-1]
        k = T0j_1[0:3, 2]
        O_0j = T0j_1[0:3, 3]
        O_jn = O_0n-O_0j

        if joint_type[jj].lower() == 'p':
            Jac[0:3, jj] = k
        elif joint_type[jj].lower() == 'r':
            Jac[:, jj] = np.concatenate((np.cross(k, O_jn), k),axis = 0)

    return Jac

    return
def toSO3(R):
    """
    Returns the projection of a 3x3 matrix to SO(3) (antysym orthogonal)
    can perform SO(3) projection on multiple matrices if a 3x3xN tensor is given
    """
    RinSO3 = np.empty(shape = R.shape)
    S = np.eye(3)
    if len(R.shape) == 3:
        for i in range(0, R.shape[2]):
            U, Sig, V = np.linalg.svd(R[:,:,i])
            VT = np.transpose(V)
            S[2,2] = np.linalg.det(U@VT)
            RinSO3[:,:,i] = U@S@VT
    else:
        U, Sig, V = np.linalg.svd(R)
        VT = np.transpose(V)
        S[2,2] = np.linalg.det(U@VT)
        RinSO3 = U@S@VT

    return RinSO3

def twistPole(x):
    """
    d = homogeneous matrix or pole vector (from old pole to new)
    """
    if x.size == 3:
        x = x.reshape(3,)
    elif x.size == 16:
        x = x[0:3, 3]
        x = x.reshape(3,)    
    I = np.eye(3)
    Z = np.zeros(shape = (3,3))
    return np.concatenate(
            (
            np.concatenate((I, hat(x)), axis = 1),
            np.concatenate((Z,      I), axis = 1),
            ),
            axis = 0)

def wrenchPole(x):
    """
    Function to represent a wrench w.r.t. to a different pole (point w.r.t. which the mechanical moments are computed)
    """
    I = np.eye(3)
    Z = np.zeros(shape=(3,3))
    if x.size == 3:
        x = x.reshape(3,)
    elif x.size == 16:
        x = x[0:3, 3]
        x = x.reshape(3,)    
    return np.concatenate(
        (
            np.concatenate((     I, Z), axis = 1),
            np.concatenate((hat(x), I), axis = 1)
        ),
        axis = 0
    )

def rotNthetaToQuat(n, theta):
    p0 = np.cos(theta/2)
    sth = np.sin(theta/2)
    p1 = sth*n[0]
    p2 = sth*n[1]
    p3 = sth*n[2]
    return np.array([p0,p1,p2,p3])

# known jacobians for orientation parametrizations

def eulParSpatialJac(q):
    """
    Jacobian associated with the Euler parameters (QUATERNIONS)
    """
    if isinstance(q, ca.SX) or isinstance(q, ca.MX):
        q = ca.reshape(q, 4, 1)
        qvec = q[1:]
        return 2*ca.horzcat(
            -qvec, np.eye(3)*q[0]+hat(qvec)
            )
    else:
        q = q.reshape(4,1)
        qvec = q[1:]
        return 2*np.concatenate(
            (-qvec, np.eye(3)*q[0]+hat(qvec)),
            axis = 1
            )

def eulParSpatialJacInv(q):
    """
    Inverse Jacobian associated with the Euler parameters (QUATERNIONS)
    """
    if isinstance(q, ca.SX) or isinstance(q, ca.MX):
        q = ca.reshape(q, 4, 1)
        qvec = q[1:]
        return 2*ca.vertcat(
            -ca.transpose(qvec), np.eye(3)*q[0]-hat(qvec)
            )
    else:
        q = q.reshape(4,1)
        qvec = q[1:]
        return 0.5*np.concatenate(
            (-np.transpose(qvec), np.eye(3)*q[0]-hat(qvec)),
            axis = 0
            )

#==============================================================================
#
#  CASADI COMPATIBLE INTEGRATORS (just Runge Kutta and direct collocation)
#
#==============================================================================

def RK4_step(x, u, xdot, dt, Nsteps = 1):
    """
    help
    """
    h = dt/Nsteps

    for ii in range(0, Nsteps):
        a1 = xdot(x, u)
        a2 = xdot(x + h/2*a1, u)
        a3 = xdot(x + h/2*a2, u)
        a4 = xdot(x + h*a3, u)
        return x + h/6*(a1+ 2*a2 + 2*a3 + a4)
    
def RK4(x_expr, u_expr, xdot_expr, x0, t0, t_end, dt, u_in, Nsteps = 1, t_expr = None):
    """
    helper
    """
    if type(x0) is list:
        x0 = np.array(x0)

    # check if system is TV
    if t_expr is not None:
        x_expr = ca.vertcat(x_expr, t_expr)
        x0 = np.r_[x0, t0]
        xdot_expr = ca.vertcat(x_expr, ca.DM(1))

    steps = int(np.floor((t_end-t0)/dt))
    x_num = np.zeros((np.size(x0), steps+1))
    x_num[:, 0] = x0

    if u_expr is None:
        sz = 1
        u_expr = []
    else:
        sz = u_expr.shape

    if not u_in:
        u_in = np.zeros((sz, steps))

    # make a casadi function from expressions
    xdot_fun = ca.Function('xdot_fun', [x_expr, u_expr], [xdot_expr])
    x_next_fun = ca.Function('xdot', [x_expr, u_expr], [RK4_step(x_expr, u_expr, xdot_fun, dt, Nsteps = Nsteps)])

    
    for ii in range(1, steps+1):
        x_num[:, ii] = x_next_fun(x_num[:, ii-1], u_in[:, ii-1]).full().squeeze()  

    return x_num

def DC_coefficients(d, method = 'legendre'):
    """
    precomputing the direct collocation coefficient matrices
    the coefficients are the pre-computed lagrange basis at the collocation points

    polynomial function apporximation, evaluated at the k-th collocation points is equal to:
    x_dot_{k} = sum_{i=0}^{d} ( xcolloc_{i}*C_{ki} )

    polynomial expression at the k-th collocation point is simply equal to:
    x_{k} = xcolloc_{k}
    """
    tau_root = ca.collocation_points(d, method)
    tau_root.insert(0, 0) # insert interval origin as node

    # coefficients at the collocation points, polynomial derivative at collocation points l_dot(tau_i)
    # each COLUMN are the Lagrange bases at the specific collocation point (collocation points along columns)
    C = np.zeros((d+1, d+1))

    # coefficients of the continuity equation
    D = np.zeros((d+1))

    # coefficients of the quadrture function
    B = np.zeros((d+1))

    for j in range(0, d+1): # loop over collocation points for which the coefficients are computed
        coeff = np.array([1])
        for r in range(0, d+1):
            if r != j:
                coeff = np.convolve(coeff, np.array([1, -tau_root[r]]))
                coeff = coeff/(tau_root[j] - tau_root[r])

        # evaluate polynomial at the interval end
        D[j] = np.polyval(coeff, 1) 

        # evaluate the polynomial derivative at colloc. points
        coeff_der = np.polyder(coeff)
        for r in range(0, d+1):
            C[j, r] = np.polyval(coeff_der, tau_root[r])
        
        # evaluate the integral of the polynomial over the interval
        coeff_int = np.polyint(coeff)
        B[j] = np.polyval(coeff_int, 1)

    return tau_root, C, D, B

def DC_step(x, u, xdot, dt, tau, D, C, B):
    """
    To compute the step we have to satisfy collocation equations:
    x_colloc_dot = f(x_colloc, u_colloc)
    for each collocation point (d in total)
    once we have satisfied the collocation points we just have to compute the state at the end interval
    x_next = sum_i x_colloc_i*D_i
    """
    d = tau.len()
    return

def DC_integrator(x, u, xdot, dt, d = 3):
    """
    implicit direct collocation integration
    """
    return

# =============================================================================
# 
#  MAIN SCRIPTs FOR TEST (examples)
# 
# =============================================================================

def main_integrators():
    import matplotlib.pyplot as plt

    # try to simulate Lorentz attractor
    sigma = 3
    beta = 1
    rho = 26.5
    s = []
    print(s == None)
    x = ca.SX.sym('x', 3, 1)

    xdot_expr = ca.vertcat(sigma*(x[1]-x[0]), rho*x[0] - x[0]*x[2]-x[1], x[0]*x[1]-beta*x[2])

    t0 = 0
    t_end = 20
    dt = 0.025

    x0 = [1, 1, 1]
    x02 = [1.0001,1,1]

    x_simulation = RK4(x, None, xdot_expr, x0, t0, t_end, dt, None, Nsteps = 1, t_expr = None)
    x_simulation2 = RK4(x, None, xdot_expr, x02, t0, t_end, dt, None, Nsteps = 1, t_expr = None)
    x_diff = x_simulation - x_simulation2
    print(x_simulation[:, 0])   # first point
    print(x_simulation[:, -1])  # last point

    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(elev=30, azim=45, roll=15)
    ax.plot(x_simulation[0,:], x_simulation[1,:], x_simulation[2,:], color = 'black')
    # ax.plot(x_simulation2[0,:], x_simulation2[1,:], x_simulation2[2,:], color = 'blue')
    # ax.plot(x_diff[0,:], x_diff[1,:], x_diff[2,:], color = 'red')
    plt.show()

def main_localPOE():
    import matplotlib.pyplot as plt

def main_globalPOE():
    import matplotlib.pyplot as plt

def main_DH_Stanford():
    import matplotlib.pyplot as plt
    import easy_plot as ep

    n = 6 # number of joints

    # Denavit-Hartemberg table
    DH_table = np.array([
        [0, -np.pi/2, 0.4, 0],
        [0, np.pi/2, 0.4, 0],
        [0, 0, 0, 0],
        [0, np.pi/2, 0.4, -np.pi/2],
        [0, np.pi/2, 0, np.pi/2],
        [0, 0, 0.4, 0]
    ])

    # joint types
    joint_type = ['R']*n
    joint_type[2] = 'P'

    # casadi joint variables
    q = ca.SX.sym('q', 6, 1)

    # Forward kinematics
    T0E, Tj, T0j = DHFWkin(DH_table, q, joint_type)
    # Jacobian
    Jac_expr = DHJac(T0j, joint_type, joint_index = None, base_offset = np.eye(4), end_offset = np.eye(4))
    Jac_fun = ca.Function('Jac', [q], [Jac_expr])
    
    # casadi expressions to functions
    T0E_fun = ca.Function('T0E', [q], [T0E])
    Tj_fun = ca.Function('Tj', [q], Tj)

    # evaluating functions numerically
    T0E_num = T0E_fun(np.zeros((n, 1)))
    Tj_num = Tj_fun(np.zeros((n, 1)))
    
    
    

    F = ep.Figure(title= 'Stanford Manipulator')

    joints = []
    parent = None
    for ii in range(0, len(Tj_num)):
        joints.append(ep.revolute_joint(F, 0.05, ax = np.array([0,0,1]), lenBot=0.15, lenTop=0.15, parent = parent))
        joints[ii].setTransform(Tj_num[ii].full())
        parent = joints[ii]

    F.updateImage()
    F.show()
    




def main_debug():
    x = ca.SX.sym('x')

    T = TtX(x)
    T_fun = ca.Function('T',[x], [T])
    Tjac = ca.jacobian(T, x)
    print(type(Tjac))

if __name__ == "__main__":
    # main_integrators()
    # main_localPOE()
    # main_globalPOE()
    main_DH_Stanford()
    # main_debug()


    