import casadi as ca
import screwCalculus as sc
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from   pyvistaqt import BackgroundPlotter

def basis_function(u, p, U):
    """
    This basis function evaluation allows only numerical evaluation
    u: coordinate value
    p: degree
    U: knot vector
    i: knot interval
    """
    if not isinstance(U, np.ndarray):
        U = np.array(U)

    i = np.argwhere(u>U)
    if i.shape[0] != 0:
        i = i[-1][0]

    if u >= U[-1]:
        i = max(U.shape) - p - 2
    if u <= U[0]:
        i = p

    # arrays initialization
    N = np.zeros((p+1)) # basis vector with non vanishing elements
    left = np.zeros((p+1))
    right = np.zeros((p+1))

    N[0] = 1
    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        res = 0
        for r in range(0, j):
            temp = N[r]/(right[r+1]+left[j-r])
            N[r] = res + right[r+1]*temp
            res = left[j-r]*temp
        N[j] = res

    # need to fill the basis vector with vanishing elements to easily perform matrix operations
    lenU = max(U.shape)
    n = lenU - p - 1
    endL = n - i - 1
    z1 = np.array([])
    z2 = np.array([])
    if endL > 0:
        z1 = np.zeros((endL))
    if i-p > 0:
        z2 = np.zeros((i-p))
    N = np.r_[z2, N, z1]

    return N

def der_basis_fun_i(u, i, p, U, derOrder):
    """
    This basis function evaluation allows only numerical evaluation
    u: coordinate value
    i: knot interval
    p: degree
    U: knot vector
    derOrder: order of derivative to compute
    """
    if not isinstance(U, np.ndarray):
        U = np.array(U)

    # arrays initialization
    ndu = np.zeros((p + 1, p + 1)) # basis matrix with non vanishing elements
    left = np.zeros((p+1))
    right = np.zeros((p+1))
    if isinstance(u, ca.SX) or isinstance(u, ca.MX):
        ndu = ca.SX(p + 1, p + 1) # basis matrix with non vanishing elements
        left = ca.SX(1, p+1)
        right = ca.SX(1, p+1)
    ndu[0, 0] = 1

    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        res = 0
        for r in range(0, j):
            ndu[j, r] = right[r+1]+left[j-r]
            temp = ndu[r, j-1]/(ndu[j, r])
            ndu[r, j] = res + right[r+1]*temp
            res = left[j-r]*temp
        ndu[j, j] = res

    # need to fill the basis vector with vanishing elements to easily perform matrix operations
    ders = np.zeros((derOrder+1, p+1))
    a = np.zeros((2, p+1))
    if isinstance(u, ca.SX) or isinstance(u, ca.MX):
        ders = ca.SX(derOrder+1, p+1) # basis matrix with non vanishing elements
        a = ca.SX(2, p+1)
    ders[0, :] = ndu[:, p]

    for r in range(0,p+1):
        s1 = 0
        s2 = 1
        a[0,0] = 1

        for k in range(1, derOrder+1):
            d = 0
            rk = r - k
            pk = p - k

            if r >= k:
                a[s2, 0] = a[s1, 0]/ndu[pk+1,rk]
                d = a[s2, 0]*ndu[rk, pk]

            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r-1 <= pk:
                j2 = k-1
            else:
                j2 = p-r

            for j in range(j1,j2+1):
                a[s2, j] = (a[s1, j] - a[s1,j-1])/ndu[pk + 1, rk + j]
                d += a[s2, j]*ndu[rk + j, pk]

            if r <= pk:
                a[s2, k] = -a[s1, k-1]/ndu[pk + 1, r]
                d += a[s2, k]*ndu[r, pk]

            ders[k, r] = d
            j = s1
            s1 = s2
            s2 = j

    r1 = p
    for k in range (1, derOrder+1):
        ders[k, :] = ders[k, :]*r1
        r1 *= (p-k)

    # need to add zeros to null bases
    lenU = max(U.shape)
    numCtrlPts = lenU-p-1
    endLen = numCtrlPts - i -1
    zers1 = np.zeros((derOrder+1, 0))
    zers2 = np.zeros((derOrder+1, 0))
    if endLen > 0:
        zers1 = np.zeros((derOrder+1, endLen))

    if i-p > 0:
        zers2 = np.zeros((derOrder+1, i-p))

    if isinstance(u, ca.SX) or isinstance(u, ca.MX):
        return ca.horzcat(zers2, ders, zers1).T
    return np.hstack([zers2, ders, zers1]).T

def basis_fun_casadi(p, U, derOrder):

    if not isinstance(U, np.ndarray):
        U = np.array(U)

    u = ca.SX.sym('u')
    i_start = np.argwhere(U == 0)
    i_start = i_start[-1][0]
    i_end = np.argwhere(U == 1)
    i_end = i_end[0][0]
    Z = ca.SX(max(U.shape)-(p+1), 1)

    for i in range(i_start,i_end):
        if i == i_end-1:
            bool = ca.logic_and(u>=U[i], u<=U[i+1])
        else:
            bool = ca.logic_and(u>=U[i], u<U[i+1])
        Z += bool*der_basis_fun_i(u, i, p, U, derOrder)
    return ca.Function('N', [u], [Z])

def chord_length_nodes(pU, pV, Q):
    """
    pU
    pV
    Q
    """
    shp = Q.shape
    n = shp[1]
    m = shp[2]

    u = np.full((m, n), np.nan)
    v = np.full((n, m), np.nan)
    X = np.reshape(Q[0,:,:], (n, m), order = 'F')
    Y = np.reshape(Q[1,:,:], (n, m), order = 'F')
    Z = np.reshape(Q[2,:,:], (n, m), order = 'F')


    for ii in range(0, m):
        u[ii, 0:n] = 0
        u[ii, -1] = 1

        cds = np.sqrt( np.diff(X[:, ii])**2 + np.diff(Y[:,ii])**2 + np.diff(Z[:,ii])**2 )
        total = np.sum(cds)
        d = 0;
        for kk in range(0, max(cds.shape)):
            u[ii, kk] = d/total
            d += cds[kk]
    u = np.sum(u, axis=0)/m

    for ii in range(0, n):
        v[ii, 0:m] = 0
        v[ii, -1] = 1

        cds = np.sqrt( np.diff(X[ii, :])**2 + np.diff(Y[ii, :])**2 + np.diff(Z[ii, :])**2 )
        total = np.sum(cds)
        d = 0;
        for kk in range(0, max(cds.shape)):
            v[ii, kk] = d/total
            d += cds[kk]
    v = np.sum(v, axis=0)/n

    knotsU = np.full((n + pU + 1), np.nan)
    knotsU[0:pU] = 0
    for j in range(0, n-pU):
        knotsU[j+pU] = 1/pU*sum(u[j: j+pU])

    knotsU[-pU:] = 1

    knotsV = np.full((m + pV + 1), np.nan)
    knotsV[0:pV] = 0
    for j in range(0, m-pV):
        knotsV[j+pV] = 1/pU*sum(u[j: j+pV])

    knotsV[-pV:] = 1

    return u, v, knotsU, knotsV

def fit_knots(uk, num_control_points, degree):
    n = max(uk.shape)
    knotsU = np.full((num_control_points + degree + 1), np.nan)
    knotsU[0:degree+1] = 0
    knotsU[-(degree+1):] = 1
    d = n/(num_control_points - degree)
    for jj in range(0, num_control_points-1-degree):
        i = int((jj+1)*d)
        alpha = (jj+1)*d - i
        knotsU[degree+jj+1] = (1-alpha)*uk[i-1] + alpha*uk[i]

    # for the moment make just a linear spacing of knots, in the inner part
    knotsU[degree:num_control_points+1] = np.linspace(0, 1, num_control_points - degree + 1)

    return knotsU

def initialize_step_writer():
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.Interface import Interface_Static
    step_writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP203")

    return step_writer

def Nurbs_to_STEPwriter(Nurbs, step_writer):
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Display.SimpleGui import init_display
    from OCC.Core.Geom import Geom_BSplineSurface
    import OCC.Core.Geom
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array2OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCC.Core.GeomLib import geomlib_ExtendSurfByLength
    #from OCC.Core.GeomLib import ExtendSurfByLength
    import OCC.Core.GeomLib

    pU = Nurbs.degreeU
    pV = Nurbs.degreeV

    Vknots = Nurbs.knotsV.tolist()
    Uknots = Nurbs.knotsU.tolist()

    control_points = Nurbs.controlPoints
    gridSizeControlPoints = control_points.shape
    gridSizeControlPoints = [gridSizeControlPoints[1], gridSizeControlPoints[2]] # transposing the control points matrix to match the pythonOCC array grid

    # initializing pythonOCC array grid
    pOCArray = TColgp_Array2OfPnt(0, gridSizeControlPoints[0]-1, 0, gridSizeControlPoints[1]-1)

    # copy those points in the pyOCC array grid
    for i in range(0, gridSizeControlPoints[0]):
        for j in range(0, gridSizeControlPoints[1]):
            pOCArray.SetValue(i, j, gp_Pnt(control_points[0,i,j],control_points[1,i,j], control_points[2,i,j]) )

    # removing knots multiplicity; pythonOCC requires unique knots values followed by a MULT array that defines multiplicity of each knot
    for i in range(pU):
        Uknots.pop(0)  # dropping zeros at beginning
        Uknots.pop(-1) # dropping ones at the end

    for i in range(pV):
        Vknots.pop(0)   # dropping zeros at beginning
        Vknots.pop(-1)  # dropping ones at the end

    UknotsOCC = TColStd_Array1OfReal(0, len(Uknots)-1)  # initializing pythonOCC arrays for Uknots
    VknotsOCC = TColStd_Array1OfReal(0, len(Vknots)-1)  # initializing pythonOCC arrays for Vknots
    multU     = TColStd_Array1OfInteger(0, len(Uknots)-1)  # initializing pythonOCC arrays for multiplicity of Uknots
    multV     = TColStd_Array1OfInteger(0, len(Vknots)-1)  # initializing pythonOCC arrays for multiplicity of Vknots

    # for loop to assign array elements... sadly can't just type OCCarray = pythonList. We need to use the SetValue method for each element
    for i in range(len(Uknots)):
        UknotsOCC.SetValue(i, Uknots[i])
        # check if it is the first or last knot
        if i == 0 or i == len(Uknots)-1:   # setting multiplicity to degree + 1
            multU.SetValue(i, pU+1)        # to the first and the final values
            continue
        # set multiplicity to 1 otherwise
        multU.SetValue(i, 1)

    # for loop to assign Vknots values and setting multiplicity array of V
    for i in range(len(Vknots)):
        VknotsOCC.SetValue(i, Vknots[i])
        # check if it is the first or last knot
        if i == 0 or i == len(Vknots)-1: # setting multiplicity to degree + 1
            multV.SetValue(i, pV+1)      # to the first and the final values
            continue
        # set multiplicity to 1 otherwise
        multV.SetValue(i, 1)

    NURBSsurf = Geom_BSplineSurface(pOCArray, UknotsOCC, VknotsOCC, multU, multV, pU, pV)
    BRepObj = BRepBuilderAPI_MakeFace(NURBSsurf, 0, 1, 0, 1, 1e-6)
    step_writer.Transfer(BRepObj.Shape(), STEPControl_AsIs)

    return NURBSsurf

def fit_nurbs_surface(target_points, degreeU, degreeV, control_points_shape):
    """

    """
    Q = target_points
    if isinstance(target_points, dict):
        shp = Q['X'].shape
        R = np.zeros((3, shp[0], shp[1]))
        R[1,:,:] = Q['X']
        R[2,:,:] = Q['Y']
        R[3,:,:] = Q['Z']
        Q = R

    shp = Q.shape
    Xint = np.reshape(Q[0,:,:], (shp[1], shp[2]), order='F')
    Yint = np.reshape(Q[1,:,:], (shp[1], shp[2]), order='F')
    Zint = np.reshape(Q[2,:,:], (shp[1], shp[2]), order='F')

    pU = degreeU
    pV = degreeV

    uk, vk, _, _ = chord_length_nodes(pU, pV, Q)
    knotsU = fit_knots(uk, control_points_shape[0], degreeU)
    knotsV = fit_knots(vk, control_points_shape[1], degreeV)
    lenU = knotsU.shape[0]
    lenV = knotsV.shape[0]

    # basis functions calculation
    Ni = np.full((uk.shape[0], lenU-pU-1), np.nan)
    for kk in range(0, uk.shape[0]):
        Ni[kk, :] = basis_function(uk[kk], pU, knotsU)

    Nj = np.full((vk.shape[0], lenV-pV-1), np.nan)
    for kk in range(0, vk.shape[0]):
        Nj[kk, :] = basis_function(vk[kk], pV, knotsV)

    Wi = np.eye(Ni.shape[0])
    Wj = np.eye(Nj.T.shape[0])

    pseudoNi = np.linalg.pinv(Ni)
    pseudoNj = np.linalg.pinv(Nj.T)

    Px = pseudoNi@Xint@pseudoNj
    Py = pseudoNi@Yint@pseudoNj
    Pz = pseudoNi@Zint@pseudoNj

    Px = np.reshape(Px, (1, control_points_shape[0], control_points_shape[1]), order='F')
    Py = np.reshape(Py, (1, control_points_shape[0], control_points_shape[1]), order='F')
    Pz = np.reshape(Pz, (1, control_points_shape[0], control_points_shape[1]), order='F')
    control_points = np.concatenate((Px, Py, Pz), axis = 0)


    return control_points, knotsU, knotsV, uk, vk

class Nurbs:
    def __init__(self, knotsU = None, knotsV= None, degU = 3, degV = 3, control_points = None) -> None:
        self.controlPoints = control_points
        self.knotsU = knotsU
        self.knotsV = knotsV
        self.degreeU = degU
        self.degreeV = degV
        self.orderU = []
        self.orderV = []
        self.fit_residuals = []
        self.casadi_eval = []
        self.casadi_normal = []
        self.casadi_Ni = []
        self.casadi_Nj = []
        self.curvature_gaussian = []
        self.curvature_mean = []
        self.curvature_max = []
        self.curvature_min = []


        if control_points is not None and knotsU is not None and knotsV is not None: # check that the basic input data should not be None
            if control_points and knotsU and knotsV:                                 # check that the basic input data are not empty lists
                self.__init_casadi_functions()

        return

    def eval(self, u_values, v_values):
        """
        returns the Nurbs evaluated at the sampled values in a dictionary S = {'x':x_value, 'y':y_value, 'z':z_value}
        """
        if not isinstance(u_values, list):
            u_values = [u_values]
        if not isinstance(u_values, np.ndarray):
            u_values = np.array(u_values)
        if not isinstance(v_values, list):
            v_values = [v_values]
        if not isinstance(v_values, np.ndarray):
            v_values = np.array(v_values)

        shp = u_values.shape; n = shp[0];
        try:
            m = shp[1]
        except:
            m = 1

        # if u_values.shape != v_values.shape:
        #     raise TypeError("u and v arryas must have the same size (shape)")
        P = self.controlPoints
        U = self.knotsU
        V = self.knotsV
        pU = self.degreeU
        pV = self.degreeV

        u = u_values.flatten() # order = 'F' to match matlab
        v = v_values.flatten() # order = 'F' to match matlab

        if u.size == 1:
            u = u*np.ones(v.shape)
        if v.size == 1:
            v = v*np.ones(u.shape)

        Ni = np.nan*np.zeros((u.size, U.size - pU - 1))
        Nj = np.nan*np.zeros((v.size, V.size - pV - 1))
        for ii in range(0, u.size):
            Ni[ii, :] = basis_function(u[ii], pU, U)
            Nj[ii, :] = basis_function(v[ii], pV, V)

        X = P[0,:,:]
        Y = P[1,:,:]
        Z = P[2,:,:]

        S = {'x': np.reshape(np.sum(Ni.T*(X @ Nj.T), axis = 0), (n, m)).squeeze(),
             'y': np.reshape(np.sum(Ni.T*(Y @ Nj.T), axis = 0), (n, m)).squeeze(),
             'z': np.reshape(np.sum(Ni.T*(Z @ Nj.T), axis = 0), (n, m)).squeeze()}

        return S

    def fit(self, target_points, degU, degV, control_points_shape, method='least_squares', **kwargs):
        """
        Fit NURBS surface to target points with multiple fitting algorithms
        
        Parameters:
        -----------
        target_points : dict or np.ndarray
            Target points to fit. If dict: {'X': x_coords, 'Y': y_coords, 'Z': z_coords}
            If array: shape (3, n_u, n_v) with [x, y, z] coordinates
        degU, degV : int
            Degrees in U and V directions
        control_points_shape : tuple
            Shape of control points grid (n_u, n_v)
        method : str, optional
            Fitting method: 'least_squares', 'regularized', 'weighted', 'iterative'
        **kwargs : dict
            Additional parameters for specific fitting methods
            
        Returns:
        --------
        dict : Fitting results with residuals and statistics
        """
        if isinstance(target_points, dict):
            grid_size = target_points['X'].shape
            points = np.zeros((3, grid_size[0], grid_size[1]))
            points[0, :, :] = target_points['X']
            points[1, :, :] = target_points['Y']
            points[2, :, :] = target_points['Z']
            target_points = points

        self.degreeU = degU
        self.degreeV = degV
        self.orderU = degU + 1
        self.orderV = degV + 1
        
        # Store original target points
        self.target_points = target_points.copy()
        
        # Choose fitting method
        if method == 'least_squares':
            result = self._fit_least_squares(target_points, control_points_shape)
        elif method == 'regularized':
            result = self._fit_regularized(target_points, control_points_shape, **kwargs)
        elif method == 'weighted':
            result = self._fit_weighted(target_points, control_points_shape, **kwargs)
        elif method == 'iterative':
            result = self._fit_iterative(target_points, control_points_shape, **kwargs)
        else:
            raise ValueError(f"Unknown fitting method: {method}")
        
        # Initialize CasADi functions and compute residuals
        self._compute_fitting_residuals()
        
        return result

    def _fit_least_squares(self, target_points, control_points_shape):
        """
        Standard least squares NURBS fitting
        """
        self.controlPoints, self.knotsU, self.knotsV, uk, vk = fit_nurbs_surface(
            target_points, self.degreeU, self.degreeV, control_points_shape
        )
        
        return {
            'method': 'least_squares',
            'success': True,
            'control_points_shape': control_points_shape
        }
    
    def _fit_regularized(self, target_points, control_points_shape, lambda_smooth=1e-6, 
                        lambda_fair=1e-6, max_iterations=100):
        """
        Regularized NURBS fitting with smoothness and fairness constraints
        """
        # Start with least squares solution
        self._fit_least_squares(target_points, control_points_shape)
        
        # Get initial parameters
        shp = target_points.shape
        uk, vk, _, _ = chord_length_nodes(self.degreeU, self.degreeV, target_points)
        
        # Iteratively refine with regularization
        prev_error = np.inf
        
        for iteration in range(max_iterations):
            # Compute basis functions
            Ni = np.zeros((uk.shape[0], self.knotsU.shape[0] - self.degreeU - 1))
            for kk in range(uk.shape[0]):
                Ni[kk, :] = basis_function(uk[kk], self.degreeU, self.knotsU)
                
            Nj = np.zeros((vk.shape[0], self.knotsV.shape[0] - self.degreeV - 1))
            for kk in range(vk.shape[0]):
                Nj[kk, :] = basis_function(vk[kk], self.degreeV, self.knotsV)
            
            # Build regularization matrices
            n_ctrl_u, n_ctrl_v = control_points_shape
            
            # Smoothness regularization (second derivatives)
            R_smooth_u = self._build_smoothness_matrix(n_ctrl_u)
            R_smooth_v = self._build_smoothness_matrix(n_ctrl_v)
            
            # Fairness regularization (minimize curvature variation)
            R_fair_u = self._build_fairness_matrix(n_ctrl_u)
            R_fair_v = self._build_fairness_matrix(n_ctrl_v)
            
            # Solve regularized system for each coordinate
            for coord in range(3):
                target_coord = target_points[coord, :, :]
                
                # Build augmented system
                A = Ni.T @ target_coord @ np.linalg.pinv(Nj.T)
                
                # Add regularization terms
                if lambda_smooth > 0:
                    A += lambda_smooth * (R_smooth_u @ self.controlPoints[coord, :, :] + 
                                         self.controlPoints[coord, :, :] @ R_smooth_v.T)
                
                if lambda_fair > 0:
                    A += lambda_fair * (R_fair_u @ self.controlPoints[coord, :, :] + 
                                      self.controlPoints[coord, :, :] @ R_fair_v.T)
                
                self.controlPoints[coord, :, :] = A
            
            # Check convergence
            self._compute_fitting_residuals()
            current_error = np.mean(np.abs(self.fit_residuals))
            
            if abs(prev_error - current_error) < 1e-8:
                break
                
            prev_error = current_error
        
        return {
            'method': 'regularized',
            'success': True,
            'iterations': iteration + 1,
            'final_error': current_error,
            'lambda_smooth': lambda_smooth,
            'lambda_fair': lambda_fair
        }
    
    def _fit_weighted(self, target_points, control_points_shape, weights=None):
        """
        Weighted least squares NURBS fitting
        """
        shp = target_points.shape
        
        if weights is None:
            # Default: uniform weights
            weights = np.ones((shp[1], shp[2]))
        
        # Modify the fitting process to include weights
        uk, vk, _, _ = chord_length_nodes(self.degreeU, self.degreeV, target_points)
        knotsU = fit_knots(uk, control_points_shape[0], self.degreeU)
        knotsV = fit_knots(vk, control_points_shape[1], self.degreeV)
        
        self.knotsU = knotsU
        self.knotsV = knotsV
        
        # Weighted basis functions
        Ni = np.zeros((uk.shape[0], knotsU.shape[0] - self.degreeU - 1))
        for kk in range(uk.shape[0]):
            Ni[kk, :] = basis_function(uk[kk], self.degreeU, knotsU)
            
        Nj = np.zeros((vk.shape[0], knotsV.shape[0] - self.degreeV - 1))
        for kk in range(vk.shape[0]):
            Nj[kk, :] = basis_function(vk[kk], self.degreeV, knotsV)
        
        # Apply weights
        W = np.sqrt(weights)
        
        # Weighted pseudo-inverse
        Ni_weighted = np.diag(W.flatten()) @ Ni
        Nj_weighted = np.diag(W.flatten()) @ Nj
        
        pseudoNi = np.linalg.pinv(Ni_weighted)
        pseudoNj = np.linalg.pinv(Nj_weighted.T)
        
        # Solve for control points
        control_points = np.zeros((3, control_points_shape[0], control_points_shape[1]))
        
        for coord in range(3):
            target_weighted = W * target_points[coord, :, :]
            control_points[coord, :, :] = pseudoNi @ target_weighted @ pseudoNj
        
        self.controlPoints = control_points
        
        return {
            'method': 'weighted',
            'success': True,
            'weights_used': True
        }
    
    def _fit_iterative(self, target_points, control_points_shape, max_iterations=50, 
                      tolerance=1e-6, adaptive_refinement=True):
        """
        Iterative NURBS fitting with adaptive refinement
        """
        # Start with coarser control points if adaptive refinement is enabled
        if adaptive_refinement:
            current_shape = (max(4, control_points_shape[0] // 2), 
                           max(4, control_points_shape[1] // 2))
        else:
            current_shape = control_points_shape
        
        iteration_results = []
        
        for iteration in range(max_iterations):
            # Fit with current control points shape
            self._fit_least_squares(target_points, current_shape)
            self._compute_fitting_residuals()
            
            current_error = np.mean(np.abs(self.fit_residuals))
            max_error = np.max(np.abs(self.fit_residuals))
            
            iteration_results.append({
                'iteration': iteration,
                'control_points_shape': current_shape,
                'mean_error': current_error,
                'max_error': max_error
            })
            
            # Check convergence
            if current_error < tolerance:
                break
            
            # Adaptive refinement: increase control points if error is too large
            if adaptive_refinement and iteration < max_iterations - 1:
                if current_error > tolerance * 10 and current_shape[0] < control_points_shape[0]:
                    current_shape = (min(current_shape[0] + 2, control_points_shape[0]),
                                   min(current_shape[1] + 2, control_points_shape[1]))
        
        return {
            'method': 'iterative',
            'success': current_error < tolerance,
            'iterations': len(iteration_results),
            'final_error': current_error,
            'iteration_history': iteration_results,
            'adaptive_refinement': adaptive_refinement
        }
    
    def _build_smoothness_matrix(self, n):
        """
        Build smoothness regularization matrix (second differences)
        """
        if n < 3:
            return np.zeros((n, n))
        
        R = np.zeros((n, n))
        for i in range(1, n - 1):
            R[i, i-1] = 1
            R[i, i] = -2
            R[i, i+1] = 1
        
        return R.T @ R
    
    def _build_fairness_matrix(self, n):
        """
        Build fairness regularization matrix (third differences)
        """
        if n < 4:
            return np.zeros((n, n))
        
        R = np.zeros((n, n))
        for i in range(1, n - 2):
            R[i, i-1] = -1
            R[i, i] = 3
            R[i, i+1] = -3
            R[i, i+2] = 1
        
        return R.T @ R
    
    def _compute_fitting_residuals(self):
        """
        Compute fitting residuals and initialize CasADi functions
        """
        if not hasattr(self, 'target_points'):
            return
            
        target_points = self.target_points
        shp = target_points.shape
        
        # Get parameter values used in fitting
        uk, vk, _, _ = chord_length_nodes(self.degreeU, self.degreeV, target_points)
        n_u = uk.shape[0]
        n_v = vk.shape[0]

        self.__init_casadi_functions()
        uk_mesh, vk_mesh = np.meshgrid(uk, vk)
        setattr(self, 'u_fit', uk_mesh.T)
        setattr(self, 'v_fit', vk_mesh.T)
        
        uk_flat = uk_mesh.flatten()
        vk_flat = vk_mesh.flatten()
        
        # Evaluate NURBS surface and normals
        normals = self.casadi_normal(np.expand_dims(uk_flat, axis=0), np.expand_dims(vk_flat, axis=0))
        points = self.casadi_eval(np.expand_dims(uk_flat, axis=0), np.expand_dims(vk_flat, axis=0))

        nX = normals[0, :].full().reshape((1, n_u, n_v), order='F')
        nY = normals[1, :].full().reshape((1, n_u, n_v), order='F')
        nZ = normals[2, :].full().reshape((1, n_u, n_v), order='F')
        X = points[0, :].full().reshape((1, n_u, n_v), order='F')
        Y = points[1, :].full().reshape((1, n_u, n_v), order='F')
        Z = points[2, :].full().reshape((1, n_u, n_v), order='F')

        evaluated_points = np.concatenate((X, Y, Z), axis=0)
        evaluated_normals = np.concatenate((nX, nY, nZ), axis=0)

        self.fit_residuals = np.sum((target_points - evaluated_points) * evaluated_normals, axis=0)
        
        # Store additional fitting statistics
        self.fitting_stats = {
            'mean_error': np.mean(np.abs(self.fit_residuals)),
            'max_error': np.max(np.abs(self.fit_residuals)),
            'rms_error': np.sqrt(np.mean(self.fit_residuals**2)),
            'std_error': np.std(self.fit_residuals)
        }

    def plot(self, nU=150, nV=150, show_normals=False, show_residuals=False, **kwargs):
        """
        Enhanced plotting with multiple visualization options
        """
        u_values = np.linspace(0, 1, nU)
        v_values = np.linspace(0, 1, nV)
        X = np.nan * np.zeros((nU, nV))
        Y = np.nan * np.zeros((nU, nV))
        Z = np.nan * np.zeros((nU, nV))

        for ii in range(0, nU):
            for jj in range(0, nV):
                S = self.eval(u_values[ii], v_values[jj])
                X[ii, jj] = S['x']
                Y[ii, jj] = S['y']
                Z[ii, jj] = S['z']

        grid = pv.StructuredGrid(X, Y, Z)
        plotter = pv.Plotter()
        
        # Default mesh options
        mesh_opts = {
            'show_edges': kwargs.get('show_edges', True),
            'edge_color': kwargs.get('edge_color', 'black'),
            'color': kwargs.get('color', 'white')
        }
        
        # Add residuals as scalar data if available and requested
        if show_residuals and hasattr(self, 'fit_residuals'):
            # Interpolate residuals to plot grid
            from scipy.interpolate import griddata
            
            u_fit_flat = self.u_fit.flatten()
            v_fit_flat = self.v_fit.flatten()
            residuals_flat = self.fit_residuals.flatten()
            
            u_plot, v_plot = np.meshgrid(u_values, v_values)
            residuals_interp = griddata(
                (u_fit_flat, v_fit_flat), residuals_flat,
                (u_plot.flatten(), v_plot.flatten()), method='linear'
            ).reshape(nU, nV)
            
            grid['Residuals'] = residuals_interp.T.flatten()
            mesh_opts.update({
                'scalars': 'Residuals',
                'cmap': 'RdBu',
                'scalar_bar_args': {
                    "title": "Fitting Residuals",
                    "vertical": True,
                    "title_font_size": 10,
                    "label_font_size": 8,
                }
            })
        
        plotter.add_mesh(grid, **mesh_opts)
        
        # Add normals if requested
        if show_normals:
            # Sample normals at fewer points for clarity
            n_normal_samples = 20
            u_normal = np.linspace(0.1, 0.9, n_normal_samples)
            v_normal = np.linspace(0.1, 0.9, n_normal_samples)
            
            normal_points = []
            normal_vectors = []
            
            for u in u_normal:
                for v in v_normal:
                    try:
                        point = self.casadi_eval(u, v)
                        normal = self.casadi_normal(u, v)
                        
                        normal_points.append([float(point[0]), float(point[1]), float(point[2])])
                        normal_vectors.append([float(normal[0]), float(normal[1]), float(normal[2])])
                    except:
                        continue
            
            if normal_points:
                normal_points = np.array(normal_points)
                normal_vectors = np.array(normal_vectors) * 0.1  # Scale for visibility
                
                plotter.add_arrows(normal_points, normal_vectors, color='red', mag=1.0)
        
        plotter.show_axes()
        plotter.show_grid()
        plotter.show()
        
        return grid
    
    def refine_surface(self, target_error=1e-6, max_control_points=(100, 100), method='adaptive'):
        """
        Refine NURBS surface to achieve target fitting error
        
        Parameters:
        -----------
        target_error : float
            Target RMS fitting error
        max_control_points : tuple
            Maximum number of control points (n_u, n_v)
        method : str
            Refinement method: 'adaptive', 'uniform', 'error_based'
        
        Returns:
        --------
        dict : Refinement results
        """
        if not hasattr(self, 'target_points'):
            raise ValueError("No target points available. Run fit() first.")
        
        current_shape = (self.controlPoints.shape[1], self.controlPoints.shape[2])
        original_error = self.fitting_stats['rms_error']
        
        refinement_history = [{
            'control_points': current_shape,
            'rms_error': original_error
        }]
        
        iteration = 0
        max_iterations = 10
        
        while (original_error > target_error and 
               current_shape[0] < max_control_points[0] and 
               current_shape[1] < max_control_points[1] and 
               iteration < max_iterations):
            
            if method == 'adaptive':
                # Increase control points where error is highest
                new_shape = self._adaptive_refinement(current_shape, max_control_points)
            elif method == 'uniform':
                # Uniform increase
                new_shape = (min(current_shape[0] + 2, max_control_points[0]),
                           min(current_shape[1] + 2, max_control_points[1]))
            elif method == 'error_based':
                # Refine based on local error distribution
                new_shape = self._error_based_refinement(current_shape, max_control_points)
            else:
                raise ValueError(f"Unknown refinement method: {method}")
            
            # Refit with new control points
            result = self.fit(self.target_points, self.degreeU, self.degreeV, 
                            new_shape, method='least_squares')
            
            current_error = self.fitting_stats['rms_error']
            current_shape = new_shape
            
            refinement_history.append({
                'control_points': current_shape,
                'rms_error': current_error
            })
            
            # Check for convergence
            if abs(original_error - current_error) < target_error * 0.1:
                break
            
            original_error = current_error
            iteration += 1
        
        return {
            'success': original_error <= target_error,
            'final_error': original_error,
            'final_control_points': current_shape,
            'iterations': iteration,
            'refinement_history': refinement_history,
            'method': method
        }
    
    def _adaptive_refinement(self, current_shape, max_shape):
        """
        Adaptive refinement based on error distribution
        """
        # Analyze error distribution to determine where to add control points
        residuals = np.abs(self.fit_residuals)
        
        # Find regions with high error
        error_threshold = np.percentile(residuals.flatten(), 75)
        high_error_mask = residuals > error_threshold
        
        # Compute error in U and V directions
        u_error = np.mean(residuals, axis=1)
        v_error = np.mean(residuals, axis=0)
        
        # Decide refinement direction
        u_needs_refinement = np.std(u_error) > np.std(v_error)
        v_needs_refinement = np.std(v_error) > np.std(u_error)
        
        new_u = current_shape[0]
        new_v = current_shape[1]
        
        if u_needs_refinement and new_u < max_shape[0]:
            new_u = min(current_shape[0] + 2, max_shape[0])
        
        if v_needs_refinement and new_v < max_shape[1]:
            new_v = min(current_shape[1] + 2, max_shape[1])
        
        # If no refinement needed in specific direction, refine uniformly
        if new_u == current_shape[0] and new_v == current_shape[1]:
            new_u = min(current_shape[0] + 1, max_shape[0])
            new_v = min(current_shape[1] + 1, max_shape[1])
        
        return (new_u, new_v)
    
    def _error_based_refinement(self, current_shape, max_shape):
        """
        Error-based refinement using local error analysis
        """
        # For now, implement as adaptive refinement
        # In a more sophisticated implementation, this would analyze
        # local curvature and error gradients
        return self._adaptive_refinement(current_shape, max_shape)
    
    def optimize_control_points(self, method='scipy', **kwargs):
        """
        Optimize control points positions for better fitting
        
        Parameters:
        -----------
        method : str
            Optimization method: 'scipy', 'gradient_descent', 'casadi'
        **kwargs : dict
            Method-specific parameters
        
        Returns:
        --------
        dict : Optimization results
        """
        if not hasattr(self, 'target_points'):
            raise ValueError("No target points available. Run fit() first.")
        
        if method == 'scipy':
            return self._optimize_scipy(**kwargs)
        elif method == 'gradient_descent':
            return self._optimize_gradient_descent(**kwargs)
        elif method == 'casadi':
            return self._optimize_casadi(**kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_scipy(self, max_iterations=1000, tolerance=1e-8):
        """
        Scipy-based optimization of control points
        """
        from scipy.optimize import minimize
        
        # Flatten control points for optimization
        x0 = self.controlPoints.flatten()
        
        def objective(x):
            # Reshape and set control points
            shape = self.controlPoints.shape
            temp_control_points = x.reshape(shape)
            
            # Temporarily set control points
            original_cp = self.controlPoints.copy()
            self.controlPoints = temp_control_points
            
            # Compute residuals
            self._compute_fitting_residuals()
            error = np.sum(self.fit_residuals**2)
            
            # Restore original control points
            self.controlPoints = original_cp
            
            return error
        
        # Run optimization
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iterations, 'ftol': tolerance})
        
        if result.success:
            # Set optimized control points
            self.controlPoints = result.x.reshape(self.controlPoints.shape)
            self._compute_fitting_residuals()
        
        return {
            'success': result.success,
            'final_error': result.fun,
            'iterations': result.nit,
            'message': result.message,
            'method': 'scipy'
        }
    
    def _optimize_gradient_descent(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-8):
        """
        Simple gradient descent optimization
        """
        original_error = np.sum(self.fit_residuals**2)
        
        for iteration in range(max_iterations):
            # Compute numerical gradients
            gradients = self._compute_numerical_gradients()
            
            # Update control points
            self.controlPoints -= learning_rate * gradients
            
            # Compute new error
            self._compute_fitting_residuals()
            new_error = np.sum(self.fit_residuals**2)
            
            # Check convergence
            if abs(original_error - new_error) < tolerance:
                break
            
            original_error = new_error
        
        return {
            'success': True,
            'final_error': new_error,
            'iterations': iteration + 1,
            'method': 'gradient_descent'
        }
    
    def _optimize_casadi(self, **kwargs):
        """
        CasADi-based optimization (placeholder for future implementation)
        """
        print("CasADi optimization not yet implemented")
        return {'success': False, 'method': 'casadi'}
    
    def _compute_numerical_gradients(self, eps=1e-6):
        """
        Compute numerical gradients of the fitting error
        """
        gradients = np.zeros_like(self.controlPoints)
        original_error = np.sum(self.fit_residuals**2)
        
        for i in range(self.controlPoints.shape[0]):
            for j in range(self.controlPoints.shape[1]):
                for k in range(self.controlPoints.shape[2]):
                    # Perturb control point
                    self.controlPoints[i, j, k] += eps
                    self._compute_fitting_residuals()
                    perturbed_error = np.sum(self.fit_residuals**2)
                    
                    # Compute gradient
                    gradients[i, j, k] = (perturbed_error - original_error) / eps
                    
                    # Restore original value
                    self.controlPoints[i, j, k] -= eps
        
        return gradients
    
    def compare_fits(self, other_nurbs, plot=True):
        """
        Compare this NURBS surface with another fitted surface
        
        Parameters:
        -----------
        other_nurbs : Nurbs
            Another Nurbs object to compare with
        plot : bool
            Whether to create comparison plots
        
        Returns:
        --------
        dict : Comparison results
        """
        if not hasattr(self, 'fitting_stats') or not hasattr(other_nurbs, 'fitting_stats'):
            raise ValueError("Both NURBS objects must have fitting statistics")
        
        comparison = {
            'surface_1': {
                'degrees': (self.degreeU, self.degreeV),
                'control_points': (self.controlPoints.shape[1], self.controlPoints.shape[2]),
                'stats': self.fitting_stats.copy()
            },
            'surface_2': {
                'degrees': (other_nurbs.degreeU, other_nurbs.degreeV),
                'control_points': (other_nurbs.controlPoints.shape[1], other_nurbs.controlPoints.shape[2]),
                'stats': other_nurbs.fitting_stats.copy()
            },
            'comparison': {
                'rms_improvement': self.fitting_stats['rms_error'] - other_nurbs.fitting_stats['rms_error'],
                'max_improvement': self.fitting_stats['max_error'] - other_nurbs.fitting_stats['max_error'],
                'better_fit': self.fitting_stats['rms_error'] < other_nurbs.fitting_stats['rms_error']
            }
        }
        
        if plot:
            self._plot_comparison(other_nurbs, comparison)
        
        return comparison
    
    def _plot_comparison(self, other_nurbs, comparison):
        """
        Create comparison plots between two NURBS surfaces
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot residuals for both surfaces
        im1 = axes[0, 0].contourf(self.u_fit, self.v_fit, self.fit_residuals * 1000, 
                                 levels=20, cmap='RdBu')
        axes[0, 0].set_title('Surface 1 - Residuals')
        axes[0, 0].set_xlabel('u')
        axes[0, 0].set_ylabel('v')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[1, 0].contourf(other_nurbs.u_fit, other_nurbs.v_fit, 
                                 other_nurbs.fit_residuals * 1000, levels=20, cmap='RdBu')
        axes[1, 0].set_title('Surface 2 - Residuals')
        axes[1, 0].set_xlabel('u')
        axes[1, 0].set_ylabel('v')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Histograms
        axes[0, 1].hist(self.fit_residuals.flatten() * 1000, bins=50, alpha=0.7, label='Surface 1')
        axes[0, 1].set_title('Residual Distribution - Surface 1')
        axes[0, 1].set_xlabel('Residuals (μm)')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[1, 1].hist(other_nurbs.fit_residuals.flatten() * 1000, bins=50, alpha=0.7, label='Surface 2')
        axes[1, 1].set_title('Residual Distribution - Surface 2')
        axes[1, 1].set_xlabel('Residuals (μm)')
        axes[1, 1].set_ylabel('Frequency')
        
        # Statistics comparison
        stats1 = comparison['surface_1']['stats']
        stats2 = comparison['surface_2']['stats']
        
        stat_names = list(stats1.keys())
        stat_values1 = [stats1[name] * 1000 for name in stat_names]  # Convert to μm
        stat_values2 = [stats2[name] * 1000 for name in stat_names]
        
        x_pos = np.arange(len(stat_names))
        width = 0.35
        
        axes[0, 2].bar(x_pos - width/2, stat_values1, width, label='Surface 1')
        axes[0, 2].bar(x_pos + width/2, stat_values2, width, label='Surface 2')
        axes[0, 2].set_title('Statistics Comparison')
        axes[0, 2].set_ylabel('Error (μm)')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels([name.replace('_', '\n') for name in stat_names], rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Summary text
        summary_text = f"""Comparison Summary:
        
Surface 1:
  Degrees: {comparison['surface_1']['degrees']}
  Control Points: {comparison['surface_1']['control_points']}
  RMS Error: {stats1['rms_error']*1000:.3f} μm
  
Surface 2:
  Degrees: {comparison['surface_2']['degrees']}
  Control Points: {comparison['surface_2']['control_points']}
  RMS Error: {stats2['rms_error']*1000:.3f} μm
  
Improvement:
  RMS: {comparison['comparison']['rms_improvement']*1000:.3f} μm
  Better Fit: {comparison['comparison']['better_fit']}
        """
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show(block=False)

    def analyze_fitting_quality(self, verbose=True):
        """
        Comprehensive analysis of fitting quality
        
        Returns:
        --------
        dict : Analysis results with various quality metrics
        """
        if not hasattr(self, 'fit_residuals'):
            raise ValueError("No fitting residuals available. Run fit() first.")
        
        residuals = self.fit_residuals.flatten()
        
        analysis = {
            'statistical_measures': {
                'mean_error': np.mean(np.abs(residuals)),
                'max_error': np.max(np.abs(residuals)),
                'min_error': np.min(np.abs(residuals)),
                'rms_error': np.sqrt(np.mean(residuals**2)),
                'std_error': np.std(residuals),
                'percentile_95': np.percentile(np.abs(residuals), 95),
                'percentile_99': np.percentile(np.abs(residuals), 99)
            },
            'distribution_analysis': {
                'skewness': self._compute_skewness(residuals),
                'kurtosis': self._compute_kurtosis(residuals),
                'is_normal': self._test_normality(residuals)
            },
            'surface_properties': self._analyze_surface_properties()
        }
        
        if verbose:
            self._print_analysis_report(analysis)
        
        return analysis
    
    def _compute_skewness(self, data):
        """
        Compute skewness of residual distribution
        """
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data):
        """
        Compute kurtosis of residual distribution
        """
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _test_normality(self, data, alpha=0.05):
        """
        Simple normality test using skewness and kurtosis
        """
        skew = self._compute_skewness(data)
        kurt = self._compute_kurtosis(data)
        
        # Simple test: if both skewness and kurtosis are close to normal distribution values
        return abs(skew) < 0.5 and abs(kurt) < 0.5
    
    def _analyze_surface_properties(self):
        """
        Analyze geometric properties of the fitted surface
        """
        if not hasattr(self, 'casadi_eval'):
            return {}
        
        # Sample surface at regular intervals
        u_samples = np.linspace(0.1, 0.9, 20)
        v_samples = np.linspace(0.1, 0.9, 20)
        
        curvatures_gauss = []
        curvatures_mean = []
        
        for u in u_samples:
            for v in v_samples:
                try:
                    k_gauss = float(self.curvature_gaussian(u, v))
                    k_mean = float(self.curvature_mean(u, v))
                    
                    if np.isfinite(k_gauss) and np.isfinite(k_mean):
                        curvatures_gauss.append(k_gauss)
                        curvatures_mean.append(k_mean)
                except:
                    continue
        
        if curvatures_gauss:
            return {
                'gaussian_curvature': {
                    'mean': np.mean(curvatures_gauss),
                    'std': np.std(curvatures_gauss),
                    'range': [np.min(curvatures_gauss), np.max(curvatures_gauss)]
                },
                'mean_curvature': {
                    'mean': np.mean(curvatures_mean),
                    'std': np.std(curvatures_mean),
                    'range': [np.min(curvatures_mean), np.max(curvatures_mean)]
                }
            }
        
        return {}
    
    def _print_analysis_report(self, analysis):
        """
        Print formatted analysis report
        """
        print("\n=== NURBS Surface Fitting Analysis ===")
        
        stats = analysis['statistical_measures']
        print(f"\nStatistical Measures:")
        print(f"  Mean Error: {stats['mean_error']:.6f}")
        print(f"  RMS Error:  {stats['rms_error']:.6f}")
        print(f"  Max Error:  {stats['max_error']:.6f}")
        print(f"  Std Error:  {stats['std_error']:.6f}")
        print(f"  95th percentile: {stats['percentile_95']:.6f}")
        print(f"  99th percentile: {stats['percentile_99']:.6f}")
        
        dist = analysis['distribution_analysis']
        print(f"\nDistribution Analysis:")
        print(f"  Skewness: {dist['skewness']:.4f}")
        print(f"  Kurtosis: {dist['kurtosis']:.4f}")
        print(f"  Normal distribution: {dist['is_normal']}")
        
        if 'surface_properties' in analysis and analysis['surface_properties']:
            surf = analysis['surface_properties']
            print(f"\nSurface Properties:")
            if 'gaussian_curvature' in surf:
                gc = surf['gaussian_curvature']
                print(f"  Gaussian curvature - Mean: {gc['mean']:.6f}, Std: {gc['std']:.6f}")
            if 'mean_curvature' in surf:
                mc = surf['mean_curvature']
                print(f"  Mean curvature - Mean: {mc['mean']:.6f}, Std: {mc['std']:.6f}")
    
    def plot_residuals(self, colormap='RdBu', show_statistics=True):
        """
        Enhanced residual plotting with statistics
        """
        if not hasattr(self, 'fit_residuals'):
            raise ValueError("No fitting residuals available. Run fit() first.")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5))
        
        # 3D surface plot of residuals
        ax1 = fig.add_subplot(131, projection='3d')
        surf = ax1.plot_surface(self.u_fit, self.v_fit, self.fit_residuals * 1000, 
                               cmap=colormap, edgecolor='none', alpha=0.9)
        ax1.set_xlabel('u (-)')
        ax1.set_ylabel('v (-)')
        ax1.set_zlabel('Residuals (μm)')
        ax1.set_title('3D Residual Surface')
        plt.colorbar(surf, ax=ax1, shrink=0.5)
        
        # 2D contour plot
        ax2 = fig.add_subplot(132)
        contour = ax2.contourf(self.u_fit, self.v_fit, self.fit_residuals * 1000, 
                              levels=20, cmap=colormap)
        ax2.set_xlabel('u (-)')
        ax2.set_ylabel('v (-)')
        ax2.set_title('Residual Contours')
        plt.colorbar(contour, ax=ax2)
        
        # Histogram of residuals
        ax3 = fig.add_subplot(133)
        residuals_flat = self.fit_residuals.flatten() * 1000
        ax3.hist(residuals_flat, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(residuals_flat), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(residuals_flat):.3f} μm')
        ax3.axvline(np.mean(residuals_flat) + np.std(residuals_flat), color='orange', 
                   linestyle='--', label=f'+1σ: {np.std(residuals_flat):.3f} μm')
        ax3.axvline(np.mean(residuals_flat) - np.std(residuals_flat), color='orange', 
                   linestyle='--', label=f'-1σ')
        ax3.set_xlabel('Residuals (μm)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residual Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        if show_statistics and hasattr(self, 'fitting_stats'):
            stats_text = f"RMS: {self.fitting_stats['rms_error']*1000:.3f} μm\n"
            stats_text += f"Max: {self.fitting_stats['max_error']*1000:.3f} μm"
            ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show(block=False)
    
    def export_fitting_report(self, filename=None):
        """
        Export comprehensive fitting report to file
        """
        if filename is None:
            filename = 'nurbs_fitting_report.txt'
        
        analysis = self.analyze_fitting_quality(verbose=False)
        
        with open(filename, 'w') as f:
            f.write("NURBS Surface Fitting Report\n")
            f.write("===========================\n\n")
            
            # Surface parameters
            f.write(f"Surface Parameters:\n")
            f.write(f"  Degree U: {self.degreeU}\n")
            f.write(f"  Degree V: {self.degreeV}\n")
            f.write(f"  Control Points: {self.controlPoints.shape[1]} x {self.controlPoints.shape[2]}\n")
            f.write(f"  Knots U: {len(self.knotsU)}\n")
            f.write(f"  Knots V: {len(self.knotsV)}\n\n")
            
            # Statistical measures
            stats = analysis['statistical_measures']
            f.write(f"Fitting Statistics:\n")
            for key, value in stats.items():
                f.write(f"  {key.replace('_', ' ').title()}: {value:.8f}\n")
            
            f.write(f"\nDistribution Analysis:\n")
            dist = analysis['distribution_analysis']
            for key, value in dist.items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
            
            if 'surface_properties' in analysis and analysis['surface_properties']:
                f.write(f"\nSurface Properties:\n")
                surf = analysis['surface_properties']
                for prop_name, prop_data in surf.items():
                    f.write(f"  {prop_name.replace('_', ' ').title()}:\n")
                    for key, value in prop_data.items():
                        if isinstance(value, list):
                            f.write(f"    {key}: [{value[0]:.6f}, {value[1]:.6f}]\n")
                        else:
                            f.write(f"    {key}: {value:.6f}\n")
        
        print(f"Fitting report exported to: {filename}")
        return filename

    def __init_casadi_functions(self):

        pU = self.degreeU
        pV = self.degreeV
        U = self.knotsU
        V = self.knotsV
        P = self.controlPoints
        X = P[0,:,:]
        Y = P[1,:,:]
        Z = P[2,:,:]

        u = ca.SX.sym('u')
        v = ca.SX.sym('v')
        Ni = basis_fun_casadi(pU, U, 2); Ni = Ni(u)
        Nj = basis_fun_casadi(pV, V, 2); Nj = Nj(v)
        self.casadi_Ni = ca.Function('Ni', [u], [Ni])
        self.casadi_Nj = ca.Function('Nj', [v], [Nj])

        Ni_0 = Ni[:, 0]
        Ni_1 = Ni[:, 1]
        Ni_2 = Ni[:, 2]

        Nj_0 = Nj[:, 0]
        Nj_1 = Nj[:, 1]
        Nj_2 = Nj[:, 2]

        S = ca.vertcat(Ni_0.T@X@Nj_0, Ni_0.T@Y@Nj_0, Ni_0.T@Z@Nj_0)
        Su = ca.vertcat(Ni_1.T@X@Nj_0, Ni_1.T@Y@Nj_0, Ni_1.T@Z@Nj_0)
        Suu = ca.vertcat(Ni_2.T@X@Nj_0, Ni_2.T@Y@Nj_0, Ni_2.T@Z@Nj_0)
        Sv = ca.vertcat(Ni_0.T@X@Nj_1, Ni_0.T@Y@Nj_1, Ni_0.T@Z@Nj_1)
        Svv = ca.vertcat(Ni_0.T@X@Nj_2, Ni_0.T@Y@Nj_2, Ni_0.T@Z@Nj_2)
        Suv = ca.vertcat(Ni_1.T@X@Nj_1, Ni_1.T@Y@Nj_1, Ni_1.T@Z@Nj_1)
        N = ca.cross(Su, Sv)
        n = N/ca.norm_2(N)
        E = Su.T@Su
        F = Su.T@Sv
        G = Sv.T@Sv
        e = Suu.T@N
        f = Suv.T@N
        g = Svv.T@N
        K = (e*g - f**2) / (E*G - F**2) # Gaussian curvature
        H = (g*E - 2*f*F + e*G)/2/(E*G - F**2) # Mean curvature
        K1 = H + ca.sqrt(H**2 - K) # max principal curvature
        K2 = K/K1                  # min principal curvature
        self.casadi_eval = ca.Function('S', [u, v], [S])
        self.casadi_normal = ca.Function('N', [u, v], [n])
        self.curvature_gaussian = ca.Function('G', [u, v], [K])
        self.curvature_mean = ca.Function('H', [u, v], [H])
        self.curvature_max = ca.Function('Kmax', [u, v], [K1])
        self.curvature_min = ca.Function('Kmin', [u, v], [K2])

        return

def demo_advanced_fitting():
    """
    Comprehensive demo of advanced NURBS surface fitting capabilities
    """
    print("=== Advanced NURBS Surface Fitting Demo ===")
    
    # Generate synthetic test surface data
    print("\n1. Generating synthetic test surface...")
    u_test = np.linspace(0, 1, 50)
    v_test = np.linspace(0, 1, 40)
    U_test, V_test = np.meshgrid(u_test, v_test)
    
    # Create a complex test surface with noise
    X_test = U_test * 10
    Y_test = V_test * 8
    Z_test = (2 * np.sin(3 * np.pi * U_test) * np.cos(2 * np.pi * V_test) + 
             0.5 * np.sin(6 * np.pi * U_test * V_test) + 
             0.1 * np.random.randn(*U_test.shape))  # Add noise
    
    test_data = {
        'X': X_test.T,  # Transpose to match expected format
        'Y': Y_test.T,
        'Z': Z_test.T
    }
    
    # Test different fitting methods
    print("\n2. Testing different fitting methods...")
    
    methods_to_test = [
        ('least_squares', {}),
        ('regularized', {'lambda_smooth': 1e-6, 'lambda_fair': 1e-6}),
        ('weighted', {}),
        ('iterative', {'max_iterations': 20, 'adaptive_refinement': True})
    ]
    
    fitted_surfaces = []
    
    for method_name, method_params in methods_to_test:
        print(f"\n  Testing {method_name} method...")
        
        nurbs = Nurbs()
        try:
            result = nurbs.fit(test_data, degU=3, degV=3, 
                             control_points_shape=(20, 15),
                             method=method_name, **method_params)
            
            analysis = nurbs.analyze_fitting_quality(verbose=False)
            fitted_surfaces.append((method_name, nurbs, result, analysis))
            
            print(f"    RMS Error: {analysis['statistical_measures']['rms_error']:.6f}")
            print(f"    Max Error: {analysis['statistical_measures']['max_error']:.6f}")
            
        except Exception as e:
            print(f"    Failed: {e}")
    
    # Compare fitting methods
    print("\n3. Comparing fitting methods...")
    if len(fitted_surfaces) >= 2:
        best_surface = min(fitted_surfaces, 
                          key=lambda x: x[3]['statistical_measures']['rms_error'])
        print(f"\nBest method: {best_surface[0]}")
        print(f"Best RMS error: {best_surface[3]['statistical_measures']['rms_error']:.6f}")
        
        # Compare first two methods
        comparison = fitted_surfaces[0][1].compare_fits(fitted_surfaces[1][1], plot=False)
        print(f"\nComparison between {fitted_surfaces[0][0]} and {fitted_surfaces[1][0]}:")
        print(f"  RMS improvement: {comparison['comparison']['rms_improvement']:.6f}")
    
    # Test surface refinement
    print("\n4. Testing surface refinement...")
    if fitted_surfaces:
        test_surface = fitted_surfaces[0][1]  # Use first fitted surface
        
        print(f"  Original error: {test_surface.fitting_stats['rms_error']:.6f}")
        
        refinement_result = test_surface.refine_surface(
            target_error=test_surface.fitting_stats['rms_error'] * 0.5,
            max_control_points=(40, 30),
            method='adaptive'
        )
        
        print(f"  After refinement:")
        print(f"    Success: {refinement_result['success']}")
        print(f"    Final error: {refinement_result['final_error']:.6f}")
        print(f"    Final control points: {refinement_result['final_control_points']}")
        print(f"    Iterations: {refinement_result['iterations']}")
    
    # Test optimization
    print("\n5. Testing control point optimization...")
    if fitted_surfaces:
        test_surface = fitted_surfaces[0][1]  # Use first fitted surface
        original_error = test_surface.fitting_stats['rms_error']
        
        print(f"  Original error: {original_error:.6f}")
        
        try:
            opt_result = test_surface.optimize_control_points(
                method='gradient_descent', 
                learning_rate=0.001, 
                max_iterations=100
            )
            
            print(f"  After optimization:")
            print(f"    Success: {opt_result['success']}")
            print(f"    Final error: {opt_result['final_error']:.6f}")
            print(f"    Iterations: {opt_result['iterations']}")
            print(f"    Improvement: {original_error - test_surface.fitting_stats['rms_error']:.6f}")
            
        except Exception as e:
            print(f"    Optimization failed: {e}")
    
    # Generate comprehensive report
    print("\n6. Generating fitting report...")
    if fitted_surfaces:
        best_surface = fitted_surfaces[0][1]
        report_file = best_surface.export_fitting_report('demo_fitting_report.txt')
        print(f"  Report saved to: {report_file}")
    
    print("\n=== Demo completed! ===")
    return fitted_surfaces

def main():
    """
    Main function - can run either original example or advanced demo
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # Run advanced demo
        return demo_advanced_fitting()
    
    # Original functionality
    try:
        file_path = 'Data/Q.dat'
        data = np.loadtxt('Data/Q.dat')

        Q = {
            'X': data[:, 0].reshape((120, 179), order='F'),
            'Y': data[:, 1].reshape((120, 179), order='F'),
            'Z': data[:, 2].reshape((120, 179), order='F')
        }
        
        print("Loading data and fitting NURBS surface...")
        nurbs = Nurbs([], [], [], [], [])
        result = nurbs.fit(Q, 3, 3, (50, 80), method='regularized')
        
        # Analyze fitting quality
        analysis = nurbs.analyze_fitting_quality()
        
        # Plot results
        nurbs.plot_residuals()
        nurbs.plot(show_residuals=True)
        
        print(f"\nCurvature at (0,0): {nurbs.curvature_max(0, 0)}")
        
        # Export to STEP file
        try:
            stp_wr = initialize_step_writer()
            NRBS_OCC = Nurbs_to_STEPwriter(nurbs, stp_wr)

            from OCC.Display.SimpleGui import init_display
            display, start_display, add_menu, add_function_to_menu = init_display()

            display.DisplayShape(NRBS_OCC)
            start_display()
        except ImportError:
            print("OpenCASCADE not available, skipping STEP export")
            
    except FileNotFoundError:
        print("Data file not found. Running advanced demo instead...")
        return demo_advanced_fitting()
    except Exception as e:
        print(f"Error in main: {e}")
        print("Running advanced demo instead...")
        return demo_advanced_fitting()

    return

if __name__ == "__main__":
    main()