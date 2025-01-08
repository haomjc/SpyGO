import screwCalculus as sc
import casadi as ca
import numpy as np
import easy_plot as ep

def biquadratic_shape_function(r,s):
    """
    NODE MAP
    P2(-1,1)----P5(0,1)-----P1(1,1)
       |           |           |
       |           |           |
       |           |           |
    P6(-1,0)----P9(0,0)-----P8(1,0)
       |           |           |
       |           |           |
       |           |           |
    P3(-1,-1)---P7(0,-1)----P4(1,-1)
    """
    splus = 1+s
    sminus = 1-s
    rplus = 1+r
    rminus = 1-r
    rminus_sq = 1-r**2
    sminus_sq = 1-s**2
    # bilinear
    h1 = 0.25*rplus*splus
    h2 = 0.25*rminus*splus
    h3 = 0.25*rminus*sminus
    h4 = 0.25*rplus*sminus

    # cubic
    h5 = 0.5*rminus_sq*splus
    h6 = 0.5*sminus_sq*rminus
    h7 = 0.5*rminus_sq*sminus
    h8 = 0.5*sminus_sq*rplus

    # quadratic
    h9 = rminus_sq*sminus_sq

    H5 = h5*0.5*h9
    H6 = h6*0.5*h9
    H7 = h7*0.5*h9
    H8 = h8*0.5*h9
    H9 = h9

    H1 = h1 - 0.5*H5 - 0.5*H8 -0.25*H9
    H2 = h2 - 0.5*H5 - 0.5*H6 -0.25*H9
    H3 = h3 - 0.5*H6 - 0.5*H7 -0.25*H9
    H4 = h4 - 0.5*H7 - 0.5*H8 -0.25*H9
    return [H1, H2, H3, H4, H5, H6, H7, H8, H9]

def mid_point_element(r, s, nodes):

    shapes = biquadratic_shape_function(r, s)
    points = np.zeros(r.shape)
    for i in range(0,9):
        points += shapes[i]*nodes[i]
    return points

def chebyshev_interpolation_casadi(fun, N, funlims):
    """
    Curve interpolation with shape functions based on Chebyshev polynomials.
    
    Parameters:
    fun : function to interpolate (casadi function or callable)
    N : order of interpolation
    funlims : function variable bounds (required to rescale)
    
    Returns:
    P_fun : CasADi function for the interpolation
    a : Coefficients of the interpolation
    T : CasADi function for Chebyshev polynomials
    fj : Evaluated function at nodes
    Tj : Interpolated shape function values at Chebyshev nodes
    """
    
    # Initialize Chebyshev with CasADi symbolic variable
    w = ca.SX.sym('w')
    
    # Define Chebyshev polynomials (up to order 11)
    tau = ca.vertcat(
        1,
        w,
        2*w**2 - 1,
        4*w**3 - 3*w,
        8*w**4 - 8*w**2 + 1,
        16*w**5 - 20*w**3 + 5*w,
        32*w**6 - 48*w**4 + 18*w**2 - 1,
        64*w**7 - 112*w**5 + 56*w**3 - 7*w,
        128*w**8 - 256*w**6 + 160*w**4 - 32*w**2 + 1,
        256*w**9 - 576*w**7 + 432*w**5 - 120*w**3 + 9*w,
        512*w**10 - 1280*w**8 + 1120*w**6 - 400*w**4 + 50*w**2 - 1,
        1024*w**11 - 2816*w**9 + 2816*w**7 - 1232*w**5 + 220*w**3 - 11*w
    )
    
    tau_fun = ca.Function('tau', [w], [tau], {'cse': True})
    
    # Auxiliary vector to shift polynomials
    aux = ca.vertcat(*[1, w])
    aux = ca.repmat(aux, 11, 1)
    
    T = ca.SX.zeros(11)
    T[0:11] = tau[0:11] - aux[0:11]
    T[0] = (1 - w) * 0.5  # T_0
    T[1:N] = T[2:N+1]
    T[N-1] = (1 + w) * 0.5  # T_1

    T_fun = ca.Function('Tf', [w], [T[0:N]], {'cse':True})
    
    # Chebyshev roots
    roots = np.cos(np.pi * (np.arange(1, N+1) - 0.5) / N)
    
    # Adapt to allow -1 +1 nodes
    xr = roots[1:-1]
    xr = np.concatenate(([-1], xr, [1]))
    
    # Scale for function values
    xmax = funlims[1]
    xmin = funlims[0]
    xfun = (roots * (xmax - xmin) + (xmax + xmin)) * 0.5
    xfun = np.concatenate(([xmin], xfun, [xmax]))
    
    # Evaluate function at nodes
    if isinstance(fun, ca.Function) or callable(fun):
        fj = np.array(fun(xfun))
    else:
        fj = fun  # Numeric values at Chebyshev nodes (must be provided)
    
    if isinstance(fj, dict):
        fj = np.vstack((fj['X'], fj['Y'], fj['Z']))
    
    if fj.shape[0] == 3:
        fj = fj.T
    else:
        fj = fj.flatten().reshape(-1, 1)
    
    # Initialize Chebyshev coefficients
    a = np.zeros((fj.shape[1], N))
    a[:, -1] = fj[-1]  # root +1
    a[:, 0] = fj[0]    # root -1
    fj = fj[1:-1]  # remove -1 and +1 roots, use only Chebyshev nodes
    
    r = roots.reshape((N,1)).T
    tau_eval = np.array(tau_fun(r)).T
    Tj = np.array(T_fun(xr.reshape((1,N)))).T
    
    for ii in range(1, N-1):
        a[:, ii] = 2 / N * (fj.T @ tau_eval[:, ii+1])
    
    # Create the interpolation function P_fun
    P_fun = ca.Function('P', [w], [a @ T[0:N]],{'cse':True})

    return P_fun, a, T_fun, fj, Tj

def main():

    u, v = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
    print(np.ones((9,)))
    P = mid_point_element(u, v, np.linspace(-1,1,9))
    # print(u.shape)
    # print(P.shape)
    # print(P)

    # F = ep.Figure()
    # S = ep.surface(F, u, v, P)
    # F.set_scale(1,1,0.5)
    # F.show()
    c = np.linspace(0,1,8)

    P_fun, a, T_fun, fj, Tj = chebyshev_interpolation_casadi(c, 6, [0, 1])

    print(P_fun(c.reshape(1,-1)))
    print(a.shape)
    print(T_fun)
    print(P_fun.n_nodes())
    return

if __name__ == '__main__':
    main()