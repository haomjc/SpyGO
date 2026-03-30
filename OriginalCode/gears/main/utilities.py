"""
Utility functions for gear computations and plotting
"""
import numpy as np
import casadi as ca
import sys
import os

# Add the parent directory to the path to import solvers
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from solvers import robust_newton_solver


def nonlinspace(start, end, n_points, power=1.0, reverse=False, endpoint=True):
    """Generate non-linearly spaced points between start and end
    
    Parameters:
    -----------
    start : float
        Starting value
    end : float 
        Ending value
    n_points : int
        Number of points to generate
    power : float
        Controls distribution:
        - power > 1: more points toward start
        - power < 1: more points toward end
        - power = 1: linear spacing
    reverse : bool
        If True, flip the distribution
    endpoint : bool
        If True, include the endpoint
        
    Returns:
    --------
    numpy.ndarray
        Non-linearly spaced points
    """
    t = np.linspace(0, 1, n_points, endpoint=endpoint)
    if reverse:
        t = 1 - t
    t_scaled = t ** power
    if reverse:
        t_scaled = 1 - t_scaled
    return start + (end - start) * t_scaled


def find_curve_intersection(fillet_fun, flank_fun, u1, u2, u3, tolerance=1e-6, use_finite_differences=True):
    """Find intersection between fillet and flank curves using CasADi residual function"""
    
    # Set up symbolic variables and residual function
    u_vars = ca.SX.sym('u', 2)  # [u_fillet, u_flank]
    
    # Define residual: fillet_point - flank_point = 0
    p_fillet = fillet_fun(u_vars[0])
    p_flank = flank_fun(u_vars[1])
    residual = p_fillet - p_flank
    
    # Create CasADi function for residual
    residual_fun = ca.Function('intersection_residual', [u_vars], [residual])

    # Initial guess: midpoints of each section
    u_f_init = (u1 + u2) / 2
    u_fl_init = (u2 + u3) / 2
    x0 = [u_f_init, u_fl_init]
    
    # Set bounds for the variables
    bounds = [(u1, u2), (u2, u3)]
    
    # Use the extracted simple Newton solver with CasADi function
    from solvers import simple_newton_solver
    solution = simple_newton_solver(residual_fun, x0, bounds, tolerance, max_iterations=50, use_finite_differences=use_finite_differences)
    
    if solution is not None:
        u_f, u_fl = solution
        
        # Final validation using CasADi function
        final_residual_vec = residual_fun(solution).full().flatten()
        final_residual = np.linalg.norm(final_residual_vec)
        
        # Check if intersection is at boundary (not real undercut)
        boundary_tolerance = 1e-3
        if (abs(u_f - u2) < boundary_tolerance and abs(u_fl - u2) < boundary_tolerance):
            return None, None
        
        if final_residual < 0.01:  # Good precision
            return u_f, u_fl
    
    return None, None


def find_gear_end_parameter(point_fun, outer_radius, u_min, u_max, internal_gear=False, tolerance=1e-6):
    """Find u parameter where gear profile reaches outer radius using bisection
    
    Parameters:
    -----------
    point_fun : casadi.Function
        Gear point function
    outer_radius : float
        Target outer radius
    u_min, u_max : float
        Search bounds
    internal_gear : bool
        True for internal gears, False for external gears
    tolerance : float
        Convergence tolerance
        
    Returns:
    --------
    float
        Parameter value where profile reaches outer radius
    """
    max_iterations = 100
    
    # For external gears: radius typically increases with u
    # For internal gears: radius typically decreases with u (smaller outer radius)
    radius_increasing = not internal_gear
    
    for _ in range(max_iterations):
        u_mid = (u_min + u_max) / 2
        try:
            point = point_fun(u_mid).full()
            radius = float(np.sqrt(point[0]**2 + point[1]**2))
            
            if abs(radius - outer_radius) < tolerance:
                return u_mid
            
            # Bisection logic depends on whether radius increases with u
            if radius_increasing:
                if radius < outer_radius:
                    u_min = u_mid
                else:
                    u_max = u_mid
            else:
                if radius > outer_radius:
                    u_min = u_mid
                else:
                    u_max = u_mid
                
        except Exception:
            u_max = u_mid
    
    return u_mid