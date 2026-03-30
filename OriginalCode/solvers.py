from scipy.optimize import fsolve, root # fsolve(f, x0=x0, fprime=jacobian)
import casadi as ca
import time
from numba import njit
import numpy as np


def simple_newton_solver(residual_fun, x0, bounds=None, tolerance=1e-6, max_iterations=50, use_finite_differences=False):
    """
    Simple Newton solver with finite differences or CasADi jacobian and line search
    
    Parameters:
    -----------
    residual_fun : callable or casadi.Function
        Function that takes numpy array and returns residual vector 
    x0 : numpy.ndarray or list
        Initial guess
    bounds : list of tuples, optional
        [(x1_min, x1_max), (x2_min, x2_max), ...] bounds for variables
    tolerance : float
        Convergence tolerance
    max_iterations : int
        Maximum number of iterations
    use_finite_differences : bool
        If True, use finite differences for jacobian. If False, use CasADi automatic differentiation
        
    Returns:
    --------
    numpy.ndarray or None
        Solution vector or None if failed to converge
    """
    
    # Convert to numpy array
    x = np.array(x0, dtype=float)
    n_vars = len(x)
    
    # Set up jacobian computation method
    if hasattr(residual_fun, 'size_out'):  # CasADi function
        def eval_residual(x_val):
            return residual_fun(x_val).full().flatten()
            
        if not use_finite_differences:
            # Set up CasADi automatic differentiation
            x_sym = ca.SX.sym('x', n_vars)
            residual_sym = residual_fun(x_sym)
            jacobian_sym = ca.jacobian(residual_sym, x_sym)
            jacobian_fun = ca.Function('jacobian', [x_sym], [jacobian_sym])
            
            def compute_jacobian(x_val):
                return jacobian_fun(x_val).full()
        else:
            compute_jacobian = None  # Will use finite differences
    else:
        eval_residual = residual_fun
        compute_jacobian = None  # Will use finite differences for non-CasADi functions
    
    for iteration in range(max_iterations):
        try:
            residual = eval_residual(x)
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm < tolerance:
                return x
            
            # Compute jacobian using selected method
            if compute_jacobian is not None and not use_finite_differences:
                # Use CasADi automatic differentiation
                J = compute_jacobian(x)
            else:
                # Use finite differences (simple and robust)
                h = 1e-8
                J = np.zeros((len(residual), n_vars))
                for i in range(n_vars):
                    x_pert = x.copy()
                    x_pert[i] += h
                    try:
                        residual_pert = eval_residual(x_pert)
                        J[:, i] = (residual_pert - residual) / h
                    except:
                        J[:, i] = 0.0  # Fallback for failed evaluation
            
            # Solve with regularization
            try:
                JTJ = J.T @ J + 1e-6 * np.eye(n_vars)
                delta = np.linalg.solve(JTJ, -J.T @ residual)
            except:
                delta = -0.1 * J.T @ residual
            
            # Line search with backtracking
            alpha = 1.0
            for _ in range(20):
                x_new = x + alpha * delta
                
                # Apply bounds if specified
                if bounds is not None:
                    for i, (x_min, x_max) in enumerate(bounds):
                        if x_min is not None:
                            x_new[i] = max(x_new[i], x_min)
                        if x_max is not None:
                            x_new[i] = min(x_new[i], x_max)
                
                try:
                    new_residual_norm = np.linalg.norm(eval_residual(x_new))
                    if new_residual_norm < residual_norm or alpha < 0.01:
                        x = x_new
                        break
                except:
                    pass  # Failed evaluation, reduce alpha
                    
                alpha *= 0.5
                
        except:
            break

    # Final check
    try:
        final_residual_norm = np.linalg.norm(eval_residual(x))
        if final_residual_norm < 0.01:  # Reasonable tolerance
            return x
    except:
        pass
        
    return None


def robust_newton_solver(residual_fun, x0, bounds=None, tolerance=1e-6, max_iterations=50, use_finite_differences=False, fd_step=1e-8, debug=False):
    """
    Compatibility wrapper - redirects to simple_newton_solver for now
    """
    # Convert CasADi function to numpy function if needed
    if hasattr(residual_fun, 'size_out'):  # CasADi function
        def numpy_residual_fun(x):
            return residual_fun(x).full().flatten()
    else:
        numpy_residual_fun = residual_fun
        
    return simple_newton_solver(numpy_residual_fun, x0, bounds, tolerance, max_iterations)

def fsolve_casadi(casadi_obj, sym_x, sym_p, x0, p, jac_fun = None, solver = None):
    """
    Solve a system of nonlinear equations using CasADi and SciPy solvers.
    Parameters:
    casadi_obj (ca.SX or ca.MX or callable): The CasADi symbolic expression or function representing the system of equations.
    sym_x (ca.SX or ca.MX): The symbolic variable for the unknowns.
    sym_p (ca.SX or ca.MX): The symbolic variable for the parameters.
    x0 (array-like): Initial guess for the solution.
    p (array-like): Parameters for the system of equations.
    jac_fun (callable, optional): Function to compute the Jacobian of the system. If None, the Jacobian is computed automatically.
    solver (str, optional): The solver method to use ('hybr', 'lm', 'broyden1', etc.). If None, `fsolve` is used.
    Returns:
    tuple: A tuple containing:
        - sol (array-like): The solution to the system of equations.
        - jac (ca.Function): The CasADi function representing the Jacobian.
        - fun (ca.Function): The CasADi function representing the system of equations.
    """
    jac = jac_fun
    fun = casadi_obj

    if isinstance(casadi_obj, ca.SX) or isinstance(casadi_obj, ca.MX):
        fun = ca.Function('f', [sym_x, sym_p], [casadi_obj], {'cse': True})

    if jac_fun is None:
        jacobian_expr = ca.jacobian(fun(sym_x, sym_p), sym_x)
        jac = ca.Function('j', [sym_x, sym_p], [jacobian_expr])

    f = lambda x, p: fun(x, p).full().squeeze(-1)
    j = lambda x, p: jac(x, p).full().squeeze(-1)

    if solver is None:
        sol = fsolve(f, x0 = x0, fprime=j, args = p)
    else:
        sol = root(f, x0 = x0, jac=j, args = p, method = solver)
    return sol, jac, fun

def generate_poll_directions(n, poll_type='positive_basis_2n'):
    """
    Generate a set of poll directions for an n-dimensional problem.
    
    Parameters
    ----------
    n : int
        Dimension of the problem.
    poll_type : str, optional
        Type of polling directions to generate. Options include:
          - 'positive_basis_2n': 2n coordinate directions (each positive and negative unit vector).
          - 'positive_basis_n+1': n+1 directions forming a positive spanning set (e.g. for simplex‐based search).
    
    Returns
    -------
    directions : ndarray, shape (m, n)
        An array whose rows are the poll directions.
    """
    if poll_type == 'positive_basis_2n':
        # Create 2n directions: the unit vectors and their negatives.
        directions = np.vstack((np.eye(n), -np.eye(n)))
    elif poll_type == 'positive_basis_n+1':
        # Create n+1 directions that form a positive spanning set.
        # One common choice is to take the n standard unit vectors and one extra vector:
        # d_{n+1} = -sum_{i=1}^n e_i.
        directions = np.eye(n)
        extra = -np.ones(n)
        directions = np.vstack((directions, extra))
    else:
        raise ValueError(f"Unknown poll_type: {poll_type}")
    
    return directions

def pattern_search(func, x0,
                    step_size = 1.0,
                    tol = 1e-6,
                    max_iter = 1000,
                    contraction_factor = 2,
                    expansion_factor = 2,
                    complete_poll = False,
                    display = 'iter', 
                    output_fun = None):
    """
    Pattern search algorithm to find the minimum of a given function.
    
    Parameters:
    - func: The objective function to minimize.
    - x0: Initial guess (numpy array).
    - step_size: Initial step size for the search pattern.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.
    
    Returns:
    - Best found solution (minimum point).
    - Value of the function at the minimum point.
    """
    n = len(x0)  # Number of variables
    directions = generate_poll_directions(n, poll_type='positive_basis_2n')
    
    x = np.copy(x0)  # Current solution
    fval = func(x)   # Current function value

    # Print the header for the log
    if display.strip() == 'iter':
        print(f"{'Iteration':<12}{'x':<30}{'f(x)':<12}{'Step Size':<12}")

    for iteration in range(max_iter):
        found_better = False
        candidates = [None]*n
        candidates_fval = [None]*n
        # Try to explore in all directions (both positive and negative)
        for i, d in enumerate(directions):
            for sign in [+1, -1]:
                candidate = x + sign * step_size * d
                candidate_fval = func(candidate)
                
                if candidate_fval < fval:
                    x = candidate
                    fval = candidate_fval
                    found_better = True
                    candidates[i] = candidate
                    candidates_fval[i] = candidate_fval
                    if not complete_poll:
                        break  # Exit direction search since we found an improvement
                
            if found_better:
                step_size *= expansion_factor # increase mesh size
                method = 'successful poll'
                break  # Move to the next iteration since we found an improvement
        
        if output_fun is not None:
            output_fun(x, fval)
            
        # If no better solution found, reduce the step size
        if not found_better:
            method = 'mesh refinement'
            step_size /= contraction_factor

        # Check for convergence
        if step_size < tol:
            if display.strip() == 'iter' or display.strip() == 'final':
                print(f"Minimum found at: {x}, function value: {fval}")
            break
        
        if display.strip() == 'iter':
            if iteration % 30 == 0:
                print(f"{'Iteration':<12}{'x':<30}{'f(x)':<12}{'Step Size':<12}{'Method':<12}")

            # Log output with consistent formatting
            x_str = np.array2string(x, precision=4, separator=',', suppress_small=True)
            print(f"{iteration+1:<12}{x_str:<30}{fval:<12.6f}{step_size:<12.4f}{method:<12}")

    if iteration+1 == max_iter and display != 'off':
        print(f"Patternsearch stopped: Maximum number of iterations exceeded: max_iter={max_iter}")
        print(f"Minimum found at: {x}, function value: {fval}")
    return x, fval

def main():
    # Example usage with a test function (Rosenbrock function)
    @njit
    def rosenbrock(x):
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1.0-x[:-1])**2.0)

    # Starting point
    x0 = np.array([-1.2, 1.0])
    
    # Starting value
    print(rosenbrock(x0))

    # Run the pattern search
    start_time = time.perf_counter() # 
    result, fval = pattern_search(rosenbrock, x0, display='iter', complete_poll=True, max_iter = 10)
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(execution_time)
    
if __name__ == "__main__":
    main()
