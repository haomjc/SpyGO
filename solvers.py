from scipy.optimize import fsolve, root # fsolve(f, x0=x0, fprime=jacobian)
import casadi as ca
import time
from numba import njit
import numpy as np

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
