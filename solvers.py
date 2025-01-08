from scipy.optimize import fsolve # fsolve(f, x0=x0, fprime=jacobian)
import casadi as ca
import time
from numba import njit

def fsolve_casadi(casadi_obj, sym_x, sym_p, x0, p, jac_fun = None):
    """
    automatic interface between casadi's expression and scipy's fsolve 
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

    sol = fsolve(f, x0 = x0, fprime=j, args = p)
    return sol, jac, fun

import numpy as np

def pattern_search(func, x0,
                    step_size = 1.0,
                    tol = 1e-6,
                    max_iter = 1000,
                    contraction_factor = 2,
                    expansion_factor = 2,
                    complete_poll = False,
                    display = 'iter'):
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
    directions = np.eye(n)  # Identity matrix to define pattern directions (axis-aligned)
    
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
    # x = ca.SX.sym('x')
    # p = ca.SX.sym('p')
    # f = x**2 +6*x**3 - x +p*3
    # sol, _, _ = fsolve_casadi(f, x, p, 1, 1)
    # print(sol)

    # Example usage with a test function (Rosenbrock function)
    # @njit
    def rosenbrock(x):
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1.0-x[:-1])**2.0)

    # Starting point
    x0 = np.array([-1.2, 1.0])
    
    # Starting value
    print(rosenbrock(x0))

    # Run the pattern search
    start_time = time.perf_counter()
    result, fval = pattern_search(rosenbrock, x0, display='off')
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(execution_time)


if __name__ == "__main__":
    main()


     