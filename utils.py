import ctypes
import numpy as np
import casadi as ca
import casadi
import os
import tkinter as tk
from tkinter import ttk
import time as time
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

########################### CLASSES
class Waitbar:
    """
    A class to create and manage a Tkinter-based progress bar window.
    Attributes:
    -----------
    total_steps : int
        The total number of steps for the progress bar.
    current_step : int
        The current step of the progress bar.
    window : tk.Tk
        The Tkinter window object.
    label : tk.Label
        The label widget to display messages.
    progress : ttk.Progressbar
        The progress bar widget.
    Methods:
    --------
    __init__(total_steps, title="Progress"):
        Initializes the Waitbar with the total number of steps and an optional title.
    update(step=1, message="Processing..."):
        Updates the progress bar and label with the current step and message.
    close():
        Closes the Tkinter window.
    """

    def __init__(self, total_steps, title="Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        
        # Create a Tkinter window
        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry("400x100")
        self.window.resizable(False, False)
        
        # Create a label
        self.label = tk.Label(self.window, text="Processing...", anchor="center")
        self.label.pack(pady=10)
        
        # Create a progress bar
        self.progress = ttk.Progressbar(self.window, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)
        self.progress["maximum"] = total_steps
        
        # Update the GUI to show the window
        self.window.update()
    
    def update(self, step=1, message="Processing..."):
        self.current_step = step
        self.progress["value"] = self.current_step
        self.label.config(text=message)
        self.window.update()
    
    def close(self, delay=0):
        time.sleep(delay)
        self.window.destroy()

################################# FUNCITONS
def compile_file(c_file):
    dll_filename = c_file[0:-2]+'.dll'
    compiler_path = r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsx86_amd64.bat'
    f = open('compile.bat', 'w')
    f.write(f'call "{compiler_path}" x64\n')
    f.write(f'cl /LD {c_file} /link /out:{dll_filename}')
    f.close()
    return

def flatten(t):
    """
    python list flattener (removes nested lists)
    """
    return [item for sublist in t for item in sublist]

def chop(value, tol):
    """
    chops precision digits
    """
    return round(value/tol)*tol

def dictprint(dictionary, tabstring = ''):
    """
    function to print the content of a dictionary to the command window. It recursively prints also nested dictionaries adding indentation tabs.
    """
    for key,value in dictionary.items():
        if type(value) is dict:
            tabstring = '   '
            print('\n')
            print(key)
            dictprint(value, tabstring)
            continue # interrupts for loop here and continues with next iteration

        if isinstance(value, np.ndarray):
            if value.size > 10:
                out_string = f"{type(value).__name__} of shape {value.shape}"
                print(tabstring, key, ':', out_string) 
                continue
        print( tabstring, key, ':', value)
    return

def dict_to_file(dictionary, filename, tabstring = '', w = True):
    """
    output_string = dict_to_file(dictionary, filename, tabstring = '', w = True)
    function to print the content of a dictionary to a txt file. It recursively prints also nested dictionaries adding indentation tabs.
    if option w is set to False the content won't be printed to a file but the output string can be returned and used.
    tabstring option shall not be used by the end user.
    """

    output_string = str()
    for key,value in dictionary.items():
        if type(value) is dict:
            tabstring = '   '
            output_string += key+'\n'
            output_string += dict_to_file(value, filename, tabstring = tabstring, w = False)
            output_string += '\n'
            continue # interrupts for loop here and continues with next iteration
        output_string += f'{tabstring}{key}  {value}\n'

    if w == True:
        with open(filename, 'w') as f:
            f.write(output_string)
        return

    return output_string

def msgbox(text, title = 'Title', style = 0):
    """
    request windows message box
    """
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def IPOPT_global_options():
    """
    default set for the IPOPT options ready to use
    """
    options = {
        'ipopt': {
            'max_iter': 3000,
            'nlp_scaling_method': 'none',
            'linear_solver': 'mumps', # 'ma57',
            'ma57_pre_alloc': 10,
            'linear_system_scaling': 'none',
            'tol': 1e-6,
            'accept_every_trial_step': 'no',
            # 'mumps_permuting_scaling': 2,
            # 'mumps_pivot_order': 3,
            # 'mumps_scaling': 10,
            'fast_step_computation': 'no',
            # 'ma97scaling': 0,
            # 'line_search_method': 'cg-penalty',
            'print_level': 5,
            # 'watchdog_shortened_iter_trigger': 0,
            # 'warm_start_init_point': 'yes',
            # 'mu_init': 1e-3,
            # 'mu_oracle': 'probing',
            'alpha_for_y': 'primal',
            'mu_strategy': 'adaptive', # 'adaptive'; % 'monotone'; %
            'adaptive_mu_globalization': 'never-monotone-mode',
            'min_refinement_steps': 20,
            'max_refinement_steps' : 30
            },
        'print_time': 0,
        'error_on_fail': False
        }
    return options

def reduce_2d(x):
    shp = x.shape
    return x.reshape(shp[0], -1, order = 'F')

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def isempty(x):
    """
    Checks for empty list, numpy array, other types to implement in the future, if necessary and/or useful
    """
    if isinstance(x, np.ndarray):
        return x.size == 0
    elif isinstance(x, list):
        return x.__len__() == 0
  
def compile_casadi_function(casadi_function, c_filename):
    # set the dll filename
    dll_filename = c_filename[0:-2]+'.dll'
    # generate c file
    casadi_function.generate(c_filename)
    # compile the function
    compile_file(c_filename)
    os.system('compile.bat')
    os.remove(c_filename[0:-2]+'.exp')
    os.remove(c_filename[0:-2]+'.lib')
    os.remove(c_filename[0:-2]+'.obj')
    os.remove(c_filename)
    os.remove('compile.bat')
    # load compiled dll
    # C = casadi.Importer(c_filename,'shell')
    # compiled_function = casadi.external('f',C)
    compiled_function = casadi.external('f', dll_filename)
    return compiled_function

def scattered_interpolant(points, values):
    """
    Creates a lambda function for linear interpolation within the convex hull 
    and nearest-neighbor extrapolation outside the hull.

    Parameters:
    - points: ndarray of shape (n_samples, n_dimensions)
      The coordinates of the data points.
    - values: ndarray of shape (n_samples,)
      The values at the data points.

    Returns:
    - interp_function: A lambda function that takes an array of query points 
      and returns interpolated/extrapolated values.
    """
    # Create the linear and nearest interpolators
    linear_interpolator = LinearNDInterpolator(points, values, fill_value=np.nan)
    nearest_interpolator = NearestNDInterpolator(points, values)

    def sample_interpolant(x_query, y_query):
        shp = x_query.shape
        query_points = np.vstack((x_query.flatten(), y_query.flatten())).T
        print(query_points)
        return np.where(
        np.isnan(linear_interpolator(query_points)),  # Check if linear interp result is nan
        nearest_interpolator(query_points),           # Use nearest neighbor for extrapolation
        linear_interpolator(query_points)             # Use linear interpolation otherwise
        ).reshape(shp[0], shp[1])
    # Return a lambda function that combines both
    return sample_interpolant

def zeros(shape, var_type):
    print(var_type)
    if issubclass(var_type, np.ndarray):
        return np.zeros(shape)
    elif var_type in (ca.SX, ca.MX, ca.DM):
        return ca.GenSX_zeros(shape[0], shape[1])
    else:
        raise TypeError("Unsupported type for zeros creation.")

def main():
    points = np.random.rand(10, 2)  # 10 points in 2D space
    values = np.random.rand(10)     # Corresponding values
    interpolator = scattered_interpolant(points, values)
    x_query = 3*np.random.rand(5, 5)  # 5 query points
    y_query = 3*np.random.rand(5, 5)  # 5 query points
    results = interpolator(x_query, y_query)
    print("Interpolated/Extrapolated Results:", results)

    print(zeros((6,5), ca.SX))

if __name__ == "__main__":
    main()

