import ctypes
import sys
import os

import multiprocessing 
import time
import shutil

def init_multyx_types(path, multyxlib):
    # multyx.dll is a shared library that exposes a few standard C functions.
    # ctypes is a python package for calling C functions.
    # Load multyx.dll using the python ctypes.cdll.LoadLibrary() function:
    
    # In order to tell python how to call each function in the DLL,
    # specify the python-C interface of thes functions:
    #
    # void* OpenMultyxSession(
    #   char* SessionFileName,
    #   MsgCallbackFunctionPointer ptr_infocallback,
    #   MsgCallbackFunctionPointer ptr_errorcallback,
    #   MsgCallbackFunctionPointer ptr_warningcallback
    # );
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)
    multyxlib.OpenMultyxSession.restype=ctypes.c_void_p
    multyxlib.OpenMultyxSession.argtypes=[
    ctypes.c_char_p,
    ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p),
    ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p),
    ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
    ]
    #
    #  Interface for setting values of multyx variables.
    # int SetValueFloatingPointVariable(void* SessionHandle,char* InputSpecifier,double Value);
    multyxlib.SetValueFloatingPointVariable.restype=ctypes.c_int
    multyxlib.SetValueFloatingPointVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_double]
    # int SetValueBoolVariable(void* SessionHandle,char* InputSpecifier,bool Value);
    multyxlib.SetValueBoolVariable.restype=ctypes.c_int
    multyxlib.SetValueBoolVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_bool]
    # int SetValueSwitchVariable(void* SessionHandle,char* InputSpecifier,int Value);
    multyxlib.SetValueSwitchVariable.restype=ctypes.c_int
    multyxlib.SetValueSwitchVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int]
    # int SetValueIntegerVariable(void* SessionHandle,char* InputSpecifier,int Value);
    multyxlib.SetValueIntegerVariable.restype=ctypes.c_int
    multyxlib.SetValueIntegerVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int]
    # int SetValueStringVariable(void* SessionHandle,char* InputSpecifier,char* Value);
    multyxlib.SetValueStringVariable.restype=ctypes.c_int
    multyxlib.SetValueStringVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_char_p]
    # int ExecuteScript(void* SessionHandle,char* Script,int ShowOutput);
    multyxlib.ExecuteScript.restype=ctypes.c_int
    multyxlib.ExecuteScript.argtypes=[ctypes.c_void_p,ctypes.c_char_p]
    #  Interface for getting values of multyx variables.
    # int GetValueFloatingPointTaggedItem(void* SessionHandle,char* OutputSpecifier,double* Value);
    multyxlib.GetValueFloatingPointTaggedItem.restype=ctypes.c_int
    multyxlib.GetValueFloatingPointTaggedItem.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_double)]
    # int GetValueBoolVariable(void* SessionHandle,char* OuputSpecifier,bool* Value);
    multyxlib.GetValueBoolVariable.restype=ctypes.c_int
    multyxlib.GetValueBoolVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_bool)]
    # int GetValueSwitchVariable(void* SessionHandle,char* OuputSpecifier,int* Value);
    multyxlib.GetValueSwitchVariable.restype=ctypes.c_int
    multyxlib.GetValueSwitchVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_int)]
    # int GetValueIntegerVariable(void* SessionHandle,char* OuputSpecifier,int* Value);
    multyxlib.GetValueIntegerVariable.restype=ctypes.c_int
    multyxlib.GetValueIntegerVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_int)]
    multyxlib.GetValueStringVariable.restype=ctypes.c_int
    # Call GetValueStringVariable() like this:
    # bufsize=1024
    # buf= (ctypes.c_char * buf_size)()
    # pbuf= (ctypes.POINTER(ctypes.c_char) * 1)(buf)
    # retval=multyxlib.GetValueStringVariable(SessionHandle,b"DESCRIPTION",pbuf,bufsize)
    multyxlib.GetValueStringVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),ctypes.c_int]
    multyxlib.CloseMultyxSession.restype=ctypes.c_int
    multyxlib.CloseMultyxSession.argtypes=[ctypes.c_void_p]
    #
    # Default Null callback. Use this if you do not wish to process
    # Informational, Error, or Warning messages:
    # NullMsgCallback=ctypes.cast(None,ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p))    
    #
    # ShowOutput=1 if you want to see the output from multyx, =0 otherwise
    ShowOutput=ctypes.c_int(0) #Don't show output
    # os.chdir(current_path)
    return multyxlib

# Take a message as parameter, and return a bool DoAboort
@ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
def InfoCallBack(msg):
    #print(msg.decode("utf-8"))
    # Do not abort session:
    return ctypes.c_bool(False)

# Take a message as parameter, and return a bool DoAboort
@ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
def ErrorCallBack(msg):
    print(msg.decode("utf-8"))
    # Do not abort when an error message is received:
    return ctypes.c_bool(False)

# Take a message as parameter, and return a bool DoAboort
@ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
def WarningCallBack(msg):
    print(msg.decode("utf-8"))
    # Do not abort session:
    return ctypes.c_bool(False)

def init_multyx_session(path, ses_file, multyxlib):
    #sys.path.insert(0,'/opt/ansol/calyx')
    
    # Start a session by passing it the name of the session file to read.
    # It will acquire the license, open the session file and wait.
    # It will return a SessionHandle of type ctypes.c_voidp, which is needed
    # for all subsequent calls to the opened session.
    SessionFileName=bytes(ses_file, encoding='utf-8')
    SessionHandle=multyxlib.OpenMultyxSession(SessionFileName, InfoCallBack, ErrorCallBack, WarningCallBack)
    if SessionHandle==ctypes.c_voidp(None):
        print("Failed to start session")
        return 1
    return (multyxlib, SessionHandle)

def worker_process(conn, worker_id, path, ses_file):
    """
    This function is run by each worker process. It initializes the multyx interface and
    session, and then waits for parameters to be sent through the pipe. When a parameter
    is received, it performs a task (e.g., squaring the parameter) and sends the result
    back to the parent process. This process continues until the parent sends the 'exit'
    command, at which point the worker shuts down.
    """

    def LOG(msg):
        msg = f'Worker {worker_id}: ' + msg
        print(msg)
        return
    # init hypoid
    ...
    # init path associated with the worker id
    ...
    # initialize multyx interfaces
    library=ctypes.cdll.LoadLibrary("C:/Program Files/Ansol/Transmission3Dx64/multyx.dll")
    library = init_multyx_types(path, library)
    interface, seshandle = init_multyx_session(path, ses_file, library)
    LOG(f"worker {worker_id} initialized")
    conn.send(None)

    while True:
        # Wait to receive data from the pipe
        param = conn.recv()

        # identify hypoid machine-tool settings to best match the ease-off defined by the parameters
        
        if isinstance(param, str) and param.lower() == "exit":
            LOG("Shutting down.")
            interface.CloseMultyxSession(seshandle)

            break

        # Perform a task (e.g., square the parameter)
        LOG(f"Received parameter {param}\n")
        LOG("Starting simulation")
        t = time.time()
        interface.ExecuteScript(seshandle, param)
        LOG(" Sending result ")
        
        # Send the result back to the parent process
        LOG(f'Simulation completed successfully in {time.time() - t} seconds!')
        conn.send(None)
    
    conn.close()  # Close the connection when done

def main():

    # Set the Number of workers (parallel processes)
    NUM_WORKERS = 2

    # get the T3D path and sesfile to be used as template for the workers
    path = r'C:\Users\egrab\Desktop\T3D_test' # path to the T3D folder
    ses_file = 'T3D.ses' # ses file to be used as template for the workers

    newpaths = [] # list of paths for the workers
    ses_files = [] # list of ses files for the workers

    for i in range(NUM_WORKERS):
        newpath = fr'{path}\worker_{i}' # path for the worker
        new_ses = f'{ses_file[0:-4]}_{i}.ses' # ses file for the worker
        if os.path.exists(newpath): # remove the folder if it exists
            shutil.rmtree(newpath)
        if not os.path.exists(newpath): # create the folder if it does not exist
            os.makedirs(newpath)
        os.system(fr'copy {path}\{ses_file} {newpath}\{new_ses}') # copy the ses file to the worker folder
        newpaths.append(newpath) 
        ses_files.append(new_ses)
        
    # Launch worker processes and establish pipes
    processes = []
    parent_connections = []
    for i in range(NUM_WORKERS):
        parent_conn, child_conn = multiprocessing.Pipe()  # Create a pipe for communication
        p = multiprocessing.Process(target=worker_process, args=(child_conn, i+1, newpaths[i], ses_files[i]))
        p.start()
        processes.append(p)
        parent_connections.append(parent_conn)  # Keep track of parent ends of the pipes
        LOG = parent_conn.recv()
        print(LOG)

    # Send initial parameters to each worker
    for i, conn in enumerate(parent_connections):
        parameters = b'GENERATE\n STARTANAL\n'
        conn.send(parameters)  # Send parameters through the pipe
        print(f"Client: Sent parameter {parameters} to worker {i+1}")

    # Wait for results from each worker
    for i, conn in enumerate(parent_connections):
        result = conn.recv()  # Receive the result from the pipe
        print(f"Client: Received result from worker {i+1}: {result}")

    # Send more parameters to each worker
    for i, conn in enumerate(parent_connections):
        parameters = b'GENERATE\n STARTANAL\n'
        conn.send(parameters)  # Send parameters through the pipe
        print(f"Client: Sent parameter {parameters} to worker {i+1}")

    # Wait for results from each worker
    for i, conn in enumerate(parent_connections):
        result = conn.recv()  # Receive the result from the pipe
        print(f"Client: Received result from worker {i+1}: {result}")

    # Send 'exit' command to each worker to shut them down
    for i, conn in enumerate(parent_connections):
        conn.send("exit")  # Send the exit signal through the pipe
        print(f"Client: Sent exit command to worker {i+1}")

    # Ensure all workers have finished
    for p in processes:
        p.join()  # Wait for each worker process to finish

    print("Client: All workers have completed.")

    return

if __name__ == '__main__':
    main()