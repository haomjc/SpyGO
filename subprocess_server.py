# worker.py
import time
import ctypes
import time
from multyxInterface import *


def worker_process(conn, worker_id, path, ses_file):
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
