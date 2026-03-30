import os
import shutil
import multiprocessing
from MultyxInterface.main.core import worker_process


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