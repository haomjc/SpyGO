import os
import numpy as np
import shutil
import subprocess

#### MSH file generation ####

def createBasicDataForMSH(member, path, mshname, n_teeth, C, nProf = 70, nFace = 22,
                          secondary=False, nFaceDiv=4, coordOrder=8, displOrder=4):
    """
    Python equivalent of MATLAB createBasicDataForMSH.
    
    Parameters
    ----------
    member : str
        'pinion' or 'gear'.
    path : str
        Base directory where 'mshData' will be created.
    mshname : str
        Name of the mesh.
    n_teeth : int
        Number of teeth of the member.
    nProf : int
        Number of profile divisions.
    nFace : int
        Number of face divisions.
    C : np.ndarray
        (6, 3) array with cone coefficients.
    secondary : optional
        Placeholder to mirror MATLAB's options.secondary (unused here).
    nFaceDiv : int, default=4
    coordOrder : int, default=8
    displOrder : int, default=4
    """
    member = member.lower()
    msh_dir = os.path.join(path, "mshData")
    os.makedirs(msh_dir, exist_ok=True)

    # Copy required executables and templates if not already present
    if not any(fname.endswith(".exe") for fname in os.listdir(msh_dir)):
        base_src = os.getcwd()
        for fname in [
            "HYPOID_MESH.exe",
            "MEDIUM_HYPOIDS.TPL",
            "FINEROOT_HYPOIDS.TPL",
            "FINEST_HYPOIDS.TPL"
        ]:
            src = os.path.join(base_src, fname)
            dst = os.path.join(msh_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, dst)

    # Write basic data file
    basic_file = os.path.join(msh_dir, f"{member}_basic_data.dat")
    with open(basic_file, "w") as fid:
        fid.write(f"{msh_dir}\n")
        fid.write(f"{mshname}\n")
        fid.write(f"{member}\n")
        fid.write(f"{n_teeth}\n")
        fid.write(f"{nProf}\n")
        fid.write(f"{nFace}\n")
        fid.write(f"{nFaceDiv}\n")
        fid.write(f"{coordOrder}\n")
        fid.write(f"{displOrder}\n")

        if secondary:
            fid.write("true\n")
            fid.write("true\n")
        else:
            fid.write("false\n")
            fid.write("false\n")

        # Expect 6 rows × 3 columns
        if C.shape != (6, 3):
            raise ValueError(f"C must be a 6×3 array, got {C.shape}")
        for row in C:
            fid.write(f"{row[0]:.10f}\t{row[1]:.10f}\t{row[2]:.10f}\n")

    # Create batch file
    disk = path[0]
    bat_path = os.path.join(msh_dir, "batch_msh_run.bat")
    with open(bat_path, "w") as fid:
        fid.write(f"{disk}:\n")
        fid.write(f"cd {msh_dir}\n")
        fid.write(f"HYPOID_MESH.exe {member}_basic_data.dat 1\n\n")

    # Execute and clean up
    os.system(bat_path)
    os.remove(bat_path)
    return

def CALYX_MSH(path, p, n, member):
    """
    Python equivalent of the MATLAB function CALYX_MSH.

    Parameters
    ----------
    path : str
        Base directory (e.g. "D:").
    p : np.ndarray
        3×N array of point coordinates.
    n : np.ndarray
        3×N array of normals.
    member : str
        Name of the member (used in filenames).
    """
    # Ensure arrays are numpy and shaped correctly
    p = np.asarray(p)
    n = np.asarray(n)
    assert p.shape[0] == 3 and n.shape[0] == 3, "p and n must be 3×N arrays"
    assert p.shape[1] == n.shape[1], "p and n must have same number of columns"

    # Prepare output directory
    msh_dir = os.path.join(path, "mshData")
    os.makedirs(msh_dir, exist_ok=True)

    # Write points + normals
    points_file = os.path.join(msh_dir, f"{member}_points_normals.dat")
    data = np.vstack((p, n)).T  # shape (N, 6)
    np.savetxt(points_file, data, fmt="%.16f", delimiter="\t")

    # Create batch file
    disk = path[0]  # e.g. 'D'
    bat_path = os.path.join(os.getcwd(), "batch_msh_run.bat")

    with open(bat_path, "w") as fid:
        fid.write(f"{disk}:\n")
        fid.write(f"cd {msh_dir}\n")
        fid.write(f"HYPOID_MESH.exe {member}_basic_data.dat 0\n\n")

    # Run batch file
    subprocess.run(bat_path, shell=True)

    # Delete batch file
    os.remove(bat_path)

def readQuadratureCones(path, member, write_to_file = True):
    """
    Python equivalent of MATLAB readQuadratureCones,
    but saves the result in a plain text file instead of .mat.

    Parameters
    ----------
    path : str
        Base directory containing the 'mshData' subfolder.
    member : str
        Member name (used in filenames).

    Returns
    -------
    Qcones : np.ndarray
        Array of shape (N, 3) containing the quadrature cone data.
    """
    msh_dir = os.path.join(path, "mshData")
    os.makedirs(msh_dir, exist_ok=True)

    file_path = os.path.join(msh_dir, f"{member}_quadrature_cones.dat")

    # Read all floating-point values
    with open(file_path, "r") as fid:
        data = np.fromfile(fid, sep=' ')

    # Reshape as 3 columns (same as MATLAB’s reshape(...,3,[])')
    Qcones = data.reshape(-1, 3)

    if write_to_file == False:
        return Qcones
    
    # Save in plain text (human-readable)
    out_path = os.path.join(msh_dir, f"Qcones_{member}.dat")
    np.savetxt(out_path, Qcones, fmt="%.16f", delimiter="\t")

    return Qcones

def conesIntersection(cone1, cone2):
    """
    Python equivalent of MATLAB conesIntersection.

    Parameters
    ----------
    cone1 : array-like of length 3
        [Ar, Az, B] coefficients for the first cone.
    cone2 : array-like of length 3
        [Ar, Az, B] coefficients for the second cone.

    Returns
    -------
    zR : np.ndarray of shape (2,)
        [z, R] coordinates of the intersection.
    """
    Ar, Az, B = cone1
    Ar2, Az2, B2 = cone2

    R = (Az2 / Az * B - B2) / (Ar2 - Az2 / Az * Ar)
    z = -(B + Ar * R) / Az

    return np.array([z, R])
