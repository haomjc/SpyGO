import scipy.io

# my_script.py
import sys

# to install and turn the script into an .exe: pyinstaller --onefile AVIO_CNC_NURBS.py
#
# Check if an argument was provided
if len(sys.argv) < 2:
    print("Usage: python AVIO_CNC_NURBS.py <*.mat file with NURBS info>")
    sys.exit(1)

# Get the first argument (after the script name)
user_input = sys.argv[1]

print(" Loading info from file directory: ", user_input)
raw_string = f"{user_input}"

# raw_string = r"C:\Program Files\MATLAB\R2021a\WP6_1_AVIO_NURBS_surface_data.mat"
# Load the .mat file
data = scipy.io.loadmat(raw_string)

control_points = data['pts']
knotsU = data['knotsU'].flatten()
knotsV = data['knotsV'].flatten()
degrees = data['degrees'].flatten()
shp = data['sZ'].flatten()
filename = data['filename'][0]

print(" Saving *.STEP to : ", filename)

control_points = control_points.reshape((3, shp[0], shp[1]), order='F')  # reshaping to have x y z coordinates in first dimension

def Nurbs_to_STEPwriter(control_points, Uknots, Vknots, pU, pV, shp, filename):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    # from OCC.Display.SimpleGui import init_display
    from OCC.Core.Geom import Geom_BSplineSurface
    import OCC.Core.Geom
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array2OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCC.Core.GeomLib import geomlib_ExtendSurfByLength
    #from OCC.Core.GeomLib import ExtendSurfByLength

    pU = int(pU)
    pV = int(pV)
    gridSizeControlPoints = [int(shp[0]), int(shp[1])] # transposing the control points matrix to match the pythonOCC array grid
    print('Grid size of control points: ', gridSizeControlPoints)
    print("Shape of control points array: ", control_points.shape)
    # initializing pythonOCC array grid
    pOCArray = TColgp_Array2OfPnt(0, gridSizeControlPoints[0]-1, 0, gridSizeControlPoints[1]-1)

    # copy those points in the pyOCC array grid
    for i in range(0, gridSizeControlPoints[0]):
        for j in range(0, gridSizeControlPoints[1]):
            pOCArray.SetValue(i, j, gp_Pnt(control_points[0,i,j],control_points[1,i,j], control_points[2,i,j]) )

    # removing knots multiplicity; pythonOCC requires unique knots values followed by a MULT array that defines multiplicity of each knot
    for i in range(pU):
        Uknots.pop(0)  # dropping zeros at beginning
        Uknots.pop(-1) # dropping ones at the end

    for i in range(pV):
        Vknots.pop(0)   # dropping zeros at beginning
        Vknots.pop(-1)  # dropping ones at the end

    UknotsOCC = TColStd_Array1OfReal(0, len(Uknots)-1)  # initializing pythonOCC arrays for Uknots
    VknotsOCC = TColStd_Array1OfReal(0, len(Vknots)-1)  # initializing pythonOCC arrays for Vknots
    multU     = TColStd_Array1OfInteger(0, len(Uknots)-1)  # initializing pythonOCC arrays for multiplicity of Uknots
    multV     = TColStd_Array1OfInteger(0, len(Vknots)-1)  # initializing pythonOCC arrays for multiplicity of Vknots

    # for loop to assign array elements... sadly can't just type OCCarray = pythonList. We need to use the SetValue method for each element
    for i in range(len(Uknots)):
        UknotsOCC.SetValue(i, Uknots[i])
        # check if it is the first or last knot
        if i == 0 or i == len(Uknots)-1:   # setting multiplicity to degree + 1
            multU.SetValue(i, pU+1)        # to the first and the final values
            continue
        # set multiplicity to 1 otherwise
        multU.SetValue(i, 1)

    # for loop to assign Vknots values and setting multiplicity array of V
    for i in range(len(Vknots)):
        VknotsOCC.SetValue(i, Vknots[i])
        # check if it is the first or last knot
        if i == 0 or i == len(Vknots)-1: # setting multiplicity to degree + 1
            multV.SetValue(i, pV+1)      # to the first and the final values
            continue
        # set multiplicity to 1 otherwise
        multV.SetValue(i, 1)

    NURBSsurf = Geom_BSplineSurface(pOCArray, UknotsOCC, VknotsOCC, multU, multV, pU, pV)
    BRepObj = BRepBuilderAPI_MakeFace(NURBSsurf, 0, 1, 0, 1, 1e-6).Face()
    print("NURBS surface created")
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.Interface import Interface_Static
    step_writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP203")
    step_writer.Transfer(BRepObj, STEPControl_AsIs)
    print("Writing STEP file...")
    step_writer.Write(f"{filename}")
    print(f"STEP file {filename} written")

    # from OCC.Display.SimpleGui import init_display
    # display, start_display, add_menu, add_function_to_menu = init_display()
    # display.DisplayShape(NURBSsurf)
    # start_display()

    return

Nurbs_to_STEPwriter(control_points, knotsU.tolist(), knotsV.tolist(), degrees[0], degrees[1], shp, filename)