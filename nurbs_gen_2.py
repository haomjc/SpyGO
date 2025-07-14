# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:09:13 2020

@author: Eugeniu Grabovic Università di Pisa (DICI)
"""

# lots of package imports for just building a NURBS and exporting to STEP...
from os import getcwd
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Display.SimpleGui import init_display
from OCC.Core.Geom import Geom_BSplineSurface
import OCC.Core.Geom
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.GeomLib import geomlib_ExtendSurfByLength
#from OCC.Core.GeomLib import ExtendSurfByLength
import OCC.Core.GeomLib

'''
The script works by reading 3 text files generated from Matlab:
    
    -1 executefilename.txt: file that contains the path where the step file has to be saved (1st line) and the namefile.step (2nd line);
                            on the 3d line it contains information about the grid size NxM of the control Points and the degree of the 
                            basis along U (pU) and the degree of the basis along V (pV)
                            
    
    -2 coefs.txt:           x then y then z values representing the Control Points of the NURBS surf.; note that those values have 
                            to be rearranged first into triplets of xyz points then into 3byNbyM matrix to match the grid
                            
    -2 knotsU.txt:          list of knot values along U variable (along profile in the case of gear tooth surface)
    
    -3 knotsV.txt:          same as knotsU.txt fot the V varaible (along the facewidth of the gears)
    
'''

# getting the step filename and path where to save
f = open(r"{path}\\executefilename.txt".format(path = getcwd()), "r")
fileInputs = f.read()
f.close()
fileInputs = fileInputs.split('\n')  # split the strings in the file along newlines
path = fileInputs[0]                 # path where to save is the first line
filename = fileInputs[1]             # file name is the second line
otherInfo = fileInputs[2].split(' ') # splitting the last line, it contains info about the NURBS

# extracting control points number and order info about the NURBS
gridSizeControlPoints = [int(otherInfo[0]), int(otherInfo[1])]
pU = int(otherInfo[2])
pV = int(otherInfo[3])


# opening control points file list
fcoefs = open(r"{path}\\coefs.txt".format(path = getcwd()), "r")
filecoefs = fcoefs.read()                                                       # read the coefficients file
fcoefs.close()                                                                  # close it
points = filecoefs.split()                                                      # split the strings in the file
lenPts = len(points)//3                                                         # the xyz coordinates are concatenated in a row, need to split 
                                                                                # the length to know the number of points
                                                                                
# extracting x y z coordinates
x = points[0:lenPts]
y = points[lenPts:2*lenPts]
z = points[2*lenPts:3*lenPts]

# conversion from string to float
x = list(map(float, x))
y = list(map(float, y))
z = list(map(float, z))

controlPointsList = []
for i in range(0, lenPts):
    controlPointsList.append(gp_Pnt(x[i], y[i], z[i]))

# initializing list of lists for grid of control points
pointsMatrixList = []

# initializing pythonOCC array grid
pOCArray = TColgp_Array2OfPnt(0, gridSizeControlPoints[0]-1, 0, gridSizeControlPoints[1]-1)

# for clarity i build the grid of points with python lists
for i in range(0, gridSizeControlPoints[1]):
    tempList = controlPointsList[gridSizeControlPoints[0]*i:gridSizeControlPoints[0]*(i+1)]
    pointsMatrixList.append(tempList)
    
# copy those points in the pyOCC array grid

for i in range(0, gridSizeControlPoints[0]):
    for j in range(0, gridSizeControlPoints[1]):
        pOCArray.SetValue(i, j, pointsMatrixList[j][i])


fUknots = open(r"{path}\\knotsU.txt".format(path = getcwd()), "r")
UknotsData = fUknots.read()                                                      # read the coefficients file
fUknots.close()                                                                  # close it
Uknots = UknotsData.split()                                                      # split the strings in the file

fVknots = open(r"{path}\\knotsV.txt".format(path = getcwd()), "r")
VknotsData = fVknots.read()                                                      # read the coefficients file
fVknots.close()                                                                  # close it
Vknots = VknotsData.split()                                                      # split the strings in the file

# string to float type transformation
Vknots = list(map(float, Vknots))
Uknots = list(map(float, Uknots))

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

# for loop to assign array elements... sadly can't just type OCCarray.SetValue(pythonList). We need to use the SetValue method for each element
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
    
# Finally creating the NURBS
NURBSsurf = Geom_BSplineSurface(pOCArray, UknotsOCC, VknotsOCC, multU, multV, pU, pV)
#ToothSpaceExtend= Handle_Geom_BoundedSurface_DownCast(NURBSsurf)
#OCC.Core.GeomLib.geomlib_ExtendSurfByLength(NURBSsurf,10,1,True,True);  # estendo superf. vano oltre tip CVX
#OCC.Core.GeomLib.geomlib_ExtendSurfByLength(NURBSsurf,10,1,True,False); # estendo superf. vano oltre tip CNV
#OCC.Core.GeomLib.geomlib_ExtendSurfByLength(NURBSsurf,10,1,False,True);
#OCC.Core.GeomLib.geomlib_ExtendSurfByLength(NURBSsurf,10,1,False,False);
# converting the Geom_BSplineSurface to BRep....Shape() which is a TopoDS_Shape, which can be built into a step file 
# (really weird jumping from many different objects like that tho...)
BRepObj = BRepBuilderAPI_MakeFace(NURBSsurf, 0, 1, 0, 1, 1e-6)
# args :                      surfaceHandle, startU, endU, startV, endV, tolerance

# initialize the STEP exporter
step_writer = STEPControl_Writer()
Interface_Static_SetCVal("write.step.schema", "AP203")

# transfer shapes and write file
step_writer.Transfer(BRepObj.Shape(), STEPControl_AsIs)
print("saving to directory: "+ path + filename)
status = step_writer.Write(path + filename)

print("Done generating the STEP file!")

# initializing display objects to display the surface (not really needed, just to visually check that the surface has been properly imported)
  
display, start_display, add_menu, add_function_to_menu = init_display()

display.DisplayShape(NURBSsurf)

start_display()