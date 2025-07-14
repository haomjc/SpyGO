from casadi.casadi import exp
import numpy as np
import casadi as ca # for symbolic computations
import screwCalculus as sc # screwCalculus is a custom module for screw theory computations 

from general_utils import * 
from typing import Literal # for type hinting
import easy_plot as ep 
import matplotlib.pyplot as plt
import matplotlib as mpl

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

from hypoid.main.utils import *
from hypoid.main.geometry import *
from hypoid.main.identification import *
from hypoid.main.dataclasses import DesignData, FlankAndMemberNumericalData, FlankAndMemberNumericalDataWithBoth, MemberData

plt.rcParams.update({  # 'text.usetex': True,  # Enable LaTeX
    'axes.labelsize': 20,  # Font size for x and y labels
    'axes.titlesize': 20,  # Font size for title
    'xtick.labelsize': 20,  # Font size for x-axis tick labels
    'ytick.labelsize': 20,  # Font size for y-axis tick labels
    'legend.fontsize': 20   # Font size for legend
})


# The lowest level: data stored for each flank (concave/convex/both)
@dataclass
class FlankData:
    data: Any = field(default_factory=list)  # could be list, array, CasADi function, etc.

# Each member (pinion/gear) has multiple flanks
@dataclass
class MemberData:
    concave: FlankData = field(default_factory=FlankData)
    convex: FlankData = field(default_factory=FlankData)
    both: FlankData = None  # Optional, only used where needed

    def __getitem__(self, key: str) -> FlankData:
        """Allow dynamic access: member['concave']"""
        return getattr(self, key)

    def __setitem__(self, key: str, value: FlankData):
        """Allow dynamic assignment: member['concave'] = value"""
        setattr(self, key, value)

# Top level: pinion and gear
@dataclass
class TemplateData:
    pinion: MemberData = field(default_factory=MemberData)
    gear: MemberData = field(default_factory=MemberData)

    def __getitem__(self, key: str) -> MemberData:
        """Dynamic access: template['pinion']"""
        return getattr(self, key)

    def __setitem__(self, key: str, value: MemberData):
        """Dynamic assignment: template['pinion'] = value"""
        setattr(self, key, value)


class Hypoid:

    def __init__(self, designData, toothData, coneData, nProf = 11, nFace = 16, nFillet = 12):
        #tooth sampling size
        self.nProf = nProf
        self.nFace = nFace
        self.nFillet = nFillet

        # info
        self.pinGenerationProcess = " "
        self.gearGenerationProcess = " "
        self.initialToothData = {}
        self.initialConeData = {}

        # main design data (machine-tool settings)
        self.designData = DesignData

        # sampling data
        self.surfPoints = FlankAndMemberNumericalData()
        self.surfNormals = FlankAndMemberNumericalData()
        self.surfTriplets = FlankAndMemberNumericalData()
        self.filletPoints = FlankAndMemberNumericalData()
        self.interpTriplets = FlankAndMemberNumericalData()
        self.surfcurvature = FlankAndMemberNumericalData()
        self.eqMeshing = FlankAndMemberNumericalData()
        self.Point = FlankAndMemberNumericalData()   # casadi function
        self.Normal = FlankAndMemberNumericalData() # casadi function
        self.pointsFullBounds = FlankAndMemberNumericalData() # it contains the four edges of the flank as arrays in a list
        self.normalsFullBounds = FlankAndMemberNumericalData()
        self.conjugateData = FlankAndMemberNumericalData()

        # nurbs output
        self.nurbsFit = FlankAndMemberNumericalDataWithBoth()

        # zR tooth data boundaries
        self.zRfillet = FlankAndMemberNumericalData()
        self.zRfillet = FlankAndMemberNumericalData()                   # flank-fillet transition line in axial plane
        self.zRfullvec = FlankAndMemberNumericalData()                  # z - R coordinates in array form derived from sampling points (nProf + nFillet)xnFace
        self.zRbounds = FlankAndMemberNumericalData()                   # bounds on flank fillet transition
        self.zRwithRoot = MemberData()                                  # bounds with the fillet
        self.zRrootTriplets = FlankAndMemberNumericalData()              # triplets for the rootcone sampling
        self.zRfullBounds = FlankAndMemberNumericalData()
        self.zRPCA = FlankAndMemberNumericalData()
        self.zRinOther = FlankAndMemberNumericalData()
        self.zRinOtherCorners = FlankAndMemberNumericalData()
        self.zRtipOther = FlankAndMemberNumericalData()
        self.rootLineStruct = FlankAndMemberNumericalData()
        self.originalRootLine = None  # to store the original rootline data before any modifications

        # rigid TCA
        self.pathCurve = []
        self.TCAfun = []
        self.pinTCA = []
        self.gearTCA = []
        self.pinPsiRange  = []   
        self.gearPsiRange = []

        # rotor info
        self.EPGalpha = []

        # data structs for mach-tool identification 
        self.currentEaseOff = []
        self.identificationProblemConjugate = []      # conjugate pinion identification problem struct
        self.identificationProblemEaseOff = []        # easeOff identification problem struct
        self.identificationProblemOptimization = []   # embedded identification problem for the automatic optimization
        self.identificationProblemSpreadBlade = []    # spread blade identification
        self.identificationProblemTopography = []     # generic topography identification

        # data for Calyx interface
        self.LTCA = {}  # LTCA data structure

        # call constructor
        # input = 'designData'
        # if toothData is not None:
        #     input = 'coneData'
        # self.constructHypoid(designData = designData, toothData = toothData, coneData = coneData, inputData = input)

    ### CONSTRUCTORS

    def from_macro_geometry(self, system_data, tooth_data, cone_data):
        method = 1
        if system_data["hypoidOffset"] == 0:
            method = 0
        gear_gen_type = system_data["gearGenType"]
        self.designData = AGMAcomputationHypoid(system_data["HAND"], 
                                        system_data["taper"],
                                        cone_data, tooth_data,
                                        rc0 = cone_data["rc0"],
                                        GearGenType = gear_gen_type,
                                        Method = method)
        
        # intialize the design data
        self.compute_parameters(self.designData)
        return
    
    def from_machine_tool_settings(self, designData: DesignData):
        """
        Initialize the Hypoid object from machine-tool settings.
        """
        self.designData = designData
        self.compute_parameters(self.designData)
        return
    
    def from_file(self, file_path: str):
        """
        Load Hypoid data from a file.
        """
        # Implement file loading logic here
        # self.designData = load_from_file(file_path))
        self.compute_parameters(self.designData)
        return
    
    ## end of class constructor
    
    @staticmethod
    def conesIntersection(cone1, cone2):
        Ar = cone1[0]
        Az = cone1[1]
        B = cone1[2]

        Ar2 = cone2[0]
        Az2 = cone2[1]
        B2 = cone2[2]

        R = (Az2/Az*B - B2)/(Ar2 - Az2/Az*Ar)
        z = -(B+Ar*R)/Az

        zR = [z, R]
        return zR
    
    @staticmethod
    def EPGalphaToFrames(EPGalpha, data):
        """
        EPGalpha misalignments converted to frame displacements and z axis orientation
        """
        handPin = data['SystemData']['HAND']
        shaft_angle = data['SystemData']['shaft_angle']
        signOffset = -(int(handPin.lower() == 'right') - int(handPin.lower() == 'left'))
        offset = data['SystemData']['hypoidOffset']
        pin_dict = {}
        gear_dict = {}
        pin_dict['originXYZ'] = [EPGalpha[1], signOffset*(offset + EPGalpha[0]), 0]
        pin_dict['Zdir'] = [sin(shaft_angle*pi/180 + EPGalpha[3]), 0, cos(shaft_angle*pi/180 + EPGalpha[3])]
        gear_dict['originXYZ'] = [0, 0, EPGalpha[2]]
        gear_dict['Zdir'] = [0, 0, 1]
        return pin_dict, gear_dict
    
    @staticmethod
    def sideFromMemberAndFlank(member, flank):
        side = 'coast'
        if (member.lower() == 'pinion' and flank.lower() == 'concave') or (member.lower() == 'gear' and flank.lower() == 'convex'):
            side = 'drive'
        return side
    
    @staticmethod
    def flankFromMemberAndSide(member, side):
        flank = 'convex'
        if (side.lower() == 'drive' and member.lower() == 'pinion') or (side.lower() == 'coast' and member.lower() == 'gear'):
            flank = 'concave'
        return flank
    
    @staticmethod
    def getIndexArray():
        """
        sequential order of the machine-tool settings
        """
        indexes = [
                1, 10, 19, 28, 37, 46, 55, 64, # Radial motion row coefficients
                2, 11, 20, 29, 38, 47, 56, 65, # Tilt motion 
                3, 12, 21, 30, 39, 48, 57, 66, # Swivel motion
                4, 13, 22, 31, 40, 49, 58, 67, # vertical motion
                5, 14, 23, 32, 41, 50, 59, 68, # helical motion
                6, 15, 24, 33, 42, 51, 60, 69, # cradle motion
                16, 25, 34, 43, 52, 61, 70,    # roll motion
                8, 17, 26, 35, 44, 53, 62, 71, # machine center to back (axial motion)
                9, 18, 27, 36, 45, 54, 63,     # root angle motion
                72, 73, 74, 75, 76, 77, 78, 79, 80, 81, # concave tool
                82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92 # convex tool
                ]
        return indexes
    
    # sampling functions

    def samplezR(self, z, R, member, flank, triplets = None, data_type = 'base', SBmachining = False):
        
        if z is None or len(z) == 0:
            points = getattr(getattr(self.surfPoints, member), flank)
            z = np.atleast_2d(points[2,:])
            R = np.atleast_2d(np.linalg.norm(points[0:2,:], axis=0))
        
        if not triplets or triplets is None:
            interpolant_function = getattr(getattr(self.interpTriplets, member), flank)
            triplets = interpolated_triplets_zR(interpolant_function, z, R)
            triplets[0, :] = np.maximum(triplets[0, :]-0.2, 0.03)
            shp = z.shape
            triplets = triplets.reshape(3, shp[0], shp[1], order = 'F')
        
        if data_type.lower() == 'base':
            data = self.designData
            points, normals, triplets = rz_sampling_casadi(R, z, data, member, flank, triplets)
        elif data_type.lower() in ['easeoff', 'ease_off']:
            data = self.identificationProblemEaseOff['designData']
            points, normals, triplets = rz_sampling_casadi(R, z, data, member, flank, triplets)
        elif data_type.lower() in ['conjugate', 'conj']:
            data = self.identificationProblemConjugate['designData']
            points, normals, triplets = rz_sampling_casadi(R, z, data, member, flank, triplets)
        elif data_type.lower() in ['opti', 'optimization']:
            data = self.identificationProblemOptimization['designData']
            points, normals, triplets = rz_sampling_casadi(R, z, data, member, flank, triplets)
        elif data_type.lower() in ['nurbs']:
            points, normals, UV = rz_sampling_NURBS_casadi(R, z, data, member, flank, triplets)

        return points, normals, triplets
    
    def sample_surface(self, member, flank, sampling_size = None, extend_tip = False, updateData = True, FW_vec = None, return_output = False, triplet_guess = None):
        """
        sampling_size = [n_face, n_profile, n_fillet]
        extend_tip = extend the tip of the blank's face cone boundary (useful for boolean subtractions)
        updateData = updates Hypoid properties (surface points, enveloping triplets, etc. ). Such data is often used as initial guess for other methods supposed to modify the tooth flank
        """
        side = self.sideFromMemberAndFlank(member, flank)
        data = self.designData

        # extract the sampling size (face, profile and fillet points)
        nF = self.nFace; nP = self.nProf; nfil = self.nFillet
        if sampling_size is not None:
            nF = sampling_size[0];  nP = sampling_size[1];  nfil = sampling_size[2]
        
        if extend_tip:
            common, subcommon = get_data_field_names(member, flank, fields = 'common')
            data[common][f'{subcommon}FACEAPEX'] = data[common][f'{subcommon}FACEAPEX'] + 1
            data[common][f'{subcommon}OUTERCONEDIST'] = data[common][f'{subcommon}OUTERCONEDIST'] + 0.3
            data[common][f'{subcommon}FACEWIDTH'] = data[common][f'{subcommon}FACEWIDTH'] + 0.6

        p, n, p_tool, n_tool, csi_theta_phi, z_tool, p_fillet, p_root, n_root, root_variables, p_bounds, n_bounds =\
              surface_sampling_casadi(data, member, flank, [nF, nP, nfil], triplet_guess = triplet_guess, spreadblade = False, FW_vec = FW_vec)
        
        if updateData == True:
            self.surfPoints[member][flank] = p
            self.surfNormals[member][flank] = n
            self.filletPoints[member][flank] = p_fillet
            self.surfTriplets[member][flank] = csi_theta_phi

            z = p_fillet[2,:]
            R = np.sqrt(p_fillet[0,:]**2 + p_fillet[1,:]**2)
            self.zRfillet[member][flank] = np.c_[z, R]

            z = p_root[2,:]
            R = np.linalg.norm(p_root[0:2,:], axis=0)
            self.rootLineStruct[member][flank] = {'zR' : np.row_stack((z,R)),
                                                  'triplets' : root_variables[0:3,:],
                                                  'points' : p_root,
                                                  'normals' : n_root}

            zR = []
            for edge in p_bounds:
                z = edge[2,:]
                R = np.linalg.norm(edge[0:2,:], axis = 0)
                zR.append(np.row_stack((z,R)))

            self.zRfullBounds[member][flank] = zR
            self.pointsFullBounds[member][flank] = p_bounds # it is a list of arrays. Each array is a boundary edge. There are 4 edges in total
            self.normalsFullBounds[member][flank] = n_bounds # same as above

            z = p[2,:]
            R = np.linalg.norm(p[0:2,:], axis=0)
            self.zRfullvec[member][flank] = np.row_stack((z,R))

            self.interpolate_triplets_over_zr(member, flank, z, R, csi_theta_phi)

        if return_output:
            return p, n, z, R, csi_theta_phi

        return 

    def compute_conjugate_points_to_gear(self, flank, zR, EPGalpha, offset_psi, designData = None, rephase_points = False):
        
        if flank.lower() == 'convex':
            opposite_flank = 'concave'
        else:
            opposite_flank = 'convex'
        
        designData = self.designData
        if designData is not None:
            designData = designData

        triplets = interpolated_triplets_zR(self.interpTriplets['gear'][flank], zR[:, 0], zR[:, 1])
        p, n, zRconj, base_triplets, psi_P, psi_G, angular_ease_off, v_pg_p, omega = \
            pinion_conjugate_to_gear(designData, flank, zR, EPGalpha, triplets, self.interpTriplets['pinion'][opposite_flank], 0)
        self.pinPsiRange = psi_P
        self.gearPsiRange = psi_G
        setattr(self, 'PsiPoffset', offset_psi)

        # store the conjugate data
        self.conjugateData['pinion'][flank] = {'points': p, 'normals': n, 'triplets': base_triplets}

        return p, n, base_triplets, zRconj, psi_P, angular_ease_off, v_pg_p, omega, psi_G
    
# parameters update functions

    def compute_parameters(self, new_data, FW_vec = None, no_sync = False, triplets = None):
        self.designData = new_data
        self.designData = shaft_segment_computation(self.designData)

        # Compute R-z boundaries
        self.designData, _, self.zRwithRoot['pinion'] = rz_boundaries_computation(self.designData, 'pinion')
        self.designData, _, self.zRwithRoot['gear']   = rz_boundaries_computation(self.designData, 'gear')

        # Extract pinion and gear data points
        PCpin = self.zRwithRoot['pinion'][3, :]
        PCgear = self.zRwithRoot['gear'][3, :]

        trip_pin_cnv, trip_pin_cvx, trip_gear_cnv, trip_gear_cvx = [], [], [], []
        if not not self.interpTriplets['pinion']['concave']: # if the initial triplets are not empty
            trip_pin_cnv = [
                self.interpTriplets['pinion']['concave']['csi'][PCpin[0], PCpin[1]], 
                self.interpTriplets['pinion']['concave']['theta'][PCpin[0], PCpin[1]],
                self.interpTriplets['pinion']['concave']['phi'][PCpin[0], PCpin[1]]
            ]

            trip_pin_cvx = [
                self.interpTriplets['pinion']['convex']['csi'][PCpin[0], PCpin[1]], 
                self.interpTriplets['pinion']['convex']['theta'][PCpin[0], PCpin[1]],
                self.interpTriplets['pinion']['convex']['phi'][PCpin[0], PCpin[1]]
            ]

            trip_gear_cnv = [
                self.interpTriplets['gear']['concave']['csi'][PCgear[0], PCgear[1]], 
                self.interpTriplets['gear']['concave']['theta'][PCgear[0], PCgear[1]],
                self.interpTriplets['gear']['concave']['phi'][PCgear[0], PCgear[1]]
            ]

            trip_gear_cvx = [
                self.interpTriplets['gear']['convex']['csi'][PCgear[0], PCgear[1]], 
                self.interpTriplets['gear']['convex']['theta'][PCgear[0], PCgear[1]],
                self.interpTriplets['gear']['convex']['phi'][PCgear[0], PCgear[1]]
            ]

        # Sampling surfaces and interpolating
        waitbar = Waitbar(step = 0, title="Initializating Hypoid", text="Generating pinion concave flank")
        pin_pts_cnv, pin_proot_cnv, pin_nroot_cnv, _, _ = self.sample_surface('pinion', 'concave',
                                                                         triplet_guess = trip_pin_cnv,
                                                                         FW_vec = FW_vec,
                                                                         return_output=True)
        waitbar.update(step = 25, message="Generating pinion convex flank")
        pin_pts_cvx, pin_proot_cvx, pin_nroot_cvx, _, _ = self.sample_surface('pinion', 'convex',
                                                                         triplet_guess = trip_pin_cvx,
                                                                         FW_vec = FW_vec,
                                                                         return_output=True)
        waitbar.update(step = 50, message="Generating gear concave flank")
        gear_pts_cnv, gear_proot_cnv, gear_nroot_cnv, _, _ = self.sample_surface('gear', 'concave',
                                                                            triplet_guess = trip_gear_cnv,
                                                                            FW_vec = FW_vec,
                                                                            return_output=True)
        waitbar.update(step = 75, message="Generating gear convex flank")
        gear_pts_cvx, gear_proot_cvx, gear_nroot_cvx, _, _ = self.sample_surface('gear', 'convex',
                                                                           triplet_guess = trip_gear_cvx,
                                                                           FW_vec = FW_vec,
                                                                           return_output=True)
        
        waitbar.update(step = 85, message="PCA computation...")
        # Spacing line between two flanks based on hand
        hand = self.get_systemHand()
        nTp = self.get_Nteeth('pinion')
        nTg = self.get_Nteeth('gear')

        if hand.lower() == 'right':
            pin_pts_cnv = sc.rotZ(2 * np.pi / nTp).dot(pin_pts_cnv[:3, :])
            gear_pts_cvx = sc.rotZ(2 * np.pi / nTg).dot(gear_pts_cvx[:3, :])
        else:
            pin_pts_cvx = sc.rotZ(2 * np.pi / nTp).dot(pin_pts_cvx[:3, :])
            gear_pts_cnv = sc.rotZ(2 * np.pi / nTg).dot(gear_pts_cnv[:3, :])

        # Compute root and topland spacing, needed for FEM meshing
        setattr(self, 'rootSpacing', {'pinion': (pin_proot_cnv + pin_proot_cvx) * 0.5, 'gear': (gear_proot_cnv + gear_proot_cvx) * 0.5})

        nF, nP, nfil = self.nFace, self.nProf, self.nFillet
        pin_pts_cnv = pin_pts_cnv.reshape(3, nF, nP + nfil - 1)[:, :, -1]
        pin_pts_cvx = pin_pts_cvx.reshape(3, nF, nP + nfil - 1)[:, :, -1]

        gear_pts_cnv = gear_pts_cnv.reshape(3, nF, nP + nfil - 1)[:, :, -1]
        gear_pts_cvx = gear_pts_cvx.reshape(3, nF, nP + nfil - 1)[:, :, -1]

        setattr(self, 'toplandSpacing', {'pinion': (pin_pts_cnv + pin_pts_cvx) * 0.5, 'gear': (gear_pts_cnv + gear_pts_cvx) * 0.5})

        for member in ("pinion", "gear"):
            for flank in ("concave", "convex"):
                self.zRbounds[member][flank] = zr_activeflank_bounds(
                    self.designData,
                    member,
                    flank,
                    self.zRfillet[member][flank]
                )
        
        # set dictionaries for modified designs (conjugate, easeOff, optimization, spreadBlade, topography)
        for attribute in ['Conjugate', 'EaseOff', 'Optimization', 'SpreadBlade', 'Topography']:
            setattr(self, f'identificationProblem{attribute}', {'designData': self.designData})

        if no_sync:
            waitbar.update(step = 100, message="Done!")
            waitbar.close(delay=1)
            return

        # PCA and synchronization computations
        for component, side in [('gear', 'drive'), ('gear', 'coast'), ('pinion', 'drive'), ('pinion', 'coast')]:
            self.compute_PCA(component, side, self.EPGalpha)

        waitbar.update(step = 90, message="Flank synchronization...")   
        self.compute_synch_angle()
        
        waitbar.update(step = 100, message="Done!")
        waitbar.close(delay=1)
        return
    
# zR functions 

    def compute_zr_grid(self, member, flank, n_prof, n_face, active_flank = True, extend_tip = False):  
        zR_bounds = self.zRbounds[member][flank]

        if active_flank == False:
            zR_fillet = self.zRwithRoot[member]
            zR_fillet = np.r_[zR_fillet[0:1,:], zR_bounds[[1,0],:]]

        if extend_tip:
            extend_coeff = 0.35
            zR_root_toe = zR_bounds[0,:]
            zR_root_heel = zR_bounds[1,:]
            zR_tip_heel = zR_bounds[2,:]
            zR_tip_toe = zR_bounds[3,:]

            zR_bounds[0,:] = zR_root_toe - extend_coeff*(zR_root_heel - zR_root_toe)/2
            zR_bounds[1,:] = zR_root_heel + extend_coeff*(zR_root_heel - zR_root_toe)/2
            zR_bounds[2,:] = zR_tip_heel + extend_coeff*(zR_tip_heel - zR_root_heel)
            zR_bounds[3,:] = zR_tip_toe + extend_coeff*(zR_tip_toe - zR_root_toe)

        u = np.linspace(-1, 1, n_face)
        v = np.linspace(-1, 1, n_prof)

        X, Y = np.meshgrid(u, v)
        z, R = grid_to_rz(X, Y, zR_bounds, method=1)

        return z, R

    def interpolate_triplets_over_zr(self, member, side, z, R, triplets, data_type: Literal['base', 'easeOff', 'conjugate', 'opti'] = 'base'):
        # Reshape z and R to be column vectors for compatibility with interpolation
        z = np.array(z).flatten(order = 'F')
        R = np.array(R).flatten(order = 'F')
        coords = np.column_stack((z, R))

        # Select the target attribute based on dataType
        if data_type.lower() == 'base':
            target = self.interpTriplets
        elif data_type.lower() in {'easeoff', 'ease_off'}:
            target = self.identificationProblemEaseOff
        elif data_type.lower() in {'conjugate', 'conj'}:
            target = self.identificationProblemConjugate
        elif data_type.lower() in {'opti', 'optimization'}:
            target = self.identificationProblemOptimization
        else:
            raise ValueError(f"Unsupported dataType: {data_type}")

        # Ensure the member and side structures exist within the target
        if member not in target:
            target[member] = {}
        if side not in target[member]:
            target[member][side] = {}

        # Interpolants for each of the triplet components and store in the chosen dictionary
        csi = triplets[0, :].flatten(order = 'F')
        theta = triplets[1, :].flatten(order = 'F')
        phi  = triplets[2, :].flatten(order = 'F')
        target[member][side] = {
            'csi' : scattered_interpolant(coords, csi),
            'theta' : scattered_interpolant(coords, theta),
            'phi' : scattered_interpolant(coords, phi),
            'phi_csitheta' : scattered_interpolant(np.column_stack((csi, theta)), phi)
        }
        
        return
   
    def compute_PCA(self, member, side, EPGalpha):
        # Select the correct flank based on the member and side
        if side.lower() == 'drive' and member.lower() == 'pinion':
            flank = 'concave'
            other_flank = 'convex'
        else:
            flank = 'convex'
            other_flank = 'concave'

        if member.lower() == 'pinion':
            other_member = 'gear'
        else:
            other_member = 'pinion'
        
        zR = PCA_computation(self.designData, member, side, EPGalpha,
                            (self.pointsFullBounds[member][flank], self.normalsFullBounds[member][flank]),
                            (self.pointsFullBounds[other_member][other_flank], self.normalsFullBounds[other_member][other_flank]),
                            self.nFace
                            )
        
        # Store the PCA data in the appropriate dictionary
        self.zRinOther[member][flank] = zR
        
        return
    
    def compute_synch_angle(self):
        zRgear = self.zRfullvec['gear']['convex']
        z = np.reshape(zRgear[0, :], (self.nFace, -1), order = 'F')
        # z = z[:, 0]
        return
    
    def buildCasadiDerivatives(self):
        # build the casadi functions for the surface points and normals
        return

# identification functions

    def backup_rootline(self):
        """Save the current rootline as original if not already saved."""
        if self.originalRootLine is None:
            self.originalRootLine = copy.deepcopy(self.rootLineStruct)
        return
    
    def buildIdentificationProblem(self, member, flank, x_index, lb, ub, zR, problem_type = 'conjugate', bound_points_tol = None, scaling_bounds = None):
        
        if zR.shape[0] == 2:
            zR = zR.T

        if flank.lower() == 'drive' and member.lower() == 'pinion': # side instead of flank provided
            flank = 'concave'
        else:
            flank = 'convex'
        
        self.backup_rootline() # save the original rootline data before any modifications

        if problem_type.lower() == 'conjugate':
            field = 'identificationProblemConjugate'
            # add rootline data also to the identification problem data
            field_data = getattr(self, field)

            if bound_points_tol is None:
                bound_points_tol = 10 # mm tolerance for the bounding box for identified points w.r.t base ones

            field_data[member][flank]['root_constraint'] = self.originalRootLine[member][flank]
            triplets = self.conjugateData[member][flank]['triplets']

        elif problem_type.lower() == 'easeoff' or problem_type.lower() == 'optimization':

            field = 'identificationProblemEaseOff'
            if problem_type.lower() == 'optimization':
                field = 'identificationProblemOptimization'

            if bound_points_tol is None:
                bound_points_tol = 0.4

            z = zR[:,0]
            R = zR[:,1]
            base_points, base_normals, base_triplets = self.samplezR(z, R, member, flank, x_index)
            field_data = getattr(self, field)
            field_data[member][flank]['root_constraint'] = self.originalRootLine[member][flank]
            root_points = field_data[member][flank]['root_constraint']['points']
            root_normals = field_data[member][flank]['root_constraint']['normals']
            root_triplets = field_data[member][flank]['root_constraint']['triplets']

            field_data[member][flank] = {'base_points': base_points, 'base_normals': base_normals, 'base_triplets': base_triplets,
                                         'root_points': root_points, 'root_normals': root_normals, 'root_triplets': root_triplets}
            
        lb_scaling = lb
        ub_scaling = ub
        if scaling_bounds is not None:
            lb_scaling = scaling_bounds[0]
            ub_scaling = scaling_bounds[1]

        # solver, settings = machine_identification_problem(triplets, x_index, lb, ub, lb_scaling, ub_scaling, 
        #                                                   self.designData, member, flank, root_constraint = root_constraint,
        #                                                   bound_points_tol = bound_points_tol
        #                                                   )
        field
                                                          
        

        return

    def buildIdentificationProblemSB(self, member, flank, x_index, lb, ub): # Spread Blade variant
        
        return
    
    # getters
    def get_Nteeth(self, member):
        common, sub_common = get_data_field_names(member, 'concave', fields = 'common')
        return self.designData[common][f'{sub_common}NTEETH']
    
    def get_systemHand(self):
        return self.designData['SystemData']['HAND']
    
    def get_settings_index(self, machine_settings_names):
        dict = machine_settings_index()
        
        # Normalize the keys to lowercase
        normalized_dict = {k.lower(): v for k, v in dict.items()}

        # Return the index for each machine setting name if it exists, otherwise return None
        indexes = [normalized_dict.get(name.lower(), None) for name in machine_settings_names]

        # check if all indexes are valid
        if None in indexes:
            raise ValueError("Invalid machine settings name(s) provided")
        
        return indexes 
    
    def get_machine_settings_names(self):

        """
        Returns the names of the machine settings as a list.
        """
        return machine_settings_index()
    # setters

    # plot
    def plot(self, member = 'pinion', flank = 'concave', whole_gear = False):
        
        HAND = self.designData['SystemData']['HAND']
        s = -1
        if HAND.lower() == 'right' and member.lower() == 'gear':
            s = +1
        elif HAND.lower() == 'left' and member.lower() == 'pinion':
            s = +1

        both_flanks = False
        if flank.lower() == 'both':
            both_flanks = True
            flank = 'concave'
        # Extract the points for the specified member and flank from the surfPoints dictionary
        points = self.surfPoints[member][flank]
        n_face = self.nFace

        # Create a new figure and plot the points
        F = ep.Figure(title=member+flank)

        # Reshape the points to be 3xN for plotting
        X = points[0,:].reshape(n_face, -1, order = 'F')
        Y = points[1,:].reshape(n_face, -1, order = 'F')
        Z = points[2,:].reshape(n_face, -1, order = 'F')

        # Create a surface plot
        S = ep.surface(F, X, Y, Z)

        if both_flanks:
            flank = 'convex'
            # Extract the points for the specified member and flank from the surfPoints dictionary
            points = self.surfPoints[member][flank]
            n_face = self.nFace

            # Reshape the points to be 3xN for plotting
            X2 = points[0,:].reshape(n_face, -1, order = 'F')
            Y2 = points[1,:].reshape(n_face, -1, order = 'F')
            Z2 = points[2,:].reshape(n_face, -1, order = 'F')

            # Create a surface plot
            S2 = ep.surface(F, X2, Y2, Z2, parent=S)
        
        if whole_gear:
            nz = self.get_Nteeth(member)
            if both_flanks:
                X = np.hstack([np.fliplr(X), X2])
                Y = np.hstack([np.fliplr(Y), Y2])
                Z = np.hstack([np.fliplr(Z), Z2])
            Xnew = X
            Ynew = Y
            Znew = Z
            for i in range(1, nz):
                
                # Create a surface plot
                Rot = sc.rotZ(2*np.pi/nz*i*s)
                Xnew = np.hstack([ X*Rot[0,0] + Y*Rot[0,1] + Z*Rot[0,2], Xnew,])
                Ynew = np.hstack([ X*Rot[1,0] + Y*Rot[1,1] + Z*Rot[1,2], Ynew])
                Znew = np.hstack([ X*Rot[2,0] + Y*Rot[2,1] + Z*Rot[2,2], Znew])
                
            ep.surface(F, Xnew, Ynew, Znew)

        F.updateImage()
        F.show()
        return
    
    def plot_zr_bounds(self, member, flank):
        zr_with_root = self.zRwithRoot[member]

        # append first point to close the loop
        zr_with_root = np.vstack((zr_with_root, zr_with_root[0,:]))

        zr_activeflank_bounds = self.zRbounds[member][flank]
        if zr_activeflank_bounds.shape[0] < zr_activeflank_bounds.shape[1]:
            zr_activeflank_bounds = zr_activeflank_bounds.T

        zr_activeflank_bounds = np.vstack((zr_activeflank_bounds, zr_activeflank_bounds[0,:]))

        # Create a new figure with pyplot
        fig = plt.figure()
        plt.title(f"{member} {flank} z-R bounds")
        plt.xlabel("z")
        plt.ylabel("R")       
        plt.plot(zr_with_root[:,0], zr_with_root[:, 1], label = "z-R with root", color = 'red')
        plt.plot(zr_activeflank_bounds[:,0], zr_activeflank_bounds[:,1], label = "Active flank bounds", color = 'black')
        plt.axis('tight')
        plt.axis('equal')
        plt.grid()
        plt.show()

        return

    # LOG funcitons
    def print_settings_names(self):

        # Get the machine settings names
        names = machine_settings_index()

        # Print dictionary keys
        print("Valid machine settings names:")
        for name in names:
            print(name)

        return
    
    def __str__(self):
        """Custom string representation to print attribute names, their types, and available methods."""
        description = f"{self.__class__.__name__} object:\n"

        # Print attributes and their types
        description += "Attributes:\n"
        for attr, value in vars(self).items():
            description += f"       {attr} : {type(value).__name__}\n"

        # Print available methods
        description += "Methods:\n"
        methods = [method for method in dir(self) if callable(getattr(self, method)) and not method.startswith("__")]
        for method in methods:
            description += f"       {method}\n"

        return description.strip()  # Remove trailing newline for cleaner output
    
# debugging

def main():
    SystemData = {
        'HAND': "Right",
        'taper' : "Standard",
        'hypoidOffset' : 25
    }

    coneData = {
        'SIGMA' : 90,
        'a' : SystemData['hypoidOffset'],
        'z1' : 9,
        'u' : 3.7,
        'de2': 225,
        'b2' : 38.8,
        'betam1' : 45,
        'rc0' : 75,
        'gearBaseThick' : 15,
        'pinBaseThick' : 8,
    }

    coneData['z2'] = round(coneData['u']*coneData['z1'])
    coneData['u'] = coneData['z2']/coneData['z1']

    toothData = {
        'alphaD' : 21,
        'alphaC' : 20,
        'falphalim' : 1,
        'khap' : 1,
        'khfp' : 1.25,
        'xhm1' : 0.45,
        'jen' : 0.1,
        'xsmn' : 0.05,
        'thetaa2' : None,
        'thetaf2' : None
    }
    H = Hypoid(SystemData, toothData, coneData)

    H.sample_surface('gear', 'concave')
    H.sample_surface('gear', 'convex')
    
    H.sample_surface('pinion', 'concave')
    H.sample_surface('pinion', 'convex')

    H.plot_zr_bounds('gear', 'concave')
    
    
    return
    
if __name__ == "__main__":
    main()
    