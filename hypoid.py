from casadi.casadi import exp
import numpy as np
import casadi as ca
import screwCalculus as sc
from hypoid_functions import *
from hypoid_utils import *
from hypoid_sampling import *
from utils import *
from typing import Literal
import easy_plot as ep

import copy

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
        self.designData = {}

        # sampling data
        template_dictionary = {'pinion' : {'concave':[], 'convex':[]}, 'gear': {'concave':[], 'convex':[]}}
        self.surfPoints = copy.deepcopy(template_dictionary)
        self.surfNormals = copy.deepcopy(template_dictionary)
        self.surfTriplets = copy.deepcopy(template_dictionary)
        self.filletPoints = copy.deepcopy(template_dictionary)
        self.interpTriplets = copy.deepcopy(template_dictionary)
        self.surfcurvature = copy.deepcopy(template_dictionary)
        self.eqMeshing = copy.deepcopy(template_dictionary)
        self.Point = copy.deepcopy(template_dictionary)   # casadi function
        self.Normal = copy.deepcopy(template_dictionary) # casadi function
        self.pointsFullBounds = copy.deepcopy(template_dictionary)
        self.normalsFullBounds = copy.deepcopy(template_dictionary)

        # nurbs output
        template_dictionary = {'pinion' : {'concave':[], 'convex':[], 'both': []}, 'gear': {'concave':[], 'convex':[], 'both': []}}
        self.nurbsFit = copy.deepcopy(template_dictionary)

        # zR tooth data boundaries
        template_dictionary = {'pinion' : {'concave':[], 'convex':[]}, 'gear': {'concave':[], 'convex':[]}}
        self.zRfillet = copy.deepcopy(template_dictionary)
        self.zRfillet = copy.deepcopy(template_dictionary)                   # flank-fillet transition line in axial plane
        self.zRfullvec = copy.deepcopy(template_dictionary)                   # z - R coordinates in array form derived from sampling points (nProf + nFillet)xnFace
        self.zRbounds = copy.deepcopy(template_dictionary)                   # bounds on flank fillet transition
        self.zRwithRoot = {'pinion':[], 'gear':[]}                           # bounds with the fillet
        self.zRrootTriplets = copy.deepcopy(template_dictionary)              # triplets for the rootcone sampling
        self.zRfullBounds = copy.deepcopy(template_dictionary)
        self.zRPCA = copy.deepcopy(template_dictionary)
        self.zRinOther = copy.deepcopy(template_dictionary)
        self.zRinOtherCorners = copy.deepcopy(template_dictionary)
        self.zRtipOther = copy.deepcopy(template_dictionary)
        self.rootLineStruct = copy.deepcopy(template_dictionary) 

        # rigid TCA
        self.pathCurve = []
        self.TCAfun = []
        self.pinTCA = []
        self.gearTCA = []
        self.pinPhiRange  = []   
        self.gearPhiRange = []

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
        input = 'designData'
        if toothData is not None:
            input = 'coneData'
        self.constructHypoid(designData = designData, toothData = toothData, coneData = coneData, inputData = input)

    ## end of class constructor
    
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
    
    def constructHypoid(self, designData = None, toothData = None, coneData = None, gearGenType = 'Generated', EPGalpha = np.zeros((4,1)), inputData = "designData", synchronization_flag = False):
        self.initialCone = coneData
        self.initialToothData = toothData

        if inputData.lower() != "designData".lower():
            systemData = designData

            method = 1
            if systemData["hypoidOffset"] == 0:
                method = 0

            self.designData = AGMAcomputationHypoid(systemData["HAND"], 
                                                    systemData["taper"],
                                                    coneData, toothData,
                                                    rc0 = coneData["rc0"],
                                                    GearGenType = gearGenType,
                                                    Method = method)
            
            self.designData = shaft_segment_computation(self.designData)
            self.designData, triplets_pin_CNV, triplets_pin_CVX = approxToolIdentification_casadi(self.designData, 'Pinion', RHO = 50000)
            self.designData, triplets_gear_CNV, triplets_gear_CVX = approxToolIdentification_casadi(self.designData, 'Gear'  , RHO = 500)
            setattr(self, 'init_triplet', {
                'pinion':
                {
                    'concave': triplets_pin_CNV,
                    'convex' : triplets_pin_CVX
                },
                'gear':
                {
                    'concave':triplets_gear_CNV,
                    'convex':triplets_gear_CVX
                }
                }
                    )
            gear_common, gear_sub_common = get_data_field_names('gear', 'concave', fields = 'common')
            pinion_common, pinion_sub_common = get_data_field_names('pinion', 'concave', fields = 'common')
            self.designData[gear_common][f'{gear_sub_common}USE_SPRD_BLD_THICKNESS'] = 'TRUE'
            self.designData[pinion_common][f'{pinion_sub_common}USE_SPRD_BLD_THICKNESS'] = 'TRUE'
        else:
            self.designData = designData

        # compute all the necessary parameters
        self.compute_parameters(self.designData, no_sync = synchronization_flag)
        return 
    
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
            self.zRfillet[member][flank] = np.r_[z, R]

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
            self.pointsFullBounds[member][flank] = p_bounds
            self.normalsFullBounds[member][flank] = n_bounds

            z = p[2,:]
            R = np.linalg.norm(p[0:2,:], axis=0)
            self.zRfullvec = np.row_stack((z,R))

            self.interpolate_triplets_over_zr(member, flank, z, R, csi_theta_phi)

        if return_output:
            return p, n, z, R, csi_theta_phi

        
        return 

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
        if not not self.init_triplet['pinion']['concave']: # if the initial triplets are not empty
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
        waitbar = Waitbar(step = 0, title="Generating pinion concave flank")
        pin_pts_cnv, pin_proot_cnv, pin_nroot_cnv = self.sample_surface('pinion', 'concave', triplet_guess = trip_pin_cnv, FW_vec = FW_vec)
        waitbar.update(step = 25, message="Generating pinion convex flank")
        pin_pts_cvx, pin_proot_cvx, pin_nroot_cvx = self.sample_surface('pinion', 'convex', triplet_guess = trip_pin_cvx, FW_vec = FW_vec)
        waitbar.update(step = 50, message="Generating gear concave flank")
        gear_pts_cnv, gear_proot_cnv, gear_nroot_cnv = self.sample_surface('gear', 'concave', triplet_guess = trip_gear_cnv, FW_vec = FW_vec)
        waitbar.update(step = 75, message="Generating gear convex flank")
        gear_pts_cvx, gear_proot_cvx, gear_nroot_cvx = self.sample_surface('gear', 'convex', triplet_guess = trip_gear_cvx, FW_vec = FW_vec)
        
        waitbar.update(step = 85, message="PCA computation...")
        # Spacing line between two flanks based on hand
        hand = self.get_systemHand()
        nTp = self.get_Nteeth('pinion')
        nTg = self.get_Nteeth('gear')

        if hand.lower() == 'right':
            pin_pts_cnv = self.rot_z(2 * np.pi / nTp).dot(pin_pts_cnv[:3, :])
            gear_pts_cvx = self.rot_z(2 * np.pi / nTg).dot(gear_pts_cvx[:3, :])
        else:
            pin_pts_cvx = self.rot_z(2 * np.pi / nTp).dot(pin_pts_cvx[:3, :])
            gear_pts_cnv = self.rot_z(2 * np.pi / nTg).dot(gear_pts_cnv[:3, :])

        # Compute root and topland spacing
        self.rootSpacing['pinion'] = (pin_proot_cnv + pin_proot_cvx) * 0.5
        self.rootSpacing['gear'] = (gear_proot_cnv + gear_proot_cvx) * 0.5

        nF, nP, nfil = self.nFace, self.nProf, self.nFillet
        pin_pts_cnv = pin_pts_cnv.reshape(3, nF, nP + nfil - 1)[:, :, -1]
        pin_pts_cvx = pin_pts_cvx.reshape(3, nF, nP + nfil - 1)[:, :, -1]
        self.toplandSpacing['pinion'] = (pin_pts_cnv + pin_pts_cvx) * 0.5

        gear_pts_cnv = gear_pts_cnv.reshape(3, nF, nP + nfil - 1)[:, :, -1]
        gear_pts_cvx = gear_pts_cvx.reshape(3, nF, nP + nfil - 1)[:, :, -1]
        self.toplandSpacing['gear'] = (gear_pts_cnv + gear_pts_cvx) * 0.5

        # set dictionaries for modified designs (conjugate, easeOff, optimization, spreadBlade, topography)
        for attribute in ['Conjugate', 'EaseOff', 'Optimization', 'SpreadBlade', 'Topography']:
            setattr(self, f'identificationProblem{attribute}', {'designData': self.designData})

        if no_sync:
            return

        # PCA and synchronization computations
        for component, side in [('gear', 'drive'), ('gear', 'coast'), ('pinion', 'drive'), ('pinion', 'coast')]:
            self.compute_PCA(component, side, self.EPGalpha)

        waitbar.update(step = 90, message="Flank synchronization...")   
        self.compute_synch_angle()
        
        waitbar.update(step = 100, message="Done!")
        waitbar.close(delay=100)
        return
    
    def compute_PCA(self, member, side, EPGalpha):
        if side.lower() == 'drive' and member.lower() == 'pinion':
            flank = 'concave'
            other_flank = 'convex'
        else:
            flank = 'convex'
            other_flank = 'concave'

        zR = PCA_computation(self.designData, member, flank, EPGalpha, bounds_points, bounds_points_other)
        return
    
    # getters
    def get_Nteeth(self, member):
        common, sub_common = get_data_field_names(member, 'concave', fields = 'common')
        return self.designData[common][f'{sub_common}NTEETH']
    
    def get_systemHand(self):
        return self.designData['SystemData']['HAND']
    
    # setters

    # plot
    def plot(self, member = 'pinion', flank = 'concave'):
        
        points = self.surfPoints[member][flank]
        n_face = self.nFace

        F = ep.Figure(title=member+flank)
        X = points[0,:].reshape(n_face, -1, order = 'F')
        Y = points[1,:].reshape(n_face, -1, order = 'F')
        Z = points[2,:].reshape(n_face, -1, order = 'F')
        S = ep.surface(F, X,Y,Z)

        F.show()
        return
    
    def buildCasadiDerivatives(self):
        return
    
    def samplezR(self):
        return
    
    
    
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

    H.sampleSurface('gear', 'concave')
    H.sampleSurface('gear', 'convex')
    
    H.sampleSurface('pinion', 'concave')
    H.sampleSurface('pinion', 'convex')
    
    
    return
    
if __name__ == "__main__":
    main()
    