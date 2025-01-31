import numpy as np
import casadi as ca
from casadi.casadi import exp
from scipy.optimize import fsolve
from scipy.sparse import lil_matrix, bmat
from math import sqrt, pi, atan, cos, sin, acos, asin, tan
import screwCalculus as sc
from utils import *


def get_data_field_names(member, flank, fields = 'all'): # fileds = 'all', 'common', 'cutter', 'machine'
    if member.lower() == 'pinion' and flank.lower() == 'concave':
        cutterFieldName = 'PinionConcaveCutterData'
        subCutterFieldName = '' #'pinCutCnv'
        commonFieldName = 'PinionCommonData'
        subCommonFieldName = '' # 'pin'
        machineFieldName = 'PinionConcaveMachineSettings'
        subMachineFieldName = '' # 'pinCnv'
    elif member.lower() == 'pinion' and flank.lower() == 'convex':
        cutterFieldName = 'PinionConvexCutterData'
        subCutterFieldName = '' # 'pinCutCvx'
        commonFieldName = 'PinionCommonData'
        subCommonFieldName = '' # 'pin'
        machineFieldName = 'PinionConvexMachineSettings'
        subMachineFieldName = '' # 'pinCvx'
    elif member.lower() == 'gear' and flank.lower() == 'concave':
        cutterFieldName = 'GearConcaveCutterData'
        subCutterFieldName = '' # 'gearCutCnv'
        commonFieldName = 'GearCommonData'
        subCommonFieldName = '' # 'gear'
        machineFieldName = 'GearConcaveMachineSettings'
        subMachineFieldName = '' # 'gearCnv'
    elif member.lower() == 'gear' and flank.lower() == 'convex':
        cutterFieldName = 'GearConvexCutterData'
        subCutterFieldName = '' # 'gearCutCvx'
        commonFieldName = 'GearCommonData'
        subCommonFieldName = '' # 'gear'
        machineFieldName = 'GearConvexMachineSettings'
        subMachineFieldName = '' # 'gearCvx'

    if fields.lower() == 'all':
        return cutterFieldName, subCutterFieldName, commonFieldName, subCommonFieldName,machineFieldName, subMachineFieldName
    elif fields.lower() == 'common':
        return commonFieldName, subCommonFieldName
    elif fields.lower() == 'cutter':
        return cutterFieldName, subCutterFieldName
    elif fields.lower() == 'machine':
        return machineFieldName, subMachineFieldName

def assign_machine_par(data, member, flank):

    mainFieldName, subFieldName = get_data_field_names(member, flank, fields = 'machine')
    common_field, sub_common_field = get_data_field_names(member, flank, fields = 'common')

    S0    = data[mainFieldName][f'{subFieldName}RADIALSETTING']
    sig0  = data[mainFieldName][f'{subFieldName}TILTANGLE']
    z0    = data[mainFieldName][f'{subFieldName}SWIVELANGLE']
    E0    = data[mainFieldName][f'{subFieldName}BLANKOFFSET']
    gam0  = data[mainFieldName][f'{subFieldName}ROOTANGLE']
    D0    = data[mainFieldName][f'{subFieldName}MACHCTRBACK']
    B0    = data[mainFieldName][f'{subFieldName}SLIDINGBASE']
    q0    = data[mainFieldName][f'{subFieldName}CRADLEANGLE']
    m     = data[mainFieldName][f'{subFieldName}RATIOROLL']
    C2    = data[mainFieldName][f'{subFieldName}2C']
    D6    = data[mainFieldName][f'{subFieldName}6D']
    E24   = data[mainFieldName][f'{subFieldName}24E']
    F120  = data[mainFieldName][f'{subFieldName}120F']
    G720  = data[mainFieldName][f'{subFieldName}720G']
    H5040 = data[mainFieldName][f'{subFieldName}5040H']
    B1    = data[mainFieldName][f'{subFieldName}H1']
    B2    = data[mainFieldName][f'{subFieldName}H2']
    B3    = data[mainFieldName][f'{subFieldName}H3']
    B4    = data[mainFieldName][f'{subFieldName}H4']
    B5    = data[mainFieldName][f'{subFieldName}H5']
    B6    = data[mainFieldName][f'{subFieldName}H6']
    E1    = data[mainFieldName][f'{subFieldName}V1']
    E2    = data[mainFieldName][f'{subFieldName}V2']
    E3    = data[mainFieldName][f'{subFieldName}V3']
    E4    = data[mainFieldName][f'{subFieldName}V4']
    E5    = data[mainFieldName][f'{subFieldName}V5']
    E6    = data[mainFieldName][f'{subFieldName}V6']
    S1    = data[mainFieldName][f'{subFieldName}R1']
    S2    = data[mainFieldName][f'{subFieldName}R2']
    S3    = data[mainFieldName][f'{subFieldName}R3']
    S4    = data[mainFieldName][f'{subFieldName}R4']
    S5    = data[mainFieldName][f'{subFieldName}R5']
    S6    = data[mainFieldName][f'{subFieldName}R6']

    if member.lower() == 'gear' and data[common_field][f'{sub_common_field}GenType'].lower() == 'formate':
        H = data[mainFieldName][f'{subFieldName}Horizontal']
        V = data[mainFieldName][f'{subFieldName}Vertical']
        q0 = atan(V/H)
        S0 = sqrt(V**2 + H**2)
        q0 = q0*180/pi
        return np.array([
            [  S0,  0,  0,  0,  0,  0,  0,  0],     #joint Tool 2
            [sig0,  0,  0,  0,  0,  0,  0,  0],     #joint Tool 4
            [  z0,  0,  0,  0,  0,  0,  0,  0],     #joint Tool 3
            [  E0,  0,  0,  0,  0,  0,  0,  0],     #joint Gear 1
            [  B0,  0,  0,  0,  0,  0,  0,  0],     #joint Gear 2
            [  q0,  0,  0,  0,  0,  0,  0,  0],     #joint Tool 1
            [   0,  0,  0,  0,  0,  0,  0,  0],     #joint Gear 5
            [  D0,  0,  0,  0,  0,  0,  0,  0],     #joint Gear 4
            [gam0,  0,  0,  0,  0,  0,  0,  0]      #joint Gear 3
            ])

    return np.array([
        [  S0, S1, S2, S3,  S4,   S5,   S6,     0],     #joint Tool 2
        [sig0,  0,  0,  0,   0,    0,    0,     0],     #joint Tool 4
        [  z0,  0,  0,  0,   0,    0,    0,     0],     #joint Tool 3
        [  E0, E1, E2, E3,  E4,   E5,   E6,     0],     #joint Gear 1
        [  B0, B1, B2, B3,  B4,   B5,   B6,     0],     #joint Gear 2
        [  q0,  1,  0,  0,   0,    0,    0,     0],     #joint Tool 1
        [   0,  m, C2, D6, E24, F120, G720, H5040],     #joint Gear 5
        [  D0,  0,  0,  0,   0,    0,    0,     0],     #joint Gear 4
        [gam0,  0,  0,  0,   0,    0,    0,     0]      #joint Gear 3
        ])  

def manage_machine_par(member, systemHand, mode = 'gleason'):
    
    # coefficient settings
    if mode.lower() == 'gleason':
        cMat = np.array([
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040],
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040],
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040],
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040],
            [1, 1, 1/2, 1/6, 1/24, 1/120*1/1000, 1/720*1/1000, 1/5040],
            [1 ,1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040],
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720*1/1000, 1/5040*1/1000],
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040],
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040]
        ])
        cMat[[1,2,5,8], [0,0,0,0]] = cMat[[1,2,5,8], [0,0,0,0]]*pi/180

    # sign settings

    if member.lower() == 'pinion':
        if systemHand.lower() == 'left':
            signMat = np.array([
                [+1, +1, +1, +1, +1, +1, +1, +1], # R radial motion
                [-1, +1, +1, +1, +1, +1, +1, +1], # sig
                [-1, +1, +1, +1, +1, +1, +1, +1], # zeta
                [+1, +1, +1, +1, +1, +1, +1, +1], # E vertical motion
                [+1, +1, +1, +1, +1, +1, +1, +1], # B sliding base (helical motion)
                [-1, +1, +1, +1, +1, +1, +1, +1], # q cradle
                [+1, +1, -1, -1, -1, -1, -1, -1], # roll
                [+1, +1, +1, +1, +1, +1, +1, +1], # machCtr
                [+1, +1, +1, +1 ,+1, +1, +1, +1], #root
                ])
        else: # right hand
            signMat = np.array([
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [+1, -1, +1, -1, +1, -1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, -1, +1, -1, +1, -1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1 ,+1, +1, +1, +1]
            ])
    else: # gear
        if systemHand.lower() == 'left':
            signMat = np.array([
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [-1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [-1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, -1, -1, -1, -1, -1, -1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1 ,+1, +1, +1, +1]
            ])
        else: # right hand
            signMat = np.array([
                [+1, +1, +1, -1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, -1, +1, -1, +1, +1, +1],
                [+1, -1, +1, -1, +1, -1, +1, -1],
                [-1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, -1, +1, -1, +1, -1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1 ,+1, +1, +1, +1]
            ])
    return cMat, signMat

def assign_Blank_Par(data, member):
    _, _, commonFieldName, subCommonFieldName,\
           machineFieldName, subMachineFieldName = get_data_field_names(member, 'concave')
    
    c_mat, s_mat = manage_machine_par(member, data['SystemData']['HAND'])

    A0 = data[commonFieldName][f'{subCommonFieldName}OUTERCONEDIST']
    Fw = data[commonFieldName][f'{subCommonFieldName}FACEWIDTH']
    beta = data[commonFieldName][f'{subCommonFieldName}SPIRALANGLE']

    # front/back cones
    deltaf = data[commonFieldName][f'{subCommonFieldName}FRONTANGLE']
    deltab = data[commonFieldName][f'{subCommonFieldName}BACKANGLE']

    # pitch cone
    gammaP = data[commonFieldName][f'{subCommonFieldName}PITCHANGLE']
    dpa = data[commonFieldName][f'{subCommonFieldName}PITCHAPEX']

    # face cone
    dfa = data[commonFieldName][f'{subCommonFieldName}FACEAPEX']
    gammaF = data[commonFieldName][f'{subCommonFieldName}FACEANGLE']
    
    # root cone
    dra = data[commonFieldName][f'{subCommonFieldName}ROOTAPEX']
    gammaR = data[machineFieldName][f'{subMachineFieldName}ROOTANGLE']

    # base cone
    gammaB = data[commonFieldName][f'{subCommonFieldName}BASECONEANGLE']
    dba = data[commonFieldName][f'{subCommonFieldName}BASECONEAPEX']
    
    RA = data[commonFieldName][f'{subCommonFieldName}ShaftRA']
    deltaf = deltaf*pi/180
    deltab = deltab*pi/180
    gammaP = gammaP*pi/180
    gammaF = gammaF*pi/180
    gammaR = gammaR*c_mat[8, 0]*s_mat[8, 0]
    gammaB = gammaB*pi/180
    beta = beta*pi/180 
    return (A0, Fw, beta, deltaf, deltab, gammaP, dpa, gammaF, dfa, gammaR, dra, gammaB, dba, RA)

def assign_tool_par(data, member, flank, stfFlank = None):
    
    alphaFlank = 0
    rho_straight = 2e6
    cutterFieldName, subCutterFieldName, _, _, machineFieldName, subMachineFieldName =\
          get_data_field_names(member, flank)
    
    Rp = data[cutterFieldName][f'{subCutterFieldName}POINTRADIUS']
    alphap = data[cutterFieldName][f'{subCutterFieldName}BLADEANGLE']
    rhof = data[cutterFieldName][f'{subCutterFieldName}EDGERADIUS']
    if data[cutterFieldName][f'{subCutterFieldName}TYPE'].lower() == 'curved':
        rho = data[cutterFieldName][f'{subCutterFieldName}RHO']
    else:
        rho = rho_straight

    if data[cutterFieldName][f'{subCutterFieldName}TopremTYPE'].lower() == 'blended':
        rhoTop = data[cutterFieldName][f'{subCutterFieldName}TopremRADIUS']
        stftop = data[cutterFieldName][f'{subCutterFieldName}TopremDEPTH']

    elif data[cutterFieldName][f'{subCutterFieldName}TopremTYPE'].lower() == 'straight':
        alphaTop = data[cutterFieldName][f'{subCutterFieldName}TopremANGLE']
        stftop = data[cutterFieldName][f'{subCutterFieldName}TopremDEPTH']      
        rhoTop = rho_straight

    else: # None (feature disabled)
        rhoTop = 100 # initialize to a reasonable blend radius in case an identification activates the feature
        Czblade = rho*sin(alphap*pi/180)
        theta_iniz_blade = asin((Czblade + rhof)/(rho + rhof))
      
        stftop = rhof - rhof*sin(theta_iniz_blade)
        alphaTop = 0
        if data[cutterFieldName][f'{subCutterFieldName}TopremANGLE'] != 0:
            msgbox("Tool with angle offset between tip & blade (toprem angle) not implemented", 'Error', 1)

    if data[cutterFieldName][f'{subCutterFieldName}FlankremTYPE'].lower() == 'blended':
        rhoFlank = data[cutterFieldName][f'{subCutterFieldName}FlankremRADIUS']
        stfFlank = data[cutterFieldName][f'{subCutterFieldName}FlankremDEPTH']
        

    elif data[cutterFieldName][f'{subCutterFieldName}FlankremTYPE'].lower() == 'straight':
        alphaFlank = data[cutterFieldName][f'{subCutterFieldName}FlankremANGLE']
        stfFlank = data[cutterFieldName][f'{subCutterFieldName}FlankremDEPTH']      
        rhoFlank = rho_straight

    else: # None (feature disabled)
        rhoFlank = rho # initialize to a reasonable blend radius in case an identification activates the feature
        if stfFlank == None:
            stfFlank = rho/3
        alphaTop = 0
        if data[cutterFieldName][f'{subCutterFieldName}FlankremANGLE'] != 0:
            msgbox("Tool with angle offset between blade and flankrem (flankrem angle) not implemented", 'Error', 1)

    return (Rp,rho,rhof,rhoTop,rhoFlank,alphap,stfFlank,stftop,alphaTop,alphaFlank)

def initial_guesses(A0, Fw, gammaP, gammaF, member, hand, q0, z0, beta, machCtr, gammaR, S0, Rp, RA):
    """
    A0 outer cone distance
    Fw facewidth
    gammaP pitchangle
    gammaF faceangle
    member 
    hand system hand
    q0 cradle angle
    z0 swivel angle (some Gleason designs have this =/= 0 even if tilt angle is zero)
    beta spiral angle
    machCtr machine center to back setting
    gammaR root angle
    S0 machine radial setting
    Rp tool pointradius
    RA radial distance of the root-toe point
    """
    xF = (A0-0.5*Fw)*sin(gammaF)
    yF = (A0-0.5*Fw)*cos(gammaF)
    xP = (A0-0.5*Fw)*sin(gammaP)
    yP = (A0-0.5*Fw)*cos(gammaP)
    csiout = 2*np.linalg.norm([xF - xP,yF - yP])
    if csiout < 0.01:
        csiout = 5

    thetag = acos( ((machCtr+(A0-1.2*Fw)*cos(gammaP))*cos(gammaR) + RA*sin(gammaR) - S0*cos(q0))/Rp) # angle on the cutter
    
    if member.lower() == 'pinion' and hand.lower() == 'left':
        thetaout = (thetag - q0) + z0
    elif member.lower() == 'pinion' and hand.lower() == 'right':
        thetaout = -(thetag + q0) + z0
    elif member.lower() == 'gear' and hand.lower() == 'left': # right gear
        thetaout = -(thetag + q0) + z0
    elif member.lower() == 'gear' and hand.lower() == 'right':
        thetaout = (thetag - q0) + z0
    
    # try for root cone initial guess
    phiout = 0
    csiout = 0
    return (csiout, thetaout, phiout)

def initial_guess_from_data(data, member, flank): # wrapper for initialguesses() to provide the designData instead of all the parameters
    hand = data['SystemData']['HAND']
    toolvec = assign_tool_par(data, member, flank)
    RawMachinePar = assign_machine_par(data, member, flank)
    cMat, signMat = manage_machine_par(member, hand)
    MachineParMatrix = cMat * signMat * RawMachinePar
    m = RawMachinePar[6, 1]
    MachineParMatrix[6, 2:] = MachineParMatrix[6, 2:] * m
    q0 = MachineParMatrix[5, 0]
    z0 = MachineParMatrix[2, 0]
    sig0 = MachineParMatrix[1, 0]
    machCtr = MachineParMatrix[7, 0]
    gammaroot = MachineParMatrix[8, 0]
    S0 = MachineParMatrix[0, 0]  # radial setting
    A0, Fw, beta, deltaf, deltab, gammaP, dpa, gammaF, dfa, gammaR, dra, gammaB, dba, RA = assign_Blank_Par(data, member)
    triplet_guess = initial_guesses(A0, Fw, gammaP, gammaF, member, hand, q0, z0, beta, machCtr, gammaroot, S0, toolvec[0], RA)
    
    return triplet_guess

def initialize_design_data(): # return designData
    commonFieldName, subCommonFieldName  = get_data_field_names('gear', 'concave', fields = 'common')
    commonFieldName2, subCommonFieldName2  = get_data_field_names('pinion', 'concave', fields = 'common')
    commonStrings = [commonFieldName, commonFieldName2]
    subCommonStrings = [subCommonFieldName, subCommonFieldName2]
    cutterStrings = []
    subCutterStrings = []
    machineStrings = []
    subMachineStrings = []

    for member in ['gear', 'pinion']:
        for flank in ['concave', 'convex']:
            field, sub_field  = get_data_field_names(member, flank, fields = 'cutter')
            cutterStrings.append(field)
            subCutterStrings.append(sub_field)

    for member in ['gear', 'pinion']:
        for flank in ['concave', 'convex']:
            field, sub_field  = get_data_field_names(member, flank, fields = 'machine')
            machineStrings.append(field)
            subMachineStrings.append(sub_field)

    SystemData = {
        'HAND': 0,
        'shaft_angle': 0,
        'ratio': 0,
        'hypoidOffset': 0,
        'E': [],
        'P': [],
        'G': [],
        'alpha': [],
        }
    
    Common = [1,2]
    Machine = [1,2,3,4]
    Cutter = [1,2,3,4]

    for ii in range(0, 2):
        Common[ii] =  {
            f'{subCommonStrings[ii]}GenType' : 0,
            f'{subCommonStrings[ii]}NTEETH' : 0,
            f'{subCommonStrings[ii]}SPIRALANGLE' : 0,
            f'{subCommonStrings[ii]}OUTERCONEDIST' : 0,
            f'{subCommonStrings[ii]}FACEWIDTH' : 0,
            f'{subCommonStrings[ii]}FACEANGLE' : 0,
            f'{subCommonStrings[ii]}BACKANGLE' : 0,
            f'{subCommonStrings[ii]}FRONTANGLE' : 0,
            f'{subCommonStrings[ii]}PITCHANGLE' : 0,
            f'{subCommonStrings[ii]}BASECONEANGLE' : 0,
            f'{subCommonStrings[ii]}PITCHAPEX' : 0,
            f'{subCommonStrings[ii]}FACEAPEX' : 0,
            f'{subCommonStrings[ii]}ROOTAPEX' : 0,
            f'{subCommonStrings[ii]}BASECONEAPEX' : 0,
        }

    for ii in range(0, 4):
        Machine[ii] = {
            f'{subMachineStrings[ii]}RADIALSETTING': 0,
            f'{subMachineStrings[ii]}CRADLEANGLE': 0,
            f'{subMachineStrings[ii]}TILTANGLE': 0,
            f'{subMachineStrings[ii]}SWIVELANGLE': 0,
            f'{subMachineStrings[ii]}BLANKOFFSET': 0,
            f'{subMachineStrings[ii]}ROOTANGLE': 0,
            f'{subMachineStrings[ii]}MACHCTRBACK': 0,
            f'{subMachineStrings[ii]}SLIDINGBASE': 0,
            f'{subMachineStrings[ii]}RATIOROLL': 0,
            f'{subMachineStrings[ii]}2C': 0,
            f'{subMachineStrings[ii]}6D': 0,
            f'{subMachineStrings[ii]}24E': 0,
            f'{subMachineStrings[ii]}120F': 0,
            f'{subMachineStrings[ii]}720G': 0,
            f'{subMachineStrings[ii]}5040H': 0,
            f'{subMachineStrings[ii]}H1': 0,
            f'{subMachineStrings[ii]}H2': 0,
            f'{subMachineStrings[ii]}H3': 0,
            f'{subMachineStrings[ii]}H4': 0,
            f'{subMachineStrings[ii]}H5': 0,
            f'{subMachineStrings[ii]}H6': 0,
            f'{subMachineStrings[ii]}V1': 0,
            f'{subMachineStrings[ii]}V2': 0,
            f'{subMachineStrings[ii]}V3': 0,
            f'{subMachineStrings[ii]}V4': 0,
            f'{subMachineStrings[ii]}V5': 0,
            f'{subMachineStrings[ii]}V6': 0,
            f'{subMachineStrings[ii]}R1': 0,
            f'{subMachineStrings[ii]}R2': 0,
            f'{subMachineStrings[ii]}R3': 0,
            f'{subMachineStrings[ii]}R4': 0,
            f'{subMachineStrings[ii]}R5': 0,
            f'{subMachineStrings[ii]}R6': 0,
        }

        Cutter[ii] =  {
            f'{subCutterStrings[ii]}POINTRADIUS' : 0,
            f'{subCutterStrings[ii]}BLADEANGLE' : 0,
            f'{subCutterStrings[ii]}EDGERADIUS' : 0,
            f'{subCutterStrings[ii]}TYPE' : "STRAIGHT",
            f'{subCutterStrings[ii]}TopremTYPE' : "NONE",
            f'{subCutterStrings[ii]}FlankremTYPE' : "NONE",
            f'{subCutterStrings[ii]}RHO' : 0,
            f'{subCutterStrings[ii]}TopremDEPTH' : 0,
            f'{subCutterStrings[ii]}TopremRADIUS' : 0,
            f'{subCutterStrings[ii]}TopremANGLE' : 0,
            f'{subCutterStrings[ii]}FlankremDEPTH' : 0,
            f'{subCutterStrings[ii]}FlankremRADIUS' : 0,
            f'{subCutterStrings[ii]}FlankremANGLE' : 0
        }

    designData = {'SystemData' : SystemData,
                commonStrings[0] : Common[0],
                commonStrings[1] : Common[1],
                machineStrings[0]: Machine[0],
                machineStrings[1]: Machine[1],
                machineStrings[2]: Machine[2],
                machineStrings[3]: Machine[3],
                cutterStrings[0]: Cutter[0],
                cutterStrings[1]: Cutter[1],
                cutterStrings[2]: Cutter[2],
                cutterStrings[3]: Cutter[3]}

    return designData

def machine_settings_index():
    
    dict = {
        'RADIALSETTING': 0,
        'TILTANGLE': 1,
        'SWIVELANGLE': 2,
        'BLANKOFFSET': 3,
        'SLIDINGBASE': 4,
        'CRADLEANGLE': 5,
        'INDEXANGLE': 6,
        'MACHCTRBACK': 7,
        'ROOTANGLE': 8,
        'R1': 9, 'R2': 18, 'R3': 27, 'R4': 36, 'R5': 45, 'R6': 54, 'R7': 63,
        'SIGMA1': 10, 'SIGMA2': 19, 'SIGMA3': 28, 'SIGMA4': 37, 'SIGMA5': 46, 'SIGMA6': 55, 'SIGMA7': 64,
        'ZETA1': 11, 'ZETA2': 20, 'ZETA3': 29, 'ZETA4': 38, 'ZETA5': 47, 'ZETA6': 56, 'ZETA7': 65,
        'V1': 12, 'V2': 21, 'V3': 30, 'V4': 39, 'V5': 48, 'V6': 57, 'V7': 66,
        'H1': 13, 'H2': 22, 'H3': 31, 'H4': 40, 'H5': 49, 'H6': 58, 'H7': 67,
        'Q1': 14, 'Q2': 23, 'Q3': 32, 'Q4': 41, 'Q5': 50, 'Q6': 59, 'Q7': 68,
        'RATIOROLL': 15, 'C2': 24, 'D6': 33, 'E24': 42, 'F120': 51, 'G720': 60, 'H5040': 69,
        'D1': 16, 'D2': 25, 'D3': 34, 'D4': 43, 'D5': 52, 'D6': 61, 'D7': 70,
        'GAMMA1': 17, 'GAMMA2': 26, 'GAMMA3': 35, 'GAMMA4': 44, 'GAMMA5': 53, 'GAMMA6': 62, 'GAMMA7': 71,
        'POINTRADIUS': 72, 'SPHERICALRADIUS': 73, 'EDGERADIUS': 74, 'TOPREMRADIUS' : 75, 'FLANKREMRADIUS' : 76,
        'BLADEANGLE': 77, 'TOPREMDEPTH': 78, 'FLANKREMDEPTH': 79, 'TOPREMANGLE': 80, 'FLANKREMANGLE': 81
    }

    return dict

def interpolated_triplets_zR(interpolant, z, R):
    """
    Generate interpolated triplets for given z and R using provided interpolant functions.
    Parameters:
    interpolant (dict): A dictionary containing interpolation functions for 'csi', 'theta', and 'phi'.
    z (array-like): An array of z values.
    R (array-like): An array of R values.
    Returns:
    numpy.ndarray: A 2D array of shape (3, max(z.shape)) containing interpolated values for 'csi', 'theta', and 'phi'.
    z and R need to be either 1D arrays or 2D arrays with the same shape.
    """

    triplets = np.full((3, max(z.shape)), np.nan)
    triplets[0,:] = interpolant['csi'](z, R)
    triplets[1,:] = interpolant['theta'](z, R)
    triplets[2,:] = interpolant['phi'](z, R)

    return triplets

def main(): # main function for scripting debug
    # designData = initialize_design_data()
    # dictprint(designData)
    # filename = r'C:\Users\egrab\Desktop\designData.txt'
    # print(filename)
    # dict_to_file(designData, filename)
    import subprocess
    import time 

    for ii in range(0,4):
        subprocess.Popen(["python.exe"], shell=True) 

def rz_to_grid(zstar, Rstar, zRBounds, method = 1):
    """
    Maps given z and R coordinates to a grid using shape functions.
    Parameters:
    zstar (array-like): The z-coordinates to be mapped.
    Rstar (array-like): The R-coordinates to be mapped.
    zRBounds (array-like): A 2D array containing the boundary coordinates for z and R (the four corners).
    options (dict, optional): A dictionary containing options for the mapping method. 
                              Default is {'type': 1}. If 'type' is 1, it maps to a grid with bounds between -1 and 1.
                              If 'type' is 2, it maps to a grid with bounds between 0 and 1.
    Returns:
    tuple: A tuple containing two arrays (u, v) which are the mapped coordinates on the grid.
    """

    R1, R2, R3, R4 = zRBounds[:, 1]
    z1, z2, z3, z4 = zRBounds[:, 0]

    # Define shape functions and their derivatives based on the method
    if method == 1:
        N1 = lambda u, v: 0.25 * (1 - v) * (1 - u)
        N2 = lambda u, v: 0.25 * (1 - v) * (1 + u)
        N3 = lambda u, v: 0.25 * (1 + v) * (1 + u)
        N4 = lambda u, v: 0.25 * (1 + v) * (1 - u)

        N1_du = lambda u, v: -0.25 * (1 - v)
        N2_du = lambda u, v: +0.25 * (1 - v)
        N3_du = lambda u, v: +0.25 * (1 + v)
        N4_du = lambda u, v: -0.25 * (1 + v)

        N1_dv = lambda u, v: -0.25 * (1 - u)
        N2_dv = lambda u, v: -0.25 * (1 + u)
        N3_dv = lambda u, v: +0.25 * (1 + u)
        N4_dv = lambda u, v: +0.25 * (1 - u)
    else:
        N1 = lambda u, v: (v - 1) * (u - 1)
        N2 = lambda u, v: u * (1 - v)
        N3 = lambda u, v: u * v
        N4 = lambda u, v: v * (1 - u)

        N1_du = lambda u, v: (v - 1)
        N2_du = lambda u, v: (1 - v)
        N3_du = lambda u, v: +v
        N4_du = lambda u, v: -v

        N1_dv = lambda u, v: (u - 1)
        N2_dv = lambda u, v: -u
        N3_dv = lambda u, v: +u
        N4_dv = lambda u, v: (1 - u)

    r, c = np.shape(np.atleast_1d(zstar))

    def obj(x, output = 'fun'):
        u_val = x[:len(x) // 2]
        v_val = x[len(x) // 2:]

        N1_ev = N1(u_val, v_val)
        N2_ev = N2(u_val, v_val)
        N3_ev = N3(u_val, v_val)
        N4_ev = N4(u_val, v_val)

        fun = np.concatenate([
            N1_ev * z1 + N2_ev * z2 + N3_ev * z3 + N4_ev * z4 - zstar.flatten(),
            N1_ev * R1 + N2_ev * R2 + N3_ev * R3 + N4_ev * R4 - Rstar.flatten()
        ])

        if output.lower() == 'fun':
            return fun
        
        N1_du_ev = N1_du(u_val, v_val)
        N2_du_ev = N2_du(u_val, v_val)
        N3_du_ev = N3_du(u_val, v_val)
        N4_du_ev = N4_du(u_val, v_val)

        N1_dv_ev = N1_dv(u_val, v_val)
        N2_dv_ev = N2_dv(u_val, v_val)
        N3_dv_ev = N3_dv(u_val, v_val)
        N4_dv_ev = N4_dv(u_val, v_val)
        
        derU = np.concatenate([
            N1_du_ev * z1 + N2_du_ev * z2 + N3_du_ev * z3 + N4_du_ev * z4,
            N1_du_ev * R1 + N2_du_ev * R2 + N3_du_ev * R3 + N4_du_ev * R4
        ])

        derV = np.concatenate([
            N1_dv_ev * z1 + N2_dv_ev * z2 + N3_dv_ev * z3 + N4_dv_ev * z4,
            N1_dv_ev * R1 + N2_dv_ev * R2 + N3_dv_ev * R3 + N4_dv_ev * R4
        ])

        block1 = lil_matrix((r, r))
        block2 = block1.copy()
        block3 = block1.copy()
        block4 = block1.copy()

        block1.setdiag(derU[:len(derU) // 2])
        block2.setdiag(derU[len(derU) // 2:])
        block3.setdiag(derV[:len(derV) // 2])
        block4.setdiag(derV[len(derV) // 2:])

        Jac = bmat([[block1, block3], [block2, block4]])

        return Jac

    # Initial guess
    guessU = np.zeros(r)
    guessV = np.zeros(r)

    sol = fsolve(lambda x: obj(x, output='fun'), np.concatenate([guessU, guessV]), xtol=1e-8, fprime=lambda x: obj(x, output = 'jac'))

    u = sol[:len(sol) // 2]
    v = sol[len(sol) // 2:]

    return u, v

def grid_to_rz(u, v, zR_bounds, method = 1):
    # Define shape functions and their derivatives based on the method
    if method == 1: #shape functions defined for -1 to 1
        N1 = 0.25 * (1 - v) * (1 - u)
        N2 = 0.25 * (1 - v) * (1 + u)
        N3 = 0.25 * (1 + v) * (1 + u)
        N4 = 0.25 * (1 + v) * (1 - u)
    else: #shape functions defined for 0 to 1
        N1 = (v - 1) * (u - 1)
        N2 = u * (1 - v)
        N3 = u * v
        N4 = v * (1 - u)

    R1, R2, R3, R4 = zR_bounds[:, 1]
    z1, z2, z3, z4 = zR_bounds[:, 0]

    z = N1 * z1 + N2 * z2 + N3 * z3 + N4 * z4
    R = N1 * R1 + N2 * R2 + N3 * R3 + N4 * R4

    return z, R

if __name__ == '__main__':
    main()