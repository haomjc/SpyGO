import numpy as np
import casadi as ca
from casadi.casadi import exp
from scipy.optimize import fsolve
from scipy.sparse import lil_matrix, bmat
from math import sqrt, pi, atan, cos, sin, acos, asin, tan

import screwCalculus as sc
from solvers import *

from general_utils import *
from hypoid.main.dataclasses import DesignData

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

def initial_guess_from_data(data:DesignData, member, flank): # wrapper for initialguesses() to provide the designData instead of all the parameters
    hand = data.system_data.hand.lower()
    toolvec = data.extract_tool_settings(member, flank)
    RawMachinePar = data.extract_machine_settings_matrix(member, flank)
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
    A0, Fw, beta, deltaf, deltab, gammaP, dpa, gammaF, dfa, gammaR, dra, gammaB, dba, RA = data.extract_blank_settings(member)
    triplet_guess = initial_guesses(A0, Fw, gammaP, gammaF, member, hand, q0, z0, beta, machCtr, gammaroot, S0, toolvec[0], RA)
    
    return triplet_guess

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

    # check if z and R are pure numbers and turn them to arrays
    z = np.atleast_1d(z)
    R = np.atleast_1d(R)

    triplets = np.full((3, max(z.shape)), np.nan)
    triplets[0,:] = interpolant['csi'](z, R)
    triplets[1,:] = interpolant['theta'](z, R)
    triplets[2,:] = interpolant['phi'](z, R)

    return triplets

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

def AGMAcomputationHypoid(Hand, taper, initialConeData, toothInitialData, Method = 1, rc0 = None, GearGenType = "Generated", gearTilt = 0):
    
    rc0Flag = True
    if rc0 is None or rc0 is np.nan:
        rc0 = 0
        rc0Flag = None

    uniformToothCoeff = 1

    # extract initial data for cone parameters determination
    SIGMA = initialConeData["SIGMA"]*pi/180         # shaft angle
    a = initialConeData["a"]                        # hypoid offset
    z1 = initialConeData["z1"]                      # pinion teeth
    z2 = initialConeData["z2"]                      # gear teeth
    u = initialConeData["u"]                        # transmission ratio
    de2 = initialConeData["de2"]                    # gear outer diameter
    b2 = initialConeData["b2"]                      # facewidth
    betam1 = initialConeData["betam1"]*pi/180       # pinion spiral angle

    # extract initial data for tooth dimensions
    alphadD = toothInitialData["alphaD"]*pi/180     # nominal design pressure angle drive side
    alphadC =  toothInitialData["alphaC"]*pi/180    # nominal design pressure angle coast side
    falphalim =  toothInitialData["falphalim"]      # influence factor of limit pressure angle
    khap = toothInitialData["khap"]                 # addendum factor
    khfp = toothInitialData["khfp"]                 # dedendum factor
    xhm1 = toothInitialData["xhm1"]                 # profile shift coefficient
    jen  = toothInitialData["jen"]                  # outer normal backlash
    xsmn = toothInitialData["xsmn"]                 # thickness modification coefficient

    DeltaSIGMA = SIGMA-pi/2                

    # method 0 (spiral bevel gears)
    delta1 = atan(sin(SIGMA)/(cos(SIGMA) + u))    # pinion pith angle
    delta2 = SIGMA - delta1                       # gear pitch angle
    Re2 = de2/(2.*sin(delta2))                    # outer cone distance (equal for both members)
    Re1 = Re2                                     # outer cone distance (equal for both members)
    Rm2 = Re2 - b2/2                              # mean cone distance (equal for both members)
    Rm1 = Rm2 
    betam2 = betam1                              # spiral angle (equal for both members)
    cbe2 = 0.5                                   # face width factor (equal for both members)

    if Method == 1: # hypoid gears
        betaDelta1 = betam1
        DeltaSIGMA = SIGMA - pi/2
        deltaint2 = atan(u*cos(DeltaSIGMA)/2/(1-u*sin(DeltaSIGMA)))
        rmpt2 = (de2 - b2*sin(deltaint2))/2
        epsiprime = asin(a*sin(deltaint2)/rmpt2)
        K1 = tan(betaDelta1)*sin(epsiprime) + cos(epsiprime)
        rmn1 = rmpt2*K1/u
        ni0 = atan(a/(rmpt2*(tan(deltaint2)*cos(DeltaSIGMA) - sin(DeltaSIGMA)) + rmn1)) 
        
        # starting guess value for the iterative process of the gear offset angle in axial plane

        # intermediate pinion offset angle in axial plane
        eps2 = lambda ni: (a - rmn1*sin(ni))/rmpt2 
        # intermediate pinion pitch angle
        deltaint1 = lambda ni: atan(sin(ni)/(tan(eps2(ni))*cos(DeltaSIGMA)) + tan(DeltaSIGMA)*cos(ni)) 
        # intermediate pinion offset angle in pitch plane
        eps2prime = lambda ni: asin( sin(eps2(ni))*cos(DeltaSIGMA)/cos(deltaint1(ni)) ) 
        # intermediate pinion mean spirla angle
        betamint1 = lambda ni: atan( (K1 - cos(eps2prime(ni)) )/sin(eps2prime(ni)) ) 
        # increment in hypoid dimension factor
        deltaK = lambda ni: sin(eps2prime(ni))*( tan(betaDelta1) - tan(betamint1(ni)) )
        # pin mean radius increment
        Deltarmpt1 = lambda ni: rmpt2*deltaK(ni)/u 
        # pinion offset angle in axial plane
        eps1 = lambda ni: asin(sin(eps2(ni)) - Deltarmpt1(ni)/rmpt2*sin(ni)) 
        # pinion pitch angle
        delta1 = lambda ni: atan(sin(ni)/(tan(eps1(ni))*cos(DeltaSIGMA)) + tan(DeltaSIGMA)*cos(ni))
        # pinion offset angle in pitch plane
        eps1prime = lambda ni: asin( sin(eps1(ni))*cos(DeltaSIGMA)/cos(delta1(ni)) )
        # pinion spiral angle
        betam1 = lambda ni: atan( (K1 + deltaK(ni) - cos(eps1prime(ni)))/sin(eps1prime(ni)) )
        # gear spiral angle
        betam2 = lambda ni: betam1(ni) - eps1prime(ni)
        # gear pitch angle
        delta2 = lambda ni: atan( sin(eps1(ni))/(tan(ni)*cos(DeltaSIGMA)) + cos(eps1(ni))*tan(DeltaSIGMA) )
        # gear mean cone distance
        Rm2 = lambda ni: rmpt2/(sin(delta2(ni)))
        # pinion mean cone distance
        Rm1 = lambda ni: (rmn1 + Deltarmpt1(ni))/(sin(delta1(ni)))
        # mean pinion radius
        rmpt1 = lambda ni: Rm1(ni)*sin(delta1(ni))
        # limit pressure angle
        alphalim = lambda ni: atan( -tan(delta1(ni))*tan(delta2(ni))/cos(eps1prime(ni))*( (Rm1(ni)*sin(betam1(ni)) - Rm2(ni)*sin(betam2(ni))) / (Rm1(ni)*tan(delta1(ni)) + Rm2(ni)*tan(delta2(ni))) ) ) 

        rc0_user = rc0
        rc0 = lambda ni: rc0_user # user assigned rc0 value

        # else use suggested mean cutter radius if user didn't specify any
        if rc0Flag == None:
            match taper.lower():
                case "standard":
                    rc0 = lambda ni: (Rm2(ni) + 1.1*Rm2(ni)*sin(betam2(ni)))/2
                    rc0 = lambda ni: Rm1(ni)*sin(betam1(ni))*1.5
                case "uniform":
                    rc0 = lambda ni: uniformToothCoeff*Rm2(ni)*sin(betam2(ni))
                case "duplex":
                    rc0 = lambda ni: (Rm2(ni) + 1.1*Rm2(ni)*sin(betam2(ni)))/2
                case "trl":
                    rc0 = lambda ni: (Rm2(ni) + 1.1*Rm2(ni)*sin(betam2(ni)))/2

        rhombeta = lambda ni: rc0(ni)

        rholim = lambda ni: 1/cos(alphalim(ni))*(tan(betam1(ni)) - tan(betam2(ni)))/(-tan(alphalim(ni))*(tan(betam1(ni))/(Rm1(ni)*tan(delta1(ni))) + tan(betam2(ni))/(Rm2(ni)*tan(delta2(ni)))) + 1/(Rm1(ni)*cos(betam1(ni))) - 1/(Rm2(ni)*cos(betam2(ni))))

        Delta = lambda ni: abs(rhombeta(ni)/rholim(ni) - 1)
        ni = 0.0875
        #ni0
        if Delta(ni0) > 0.01: # check if ni0 value satisfies the constraints
            Delta = lambda ni: abs(rhombeta(ni)/rholim(ni) - 1) - 0.005
            ni, opts , flag, msg = fsolve(Delta, ni0, full_output = True)
            if flag != 1:
                raise ValueError('Iterative algorithm did not converge properly!')
            
        rc0 = rc0(ni)
        Rm1 = Rm1(ni) # pinion mean cone distance
        Rm2 = Rm2(ni) # gear mean cone distance
        betam2 = betam2(ni) # gear spiral angle
        betam1 = betam1(ni) # pinion spiral angle
        delta2 = delta2(ni) # gear pitch angle
        delta1 = delta1(ni) # pinion pitch angle
        cbe2 = (de2/2/sin(delta2) - Rm2)/b2 # facewidth factor

    # Determination of basic data
    dm1 = 2*Rm1*sin(delta1)# pinion mean pitch diameter
    dm2 = 2*Rm2*sin(delta2)# gear mean pitch diameter
    zetam = asin(2*a/(dm2 + dm1*cos(delta2)/cos(delta1))) # offset angle in pinion axial plane
    zetamp = asin(sin(zetam)*sin(DeltaSIGMA)/cos(delta1)) # offset angle in the pitch plane
    ap = Rm2*sin(zetamp) # offset in pitch plane
    mmn = 2*Rm2*sin(delta2)*cos(betam2)/z2 # mean normal module
    alphalim = -atan( tan(delta1)*tan(delta2)/cos(zetamp)*( (Rm1*sin(betam1) - Rm2*sin(betam2)) / (Rm1*tan(delta1) + Rm2*tan(delta2)) ) ) # limit pressure angle
    alphanD = alphadD + falphalim*alphalim #generated normal pressure angle drive side
    alphanC = alphadC - falphalim*alphalim # generated normla pressure angle coast side
    alphaeD = alphanD - alphalim # effective pressure angle drive side (useless ?)
    alphaeC = alphanC + alphalim # effective pressure angle coast side
    alphan = (alphanD + alphanC)/2 # mean normal pressure angle
    Re2 = Rm2 + cbe2*b2 # outer pitch cone distance
    Ri2 = Re2 - b2 # inner pitch cone distance
    de2 = 2*Re2*sin(delta2) # gear outer pitch diameter
    di2 = 2*Ri2*sin(delta2) # gear inner pitch diameter
    met = de2/z2 # outer transverse module
    be2 = Re2 - Rm2 # gear facewidth from calculation point to outside
    bi2 = Rm2 - Ri2 # gear facewidth from calculation point to inside
    tzm2 = dm1*sin(delta2)/(2*cos(delta1)) - 0.5*cos(zetam)*tan(DeltaSIGMA)*(dm2 + dm1*cos(delta2)/cos(delta1)) # crossing point to calculation point along gear axis
    tzm1 = dm2/2*cos(zetam)*cos(DeltaSIGMA) - tzm2*sin(DeltaSIGMA) # crossing point to calculation point along pinion axis
    tz1 =  Rm1*cos(delta1) - tzm1 # pitch apex beyond crossing point along axis, pinion
    tz2 = Rm2*cos(delta2) - tzm2 # pitch apex beyond crossing point along axis, gear

    # determination of tooth depth at calculation point
    hmw = 2*mmn*khap # mean working depth
    ham2 = mmn*(khap - xhm1) # gear mean addendum
    hfm2 = mmn*(khfp + xhm1) # gear mean dedendum
    ham1 = mmn*(khap + xhm1) # pinion mean addendum
    hfm1 = mmn*(khfp - xhm1) # pinion mean dedendum
    c = mmn*(khfp - khap) # clearance
    hm = ham1 + hfm1 # mean whole depth (equal for both pinion and gear)
    hm = mmn*(khap + khfp) # mean whole depth (same formula as spur gears : tooth height = module*(1+1.25))
    
    # determination of dedendum angles
    match taper:
        case "Standard":
            sumthetafS =  atan(hfm1/Rm2) +  atan(hfm2/Rm2)  # sum dedendum angles
            thetaa2 =  atan(hfm1/Rm2)                     # addendum gear
            thetaf2 = sumthetafS - thetaa2               # dedendum gear
        case "Uniform":
            sumthetafU = 0 
            thetaa2 = 0 
            thetaf2 = 0 
        case "Duplex":
            sumthetafC = (pi*met/Re2/tan(alphan)/cos(betam2))*(1 - Rm2*sin(betam2)/rc0) 
            thetaa2 = sumthetafC*ham2/hmw 
            thetaf2 = sumthetafC - thetaa2 
        case "TRL":
            sumthetafM = min(   atan(hfm1/Rm2) +  atan(hfm2/Rm2)  ,  (pi/2*met/Re2/tan(alphan)/cos(betam2))*(1 - Rm2*sin(betam2)/rc0)  ) 
            thetaa2 = sumthetafM*ham2/hmw 
            thetaf2 = sumthetafM - thetaa2 
     
    if toothInitialData["thetaa2"] is not None:
        thetaa2 = toothInitialData["thetaa2"]
        thetaf2 = toothInitialData["thetaf2"]

    # determination of root angles and face angles

    deltaa2 = delta2 + thetaa2 # face angle gear
    deltaf2 = delta2 - thetaf2 # root angle gear
    phiR =  atan( a*tan(DeltaSIGMA)*cos(deltaf2)/(Rm2*cos(thetaf2) - tz2*cos(deltaf2) ) ) # auxiliary angle for calculating pinion offset angle in root plane
    phiO =  atan(a*tan(DeltaSIGMA)*cos(deltaa2)/(Rm2*cos(thetaa2) - tz2*cos(deltaa2)) ) # auxiliary angle for calculating pinion offset angle in face plane
    zetaR =  asin(a*cos(phiR)* sin(deltaf2)/(Rm2*cos(thetaf2) - tz2*cos(deltaf2))) - phiR # pinion offset angle in root plane
    zetaO =  asin(a*cos(phiO)*sin(deltaa2)/(Rm2*cos(thetaa2) - tz2*cos(deltaa2))) - phiO # pinion offset angle in face plane
    deltaa1 =  asin(sin(DeltaSIGMA)*sin(deltaf2) + cos(DeltaSIGMA)*cos(deltaf2)*cos(zetaR)) # pinion face angle
    deltaf1 =  asin(sin(DeltaSIGMA)*sin(deltaa2) + cos(DeltaSIGMA)*cos(deltaa2)*cos(zetaO)) # pinion root angle
    thetaa1 = deltaa1 - delta1 # pinion addendum angle
    thetaf1 = delta1 - deltaf1 # pinion dedendum angle
    tzF2 = tz2 - (Rm2*sin(thetaa2) - ham2*cos(thetaa2))/sin(deltaa2) # gear face apex beyon crossing point along axi
    tzR2 = tz2 + (Rm2*sin(thetaf2) - hfm2*cos(thetaf2))/sin(deltaf2) # gear root apex beyond crossing point  along axi
    tzF1 = (a*sin(zetaR)*cos(deltaf2) - tzR2*sin(deltaf2) - c)/sin(deltaa1) # pinion face apex beyond crossing point
    tzR1 = (a*sin(zetaO)*cos(deltaa2) - tzF2*sin(deltaa2) - c)/sin(deltaf1) # pinion root apex beyond crossing point

    # determination of pinion face width
    # method 0 values
    b1 = b2
    be1 = cbe2*b1
    bi1 = b1 - be1
    if Method == 1: # hypoid values
        bp1 = sqrt(Re2**2 - ap**2) - sqrt(Ri2**2 - ap**2) # pinion facewidth in pitchplane
        b1A = sqrt(Rm2**2 - ap**2) - sqrt(Ri2**2 - ap**2) # pinion facewidth from calculation point to front crown
        lambdaprime = atan(sin(zetamp)*cos(delta2)/(u*cos(delta1) + cos(delta2)*cos(zetamp))) # auxiliary angle
        breri1 = b2*cos(lambdaprime)/cos(zetamp - lambdaprime) # pinion facewidth
        Deltabx1 = hmw*sin(zetaR)*(1 - 1/u) # pinion facewidth increment along pinion axis
        Deltagxe = cbe2*breri1/cos(thetaa1)*cos(deltaa1) + Deltabx1 - (hfm2 - c)*sin(delta1) # increment along pinion axis from calculation point to outside
        Deltagxi = (1 - cbe2)*breri1*cos(deltaa1)/cos(thetaa1) + Deltabx1 + (hfm2 - c)*sin(delta1) # increment along pinion axis from calculation point to inside
        be1 = (Deltagxe + ham1*sin(delta1))/cos(deltaa1)*cos(thetaa1) # pinion face width from calculation point to outside
        bi1 = ( Deltagxi - ham1*sin(delta1) )/ ( cos(delta1) - tan(thetaa1)*sin(delta1) ) # pinion facewidth from calculation point to inside
        b1 = bi1 + be1 # pinion facewidth along pitch cone


    ## determination of inner and outer spiral angles
    #pinion
    Re21 = sqrt(Rm2**2 + be1**2 + 2*Rm2*be1*cos(zetamp)) # gear cone distance of outer pinion boundary point 
    Ri21 = sqrt(Rm2**2 + bi1**2 - 2*Rm2*bi1*cos(zetamp)) # gear cone distance of inner pinion boundary point
    betae21 = asin( (2*Rm2*rc0*sin(betam2) - Rm2**2 + Re21**2)/(2*Re21*rc0) ) # gear spiral angle at outer boundary point
    betai21 = asin( (2*Rm2*rc0*sin(betam2) - Rm2**2 + Ri21**2) / (2*Ri21*rc0) ) # gear spiral anfle at inner boundary point
    zetaep21 = asin(ap/Re21) # pinion offset angle in pitch plane at outer boundary point
    zetaip21 = asin(ap/Ri21) # pinion offset angle at pitch plane at inner boundary point
    betae1 = betae21 + zetaep21 # outer pinion spiral angle
    betai1 = betai21 + zetaip21 # inner pinion spiral angle
    # gear
    betae2 = asin( (2*Rm2*rc0*sin(betam2) - Rm2**2 + Re2**2) / (2*Re2*rc0) ) # outer gear spiral angle
    betai2 = asin( (2*Rm2*rc0*sin(betam2) - Rm2**2 + Ri2**2) / (2*Ri2*rc0) ) # inner gear spiral angle
    ## determination of tooth depth
    hae1 = ham1 + be1*tan(thetaa1) # pinion outer addendum
    hae2 = ham2 + be2*tan(thetaa2) # gear outer addendum
    hfe1 = hfm1 + be1*tan(thetaf1) # pinion outer dedendum
    hfe2 = hfm2 + be2*tan(thetaf2) # gear outer dedendum
    he1 = hae1 + hfe1 # pinion outer whole depth
    he2 = hae2 + hfe2 # gear outer whole depth
    hai1 = ham1 - bi1*tan(thetaa1) # pinion inner addendum
    hai2 = ham2 - bi2*tan(thetaa2) # gear inner addendum
    hfi1 = hfm1 - bi1*tan(thetaf1) # pinion inner dedendum
    hfi2 = hfm2 - bi2*tan(thetaf2) # gear inner dedendum
    hi1 = hai1 + hfi1 # pinion inner whole depth
    hi2 = hai2 + hfi2 # gear inner whole depth
    ## determination of tooth thickness, pag 234 pdf    pag 44 ISO journal
    #alphan PAG 235
    xsm1 = xsmn - jen*Rm2*cos(betam2)/( 4*mmn*cos(alphan)*Re2*cos(betae2) ) # pinion thickness modification coefficient 
    smn1 = 0.5*mmn*pi + 2*mmn*(xsm1 + xhm1*tan(alphan)) # pinion mean normal circular tooth thickness smn1
    xsm2 = -xsmn - jen*Rm2*cos(betam2)/( 4*mmn*cos(alphan)*Re2*cos(betae2) ) # gear thickness modification coefficient
    smn2 = 0.5*mmn*pi + 2*mmn*(xsm2 - xhm1*tan(alphan)) # gear mean normal circular tooth thickness
    smt1 = smn1/(cos(betam1)) # pinion mean transverse circular thickness
    smt2 = smn2/(cos(betam2)) # gear mean transverse circular thickness
    dmn1 = dm1/((1 - sin(betam1)**2*cos(alphan)**2)*cos(delta1)) # pinion normal diameter
    dmn2 = dm2/((1 - sin(betam2)**2*cos(alphan)**2)*cos(delta2)) # gear normal diameter
    smnc1 = dmn1*sin(smn1/dmn1) # pinion mean normal chordal tooth thickness
    smnc2 = dmn2*sin(smn2/dmn2) # gear mean normal chordal tooth thickness
    hamc1 = ham1 + 0.5*dmn1*cos(delta1)*(1 - cos(smn1/dmn1)) # pinion mean chordal addendum
    hamc2 = ham2 + 0.5*dmn2*cos(delta2)*(1 - cos(smn2/dmn2)) # gear mean chordal addendum
    ## determination of remaining dimension
    Re1 = Rm1 + be1 # pinion outer pitch cone distance
    Ri1 = Rm1 - bi1 # pinion inner pitch cone distance
    de1 = 2*Re1*sin(delta1) # pinion outer pitch diameter
    di1 = 2*Ri1*sin(delta1) # pinion inner pitch diameter
    dae1 = de1 + 2*hae1*cos(delta1) # pinion outside addendum diameter 
    dae2 = de2 + 2*hae2*cos(delta2) # gear outside addendum diameter
    dfe1 = de1 - 2*hfe1*cos(delta1) # pinion outside dedendum diameter
    dfe2 = de2 - 2*hfe2*cos(delta2) # gear outside dedendum diameter
    dai1 = di1 + 2*hai1*cos(delta1) # pinion inside addendum diameter
    dai2 = di2 + 2*hai2*cos(delta2) # gear inside addendum diameter
    dfi1 = di1 - hfi1*cos(delta1) # pinion inside dedendum diameter
    dfi2 = di2 - hfi2*cos(delta2) # pinion inside dedendum diameter
    txo1 = tzm1 + be1*cos(delta1) - hae1*sin(delta1) # pinion crossing point to crown along axis
    txo2 = tzm2 + be2*cos(delta2) - hae2*sin(delta2) # gear crossing point to crown along axis
    txi1 = tzm1 - bi1*cos(delta1) - hai1*sin(delta1) # pinion crossing point to front crown along axis
    txi2 = tzm2 - bi2*cos(delta2) - hai2*sin(delta2) # gear crossing point to front crown along axis
    ht1 = (tzF1 + txo1)/cos(deltaa1)*sin(thetaa1 + thetaf1) - (tzR1 - tzF1)*sin(deltaf1) # pinion whole depth perpendicular to the root cone
    ## undercut check
    # pinion at the moment we choose the inner pitch cone distance as check
    # point which in most cases might be the most critical one (verify this conjecture)
    Rx1 = Ri1 # point to be checked for undercut internal pitch cone distance
    Rx2 = sqrt(Rm2**2 + (Rm1 - Rx1)**2 - 2*Rm2*(Rm1 - Rx1)*cos(zetamp)) # gear cone distance
    betax2 = asin((2*Rm2*rc0*sin(betam2) - Rm2**2 + Rx2**2)/(2*Rx2*rc0)) # gear spiral angle at check point
    zetaxp2 = asin(ap/Rx2) # pinion offset angle in pitch plane at check point
    betax1 = betax2 + zetaxp2 # pinion spiral angle at check point 
    dx1 = 2*Rx1*sin(delta1) # pinion pitch diameter at check point
    dx2 = 2*Rx2*sin(delta2) # gear pitch diameter at check point
    mxn = dx2/z2*cos(betax2) # mean normal module at check point
    dEx1 = dx2*z1*cos(betax2)/z2/cos(betax1) # pinion effective diameter at check point
    REx1 = dEx1/2/sin(delta1) # pinion appropriate cone distance
    znx1 = z1/((1-sin(betax1)**2*cos(alphan)**2)*cos(betax1)*cos(delta1)) # intermediate value
    alphalimx = -atan(    tan(delta1)*tan(delta2)/(cos(zetamp))*(  (REx1*sin(betax1) - Rx2*sin(betax2))/(REx1*tan(delta1) + Rx2*tan(delta2))  )    ) # limit pressure angle at checkpoint
    alphaeDx = alphanD - alphalimx # effective pressure anfle at check point drive side
    alphaeCx = alphanC + alphalimx # effective pressure angle at check point coast side
    alphaeminx = min(alphaeDx, alphaeCx) # smaller effective pressure angle
    ## determination of mionimum profile shift coefficient at calculation point on the pinion
    khapx = khap + (Rx2 - Rm2)*tan(thetaa2)/mmn # working tool addendum at checkpoint
    xhx1 = 1.1*khapx - (znx1*mxn*sin(alphaeminx)**2)/(2*mmn) # pinion minimum profile shift coefficient at checkpoint
    xhmminx1 = xhx1 + (dEx1 - dx1)*cos(delta1)/(2*mmn) # pinion minimum profile shift coefficient at calculation point 

    ## generating basic design data structure
    basicDesignData = DesignData()

    baseconethickPin = initialConeData["pinBaseThick"]
    baseconethickGear = initialConeData["gearBaseThick"]
    baseconApexPin = tzR1 - baseconethickPin/cos(pi/2-deltaf1)
    baseconApexGear = tzR2 - baseconethickGear/cos(pi/2-deltaf2)

    ## System data
    basicDesignData.system_data.hand = Hand
    basicDesignData.system_data.shaft_angle = SIGMA*180/pi
    basicDesignData.system_data.taperType = taper
    basicDesignData.system_data.transmissionType = "hypoid"
    basicDesignData.system_data.ratio = u
    basicDesignData.system_data.hypoid_offset = a
    basicDesignData.system_data.NOMINALDRIVEPRESSUREANGLE = alphanD*180/pi
    basicDesignData.system_data.NOMINALCOASTPRESSUREANGLE= alphanC*180/pi
    basicDesignData.system_data.NORMALMODULE = mmn

    basicDesignData.pinion_common_data.gen_type = 'GENERATED'
    basicDesignData.pinion_common_data.NTEETH = z1
    basicDesignData.pinion_common_data.SPIRALANGLE = betam1*180/pi
    basicDesignData.pinion_common_data.OUTERCONEDIST = Re1
    basicDesignData.pinion_common_data.INNERCONEDIST = Ri1
    basicDesignData.pinion_common_data.MEANCONEDIST = Rm1
    basicDesignData.pinion_common_data.FACEWIDTH = b1
    basicDesignData.pinion_common_data.FACEANGLE = deltaa1*180/pi
    basicDesignData.pinion_common_data.BACKANGLE = delta1*180/pi
    basicDesignData.pinion_common_data.FRONTANGLE = delta1*180/pi
    basicDesignData.pinion_common_data.PITCHANGLE = delta1*180/pi
    basicDesignData.pinion_common_data.BASECONEANGLE = deltaf1*180/pi
    basicDesignData.pinion_common_data.PITCHAPEX = tz1
    basicDesignData.pinion_common_data.FACEAPEX = tzF1
    basicDesignData.pinion_common_data.ROOTAPEX = tzR1
    basicDesignData.pinion_common_data.BASECONEAPEX = baseconApexPin
    setattr(basicDesignData.pinion_common_data, 'MEANCUTTERRAIDUS', rc0)
    setattr(basicDesignData.pinion_common_data, 'NORMAL_THICKNESS', smnc1)
    setattr(basicDesignData.pinion_common_data, 'XVERSECIRCULAR', smt1)
    setattr(basicDesignData.pinion_common_data, 'MEANNORMALCHORDALTHICKNESS', smnc1)
    setattr(basicDesignData.pinion_common_data, 'MEANADDENDUM', ham1)
    setattr(basicDesignData.pinion_common_data, 'MEANCHORDALADDENDUM', hamc1)

    basicDesignData.gear_common_data.gen_type = 'GENERATED'
    basicDesignData.gear_common_data.NTEETH = z2
    basicDesignData.gear_common_data.SPIRALANGLE = betam2*180/pi
    basicDesignData.gear_common_data.OUTERCONEDIST = Re2
    basicDesignData.gear_common_data.MEANCONEDIST = Rm2
    basicDesignData.gear_common_data.INNERCONEDIST = Ri2
    
    basicDesignData.gear_common_data.gen_type = GearGenType
    basicDesignData.gear_common_data.NTEETH = z2
    basicDesignData.gear_common_data.SPIRALANGLE = betam2*180/pi
    basicDesignData.gear_common_data.OUTERCONEDIST = Re2
    basicDesignData.gear_common_data.MEANCONEDIST = Rm2
    basicDesignData.gear_common_data.INNERCONEDIST = Ri2
    basicDesignData.gear_common_data.MEANNORMALCHORDALTHICKNESS = smnc2
    basicDesignData.gear_common_data.MEANADDENDUM = ham2
    basicDesignData.gear_common_data.MEANCHORDALADDENDUM = hamc2
    basicDesignData.gear_common_data.FACEWIDTH = b2
    basicDesignData.gear_common_data.FACEANGLE = deltaa2*180/pi
    basicDesignData.gear_common_data.BACKANGLE = delta2*180/pi
    basicDesignData.gear_common_data.FRONTANGLE = delta2*180/pi
    basicDesignData.gear_common_data.PITCHANGLE = delta2*180/pi
    basicDesignData.gear_common_data.BASECONEANGLE = deltaf2*180/pi
    basicDesignData.gear_common_data.PITCHAPEX = tz2
    basicDesignData.gear_common_data.FACEAPEX = tzF2
    basicDesignData.gear_common_data.ROOTAPEX = tzR2
    basicDesignData.gear_common_data.BASECONEAPEX = baseconApexGear
    basicDesignData.gear_common_data.MEANCUTTERRAIDUS = rc0
    basicDesignData.gear_common_data.NORMAL_THICKNESS = smnc2
    basicDesignData.gear_common_data.XVERSECIRCULAR = smt2
    
    handSign = +1
    phicnv = alphanC
    phicvx = alphanD
    if Hand.lower() == 'left':
        handSign = -1
        phicnv = alphanD
        phicvx = alphanC

    slidingBasePin = Rm1*sin(thetaf1) - hfm1*cos(thetaf1)
    slidingBaseGear = Rm2*sin(thetaf2) - hfm2*cos(thetaf2)
    machCtrbckPin = tzR1# - slidingBasePin/sin(deltaf1)
    machCtrbckGear = tzR2# - slidingBaseGear/sin(deltaf2)
    tz2 + (Rm2*sin(thetaf2) - hfm2*cos(thetaf2))/sin(deltaf2)
    
    ## generated gear
    # concave gear

    CM = Rm2*cos(thetaf2) - (tz2 - tzR2)*cos(deltaf2)
    q = atan((rc0*cos(betam2)/(CM - rc0*sin(betam2))))*180/pi
    S = rc0*cos(betam2)/sin(q*pi/180)
    basicDesignData.gear_machine_settings.concave.RADIALSETTING = S
    basicDesignData.gear_machine_settings.concave.CRADLEANGLE = q
    basicDesignData.gear_machine_settings.concave.ROOTANGLE = deltaf2*180/pi
    basicDesignData.gear_machine_settings.concave.SLIDINGBASE = slidingBaseGear*0 + gearTilt*pi/180*rc0
    basicDesignData.gear_machine_settings.concave.MACHCTRBACK = machCtrbckGear
    basicDesignData.gear_machine_settings.concave.RATIOROLL = cos(thetaf2)/sin(delta2)
    basicDesignData.gear_machine_settings.concave.TILTANGLE = gearTilt
    basicDesignData.gear_machine_settings.concave.SWIVELANGLE = betam2*180/pi

    # convex gear
    basicDesignData.gear_machine_settings.convex.RADIALSETTING = S
    basicDesignData.gear_machine_settings.convex.CRADLEANGLE = q
    basicDesignData.gear_machine_settings.convex.ROOTANGLE = deltaf2*180/pi
    basicDesignData.gear_machine_settings.convex.SLIDINGBASE = slidingBaseGear*0 + gearTilt*pi/180*rc0
    basicDesignData.gear_machine_settings.convex.MACHCTRBACK = machCtrbckGear
    basicDesignData.gear_machine_settings.convex.RATIOROLL = cos(thetaf2)/sin(delta2)
    basicDesignData.gear_machine_settings.convex.TILTANGLE = gearTilt
    basicDesignData.gear_machine_settings.convex.SWIVELANGLE = betam2*180/pi

    # concave pinion
    OM = Rm1*cos(thetaf1)
    q = atan((rc0*cos(betam1)/(OM - rc0*sin(betam1))))*180/pi
    S = rc0*cos(betam1)/sin(q*pi/180)
    basicDesignData.pinion_machine_settings.concave.RADIALSETTING = S
    basicDesignData.pinion_machine_settings.concave.CRADLEANGLE = q
    basicDesignData.pinion_machine_settings.concave.ROOTANGLE = deltaf1*180/pi
    basicDesignData.pinion_machine_settings.concave.SLIDINGBASE = slidingBasePin*0
    basicDesignData.pinion_machine_settings.concave.MACHCTRBACK = machCtrbckPin
    basicDesignData.pinion_machine_settings.concave.RATIOROLL = cos(thetaf1)/sin(delta1)

    # convex pinion
    basicDesignData.pinion_machine_settings.convex.RADIALSETTING = S
    basicDesignData.pinion_machine_settings.convex.CRADLEANGLE = q
    basicDesignData.pinion_machine_settings.convex.ROOTANGLE = deltaf1*180/pi
    basicDesignData.pinion_machine_settings.convex.SLIDINGBASE = slidingBasePin*0
    basicDesignData.pinion_machine_settings.convex.MACHCTRBACK = machCtrbckPin
    basicDesignData.pinion_machine_settings.convex.RATIOROLL = cos(thetaf1)/sin(delta1)

    # cutter data
    PQ = (hfm2 - ham2)/2
    P0N = hfm2*cos(thetaf2) - PQ*cos(thetaf2)
    Np = z2*cos(thetaf2)/sin(delta2)
    pointRadiusConcave = rc0 + smnc2/2
    pointRadiusConvex = rc0 - smnc2/2
    edgeRadius = 0.15*met

    # gear cutter
    # concave
    basicDesignData.gear_cutter_data.concave.POINTRADIUS = pointRadiusConcave
    basicDesignData.gear_cutter_data.concave.BLADEANGLE = phicnv*180/pi - gearTilt
    basicDesignData.gear_cutter_data.concave.EDGERADIUS = edgeRadius
    basicDesignData.gear_cutter_data.concave.TYPE = 'CURVED'
    basicDesignData.gear_cutter_data.concave.topremTYPE = 'NONE'
    basicDesignData.gear_cutter_data.concave.flankremTYPE = 'NONE'
    basicDesignData.gear_cutter_data.concave.RHO = 800
    basicDesignData.gear_cutter_data.concave.topremDEPTH = None
    basicDesignData.gear_cutter_data.concave.TopremRADIUS = None
    basicDesignData.gear_cutter_data.concave.flankremDEPTH = None
    basicDesignData.gear_cutter_data.concave.flankremRADIUS = None

    # convex
    basicDesignData.gear_cutter_data.convex.POINTRADIUS = pointRadiusConvex
    basicDesignData.gear_cutter_data.convex.BLADEANGLE = phicvx*180/pi + gearTilt
    basicDesignData.gear_cutter_data.convex.EDGERADIUS = edgeRadius
    basicDesignData.gear_cutter_data.convex.TYPE = 'CURVED'
    basicDesignData.gear_cutter_data.convex.topremTYPE = 'NONE'
    basicDesignData.gear_cutter_data.convex.flankremTYPE = 'NONE'
    basicDesignData.gear_cutter_data.convex.RHO = 800
    basicDesignData.gear_cutter_data.convex.topremDEPTH = None
    basicDesignData.gear_cutter_data.convex.topremRADIUS = None
    basicDesignData.gear_cutter_data.convex.flankremDEPTH = None
    basicDesignData.gear_cutter_data.convex.flankremRADIUS = None

    # pinion cutter
    rc0P = rc0
    if rc0Flag == None:
        match taper.lower():
            case "standard":
                rc0P = (Rm1 + 1.1*Rm1*sin(betam1))/2
            case "uniform":
                rc0P = uniformToothCoeff*Rm1*sin(betam1)
            case "duplex":
                rc0P = (Rm1 + 1.1*Rm1*sin(betam1))/2
            case "trl":
                rc0P = (Rm1 + 1.1*Rm1*sin(betam1))/2

    pointRadiusConcave = rc0P + smnc1/2
    pointRadiusConvex = rc0P - smnc1/2

    # concave
    basicDesignData.pinion_cutter_data.concave.POINTRADIUS = pointRadiusConcave
    basicDesignData.pinion_cutter_data.concave.BLADEANGLE = phicnv*180/pi
    basicDesignData.pinion_cutter_data.concave.EDGERADIUS = edgeRadius
    basicDesignData.pinion_cutter_data.concave.TYPE = 'STRAIGHT'
    basicDesignData.pinion_cutter_data.concave.TopremTYPE = 'NONE'
    basicDesignData.pinion_cutter_data.concave.FlankremTYPE = 'NONE'
    basicDesignData.pinion_cutter_data.concave.RHO = None
    basicDesignData.pinion_cutter_data.concave.TopremDEPTH = None
    basicDesignData.pinion_cutter_data.concave.TopremRADIUS = None
    basicDesignData.pinion_cutter_data.concave.FlankremDEPTH = None
    basicDesignData.pinion_cutter_data.concave.FlankremRADIUS = None

    # convex
    basicDesignData.pinion_cutter_data.convex.POINTRADIUS = pointRadiusConvex
    basicDesignData.pinion_cutter_data.convex.BLADEANGLE = phicvx*180/pi
    basicDesignData.pinion_cutter_data.convex.EDGERADIUS = edgeRadius
    basicDesignData.pinion_cutter_data.convex.TYPE = 'STRAIGHT'
    basicDesignData.pinion_cutter_data.convex.TopremTYPE = 'NONE'
    basicDesignData.pinion_cutter_data.convex.FlankremTYPE = 'NONE'
    basicDesignData.pinion_cutter_data.convex.RHO = None
    basicDesignData.pinion_cutter_data.convex.TopremDEPTH = None
    basicDesignData.pinion_cutter_data.convex.TopremRADIUS = None
    basicDesignData.pinion_cutter_data.convex.FlankremDEPTH = None
    basicDesignData.pinion_cutter_data.convex.FlankremRADIUS = None

    return basicDesignData

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

if __name__ == '__main__':
    main()