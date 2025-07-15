from dataclasses import dataclass, field
from typing import Literal, Dict
import numpy as np
from math import atan, pi, sqrt, sin , asin
from general_utils import *

# 🔵 Top-level system data
@dataclass
class SystemData:
    # source: str = "Macrogeometry_data"
    hand: Literal["left", "right"] = "right"  # pinion hand
    shaft_angle: float = 0.0
    ratio: float = 0.0
    hypoid_offset: float = 0.0
    E: float = 0.0
    P: float = 0.0
    G: float = 0.0
    alpha: float = 0.0
    taperType: str = "Unkown"
    transmissionType: Literal["Hypoid", "Bevel"] = "Bevel"  # e.g., Hypoid, Spiral, Bevel
    NOMINALDRIVEPRESSUREANGLE: float = 0.0
    NOMINALCOASTPRESSUREANGLE: float = 0.0
    NORMALMODULE: float = 0.0


# 🔵 Common gear/blank parameters (gear or pinion)
@dataclass
class CommonField:
    gen_type: str = ""
    NTEETH: int = 0
    SPIRALANGLE: float = 0.0
    OUTERCONEDIST: float = 0.0
    INNERCONEDIST: float = 0.0
    MEANCONEDIST: float = None
    FACEWIDTH: float = 0.0
    FACEANGLE: float = 0.0
    BACKANGLE: float = 0.0
    FRONTANGLE: float = 0.0
    PITCHANGLE: float = 0.0
    BASECONEANGLE: float = 0.0
    PITCHAPEX: float = 0.0
    FACEAPEX: float = 0.0
    ROOTAPEX: float = 0.0
    BASECONEAPEX: float = 0.0
    MEANNORMALCHORDALTHICKNESS: float = None
    MEANADDENDUM: float = None
    MEANCHORDALADDENDUM: float = None
    MEANCUTTERRAIDUS: float = None
    NORMAL_THICKNESS: float = None
    XVERSECIRCULAR: float = None

    # axial plane shaft points
    ShaftzB: float = 0.0
    ShaftRB: float = 0.0
    ShaftzA: float = 0.0
    ShaftRA: float = 0.0
    ShaftDiA: float = 0.0  # internal diameter at point A
    ShaftDiB: float = 0.0  # internal diameter at point B
     
    # axial plane boundaries
    zFACEHEEL: float = None
    RFACEHEEL: float = None
    zFACETOE: float = None
    RFACETOE: float = None
    zROOTHEEL: float = None
    RROOTHEEL: float = None
    zROOTTOE: float = None
    RROOTTOE: float = None
    # Additional fields can be added as needed



# 🔵 Cutter parameters (for a flank)
@dataclass
class CutterField:
    POINTRADIUS: float = 0.0
    BLADEANGLE: float = 0.0
    EDGERADIUS: float = 0.0
    TYPE: str = "NONE"
    topremTYPE: str = "NONE"
    flankremTYPE: str = "NONE"
    RHO: float = 0.0
    topremDEPTH: float = 0.0
    topremRADIUS: float = 0.0
    topremANGLE: float = 0.0
    flankremDEPTH: float = 0.0
    flankremRADIUS: float = 0.0
    flankremANGLE: float = 0.0


# 🔵 Machine parameters (for a flank)
@dataclass
class MachineField:
    RADIALSETTING: float = 0.0
    CRADLEANGLE: float = 0.0
    TILTANGLE: float = 0.0
    SWIVELANGLE: float = 0.0
    BLANKOFFSET: float = 0.0
    ROOTANGLE: float = 0.0
    MACHCTRBACK: float = 0.0
    SLIDINGBASE: float = 0.0
    RATIOROLL: float = 0.0
    C2: float = 0.0
    D6: float = 0.0
    E24: float = 0.0
    F120: float = 0.0
    G720: float = 0.0
    H5040: float = 0.0
    H1: float = 0.0
    H2: float = 0.0
    H3: float = 0.0
    H4: float = 0.0
    H5: float = 0.0
    H6: float = 0.0
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    R1: float = 0.0
    R2: float = 0.0
    R3: float = 0.0
    R4: float = 0.0
    R5: float = 0.0
    R6: float = 0.0
    # Formate settings
    HORIZONTAL: float = 0.0
    VERTICAL: float = 0.0


# 🔵 Group the flanks for each member
@dataclass
class MemberFlankData:
    concave: MachineField = field(default_factory=MachineField)
    convex: MachineField = field(default_factory=MachineField)


@dataclass
class MemberCutterData:
    concave: CutterField = field(default_factory=CutterField)
    convex: CutterField = field(default_factory=CutterField)


# 🔵 Full design data
@dataclass
class DesignData:
    system_data: SystemData = field(default_factory=SystemData)
    gear_common_data: CommonField = field(default_factory=CommonField)
    pinion_common_data: CommonField = field(default_factory=CommonField)
    gear_machine_settings: MemberFlankData = field(default_factory=MemberFlankData)
    pinion_machine_settings: MemberFlankData = field(default_factory=MemberFlankData)
    gear_cutter_data: MemberCutterData = field(default_factory=MemberCutterData)
    pinion_cutter_data: MemberCutterData = field(default_factory=MemberCutterData)

    def extract_machine_settings_matrix(self, member: Literal["gear", "pinion"], flank: Literal["concave", "convex"]) -> MachineField:
        if member.lower() == "gear":
            machine_settings = getattr(self.gear_machine_settings, flank.lower())
        else:
            machine_settings = getattr(self.pinion_machine_settings, flank.lower())

            S0    = machine_settings.RADIALSETTING
            sig0  = machine_settings.TILTANGLE
            z0    = machine_settings.SWIVELANGLE
            E0    = machine_settings.BLANKOFFSET
            gam0  = machine_settings.ROOTANGLE
            D0    = machine_settings.MACHCTRBACK
            B0    = machine_settings.SLIDINGBASE
            q0    = machine_settings.CRADLEANGLE
            m     = machine_settings.RATIOROLL
            C2    = machine_settings.C2
            D6    = machine_settings.D6
            E24   = machine_settings.E24
            F120  = machine_settings.F120
            G720  = machine_settings.G720
            H5040 = machine_settings.H5040
            B1    = machine_settings.H1
            B2    = machine_settings.H2
            B3    = machine_settings.H3
            B4    = machine_settings.H4
            B5    = machine_settings.H5
            B6    = machine_settings.H6
            E1    = machine_settings.V1
            E2    = machine_settings.V2
            E3    = machine_settings.V3
            E4    = machine_settings.V4
            E5    = machine_settings.V5
            E6    = machine_settings.V6
            S1    = machine_settings.R1
            S2    = machine_settings.R2
            S3    = machine_settings.R3
            S4    = machine_settings.R4
            S5    = machine_settings.R5
            S6    = machine_settings.R6

        if member.lower() == 'gear' and self.gear_common_data.gen_type.lower() == 'formate':
            H = machine_settings.HORIZONTAL
            V = machine_settings.VERTICAL
            q0 = atan(V/H)
            S0 = sqrt(V**2 + H**2)
            q0 = q0*180/pi
            return np.array([
                [  S0,  0,  0,  0,  0,  0,  0,  0],     #joint Tool 2
                [sig0,  0,  0,  0,  0,  0,  0,  0],     #joint Tool 4
                [  z0,  0,  0,  0,  0,  0,  0,  0],     #joint Tool 3
                [  E0,  0,  0,  0,  0,  0 ,  0,  0],     #joint Gear 1
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

    def extract_blank_settings(self, member: Literal["gear", "pinion"]):
        if member.lower() == "gear":
            common_settings = self.gear_common_data
            root_angle = self.pinion_machine_settings.concave.ROOTANGLE
        else:
            common_settings = self.pinion_common_data
            root_angle = self.gear_machine_settings.concave.ROOTANGLE

        A0 = common_settings.OUTERCONEDIST
        Fw = common_settings.FACEWIDTH
        beta = common_settings.SPIRALANGLE

        # front/back cones
        deltaf = common_settings.FRONTANGLE
        deltab = common_settings.BACKANGLE

        # pitch cone
        gammaP = common_settings.PITCHANGLE
        dpa = common_settings.PITCHAPEX

        # face cone
        dfa = common_settings.FACEAPEX
        gammaF = common_settings.FACECANGLE

        # root cone
        dra = common_settings.ROOTAPEX
        gammaR = root_angle

        # base cone
        gammaB = common_settings.BASECONEANGLE
        dba = common_settings.BASECONEAPEX
        
        RA = common_settings.ShaftRA
        deltaf = deltaf*pi/180
        deltab = deltab*pi/180
        gammaP = gammaP*pi/180
        gammaF = gammaF*pi/180
        gammaR = gammaR*pi/180
        gammaB = gammaB*pi/180
        beta = beta*pi/180 
        return (A0, Fw, beta, deltaf, deltab, gammaP, dpa, gammaF, dfa, gammaR, dra, gammaB, dba, RA)
            
    def extract_tool_settings(self, member: Literal["gear", "pinion"], flank: Literal["concave", "convex"]):
        if member.lower() == "gear":
            member_tool_settings = self.gear_cutter_data
        else:
            member_tool_settings = self.pinion_cutter_data

        if flank.lower() == 'concave':
            tool_settings = member_tool_settings.concave
        else:
            tool_settings = member_tool_settings.convex

        alphaFlank = 0
        RHO_STRAIGHT = 2e6

        POINTRADIUS = tool_settings.POINTRADIUS
        BLADEANGLE = tool_settings.BLADEANGLE
        EDGERADIUS = tool_settings.EDGERADIUS

        TYPE = tool_settings.TYPE
        topremTYPE = tool_settings.topremTYPE
        flankremTYPE = tool_settings.flankremTYPE

        if TYPE.lower() == 'curved':
            RHO = tool_settings.RHO
        else:
            RHO = RHO_STRAIGHT

        if topremTYPE.lower() == 'blended':
            topremRADIUS = tool_settings.topremRADIUS
            topremDEPTH = tool_settings.topremDEPTH
        elif topremTYPE.lower() == 'straight':
            topremANGLE = tool_settings.topremANGLE
            topremDEPTH = tool_settings.topremDEPTH
            topremRADIUS = RHO_STRAIGHT
        else: # feature disabled
            topremRADIUS = 100
            Czblade = RHO*sin(BLADEANGLE*pi/180)
            theta_iniz_blade = asin((Czblade + EDGERADIUS)/(RHO + EDGERADIUS))
        
            topremDEPTH = EDGERADIUS - EDGERADIUS*sin(theta_iniz_blade)
            topremANGLE = 0
            if tool_settings.topremANGLE != 0:
                msgbox("Tool with angle offset between tip & blade (toprem angle) not implemented", 'Error', 1)

        if flankremTYPE.lower() == 'blended':
            flankremRADIUS = tool_settings.flankremRADIUS
            flankremDEPTH = tool_settings.flankremDEPTH
        elif flankremTYPE.lower() == 'straight':
            flankremANGLE = tool_settings.flankremANGLE
            flankremDEPTH = tool_settings.flankremDEPTH
            flankremRADIUS = RHO_STRAIGHT
        else: # feature diabled
            flankremRADIUS = RHO
            if flankremDEPTH is None:
                flankremDEPTH = RHO/3
            flankremANGLE = 0
            if tool_settings.topremANGLE != 0:
                msgbox("Tool with angle offset blade and flanrem (flankrem angle) not implemented", 'Error', 1)

        return (POINTRADIUS, RHO, EDGERADIUS, topremRADIUS, flankremRADIUS, BLADEANGLE, flankremDEPTH, topremDEPTH, topremANGLE, flankremANGLE)
        

def initialize_design_data() -> DesignData:
    return DesignData()

# dataclasses for gear/pinion concave/convex data attributes of Hypoid object

@dataclass
class flankData:
    concave: object = field(default_factory=list)
    convex: object = field(default_factory=list)

class flankDataWithBoth:
    concave: object = field(default_factory=list)
    convex: object = field(default_factory=list)    
    both: object = field(default_factory=list)

@dataclass
class FlankAndMemberNumericalData:
    gear: flankData = field(default_factory=flankData)
    pinion: flankData = field(default_factory=flankData)

@dataclass
class FlankAndMemberNumericalDataWithBoth:
    gear: flankDataWithBoth = field(default_factory=flankDataWithBoth)
    pinion: flankDataWithBoth = field(default_factory=flankDataWithBoth)

@dataclass
class MemberData:
    gear: object = field(default_factory=list)
    pinion: object = field(default_factory=list)

def numerical_template_data() -> FlankAndMemberNumericalData:
    return FlankAndMemberNumericalData()


