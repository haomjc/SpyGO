from dataclasses import dataclass, field, asdict
from typing import Literal, Dict
import numpy as np
from math import atan, pi, sqrt, sin , asin
from general_utils import *
import json
from copy import deepcopy

def log_dataclass(obj, tabstring = '', indent = 3):
    
    if not is_dataclass(obj):
        raise ValueError(f"Provided object {obj} is not a dataclass instance")

    description = str()

    for field in fields(obj):
        key = field.name
        value = getattr(obj, key)

        if is_dataclass(value):  # nested dataclass
            description += f"\n{tabstring}{key}:\n"
            description += log_dataclass(value, tabstring + ' ' * indent, indent)
        elif isinstance(value, dict):  # fallback for dicts inside dataclasses
            description += f"\n{tabstring}{key}: [dict]\n"
            for subkey, subvalue in value.items():
                description += f"{tabstring + ' ' * indent}{subkey}: {subvalue}\n"
        elif isinstance(value, np.ndarray):
            if value.size > 10:
                description += f"{tabstring}{key}: {type(value).__name__} of shape {value.shape}\n"
            else:
                description += f"{tabstring}{key}: {value}\n"
        else:
            # numpy arrays precision can be set globally. For simple numbers we need to format the string
            if isinstance(value, float) or isinstance(value, int):
                description += f"{tabstring}{key}: {value:.4f}\n"
            else: # print simple strings or other objects
                description += f"{tabstring}{key}: {value}\n"

    return description

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
    
    def __str__(self, tabstring = '', indent = 3):
        return log_dataclass(self)
    
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

    USE_SPRD_BLD_THICKNESS: bool = False

    def __str__(self, tabstring = '', indent = 3):
        return log_dataclass(self)

# 🔵 Cutter parameters (for a flank)
@dataclass
class CutterField:
    POINTRADIUS: float = 0.0
    EDGERADIUS: float = 0.0

    # Main blade settings
    TYPE: str = "NONE"
    RHO: float = 0.0
    BLADEANGLE: float = 0.0

    # Toprem settings
    topremTYPE: str = "NONE"
    topremDEPTH: float = 0.0
    topremRADIUS: float = 0.0
    topremANGLE: float = 0.0

    # Flankrem settings
    flankremTYPE: str = "NONE"
    flankremDEPTH: float = 0.0
    flankremRADIUS: float = 0.0
    flankremANGLE: float = 0.0

    def __str__(self, tabstring = '', indent = 3):
        return log_dataclass(self)
    
    def extract_settings(self, bounds_data = False):

        tool_settings = self

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

        if bounds_data == True:
            RHO = tool_settings.RHO

        if topremTYPE.lower() == 'blended':
            topremRADIUS = tool_settings.topremRADIUS
            topremDEPTH = tool_settings.topremDEPTH
        elif topremTYPE.lower() == 'straight':
            topremANGLE = tool_settings.topremANGLE
            topremDEPTH = tool_settings.topremDEPTH
            topremRADIUS = RHO_STRAIGHT
        else: # feature disabled
            topremRADIUS = 1000
            Czblade = RHO*sin(BLADEANGLE*pi/180)
            theta_iniz_blade = asin((Czblade + EDGERADIUS)/(RHO + EDGERADIUS))
        
            topremDEPTH = EDGERADIUS*(1-sin(theta_iniz_blade))
            topremANGLE = 0
            if bounds_data == False:
                if tool_settings.topremANGLE != 0:
                    msgbox("Tool with angle offset between tip & blade (toprem angle) not implemented", 'Error', 1)

        if bounds_data == True:
            topremANGLE = tool_settings.topremANGLE
            topremDEPTH = tool_settings.topremDEPTH
            topremRADIUS = tool_settings.topremRADIUS
        
        if flankremTYPE.lower() == 'blended':
            flankremRADIUS = tool_settings.flankremRADIUS
            flankremDEPTH = tool_settings.flankremDEPTH
        elif flankremTYPE.lower() == 'straight':
            flankremANGLE = tool_settings.flankremANGLE
            flankremDEPTH = tool_settings.flankremDEPTH
            flankremRADIUS = RHO_STRAIGHT
        else: # feature diabled
            flankremRADIUS = RHO
            flankremDEPTH = 40
            flankremANGLE = 0
            if bounds_data == False:
                if tool_settings.flankremANGLE != 0:
                    msgbox("Tool with angle offset blade and flanrem (flankrem angle) not implemented", 'Error', 1)

        if bounds_data == True:
            flankremANGLE = tool_settings.flankremANGLE
            flankremDEPTH = tool_settings.flankremDEPTH
            flankremRADIUS = tool_settings.flankremRADIUS

        return (POINTRADIUS, EDGERADIUS, RHO, BLADEANGLE, topremDEPTH, topremRADIUS, topremANGLE, flankremDEPTH, flankremRADIUS, flankremANGLE)

    def update_settings(self, x_index, x_values) -> None:
        fields_list = ['POINTRADIUS', 'EDGERADIUS', 'RHO', 'BLADEANGLE', 'topremDEPTH', 'topremRADIUS', 'topremANGLE', 'flankremDEPTH', 'flankremRADIUS', 'flankremANGLE']
        
        for index, value in zip(x_index, x_values):
            setattr(self, fields_list[index], value)
        
        if 5 in x_index:
            self.topremTYPE = 'BLENDED'
        
        if 8 in x_index:
            self.flankremTYPE = 'BLENDED'

        if 2 in x_index:
            self.TYPE = 'CURVED'

        return
# 🔵 Machine parameters (for a flank)

@dataclass
class MachineField:
    RADIALSETTING: float = 0.0
    TILTANGLE: float = 0.0
    SWIVELANGLE: float = 0.0
    BLANKOFFSET: float = 0.0
    SLIDINGBASE: float = 0.0
    CRADLEANGLE: float = 0.0
    MACHCTRBACK: float = 0.0
    ROOTANGLE: float = 0.0
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
    TLT1: float = 0.0
    TLT2: float = 0.0
    TLT3: float = 0.0
    TLT4: float = 0.0
    TLT5: float = 0.0
    TLT6: float = 0.0
    SW1: float = 0.0
    SW2: float = 0.0
    SW3: float = 0.0
    SW4: float = 0.0
    SW5: float = 0.0
    SW6: float = 0.0

    def __str__(self, tabstring = '', indent = 3):
        return log_dataclass(self)
        return self.to_string()

    def extract_settings(self) -> np.array:
        machine_settings = self
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
        TLT1  = machine_settings.TLT1
        TLT2  = machine_settings.TLT2
        TLT3  = machine_settings.TLT3
        TLT4  = machine_settings.TLT4
        TLT5  = machine_settings.TLT5
        TLT6  = machine_settings.TLT6
        SW1   = machine_settings.SW1
        SW2   = machine_settings.SW2
        SW3   = machine_settings.SW3
        SW4   = machine_settings.SW4
        SW5   = machine_settings.SW5
        SW6   = machine_settings.SW6

        
        if m == 0: # Zero ratio roll means it is a formate member (usually should be gear)
            H = machine_settings.HORIZONTAL
            V = machine_settings.VERTICAL
            q0 = atan(V/H)
            S0 = sqrt(V**2 + H**2)
            q0 = q0*180/pi
            return np.array([
                [  S0,     0,     0,     0,     0,     0,     0,     0],     #joint Tool 2
                [sig0,     0,     0,     0,     0,     0,     0,     0],     #joint Tool 4
                [  z0,     0,     0,     0,     0,     0,     0,     0],     #joint Tool 3
                [  E0,     0,     0,     0,     0,     0,     0,     0],     #joint Gear 1
                [  B0,     0,     0,     0,     0,     0,     0,     0],     #joint Gear 2
                [  q0,     0,     0,     0,     0,     0,     0,     0],     #joint Tool 1
                [   0,     0,     0,     0,     0,     0,     0,     0],     #joint Gear 5
                [  D0,     0,     0,     0,     0,     0,     0,     0],     #joint Gear 4
                [gam0,     0,     0,     0,     0,     0,     0,     0]      #joint Gear 3
                ])

        return np.array([
            [  S0,   S1,   S2,   S3,   S4,   S5,   S6,     0],     #joint Tool 2
            [sig0, TLT1, TLT2, TLT3, TLT4, TLT5, TLT6,     0],     #joint Tool 4
            [  z0,  SW1,  SW2,  SW3,  SW4,  SW5,  SW6,     0],     #joint Tool 3
            [  E0,   E1,   E2,   E3,   E4,   E5,   E6,     0],     #joint Gear 1
            [  B0,   B1,   B2,   B3,   B4,   B5,   B6,     0],     #joint Gear 2
            [  q0,    1,    0,    0,    0,    0,    0,     0],     #joint Tool 1
            [   0,    m,   C2,   D6,  E24, F120, G720, H5040],     #joint Gear 5
            [  D0,    0,    0,    0,    0,    0,    0,     0],     #joint Gear 4
            [gam0,    0,    0,    0,    0,    0,    0,     0]      #joint Gear 3
            ])  

    def update_settings(self, x_index, x_values) -> None:

        fields_list = [
            "RADIALSETTING", "TILTANGLE", "SWIVELANGLE", "BLANKOFFSET", "SLIDINGBASE", "CRADLEANGLE", "MACHCTRBACK", "ROOTANGLE", 
            "RATIOROLL", "C2", "D6", "E24", "F120", "G720", "H5040",
            "H1", "H2", "H3", "H4", "H5", "H6",
            "V1", "V2", "V3", "V4", "V5", "V6",
            "R1", "R2", "R3", "R4", "R5", "R6",
            "TLT1", "TLT2", "TLT3", "TLT4", "TLT5", "TLT6",
            "SW1", "SW2", "SW3", "SW4", "SW5", "SW6",
        ]
        index_map = [0,1,2,3,4,5,7,8,  # config
                    15,24,33,42,51,60,69, # roll
                    13,22,31,40,49,58, # helical
                    12,21,30,39,48,57, # vertical
                    9,18,27,36,45,54,  # radial
                    10,19,28,37,46,55, # mod. tilt
                    11,20,29,38,47,56  # mod. swivel
                    ]
        
        for value, index in zip(x_values, x_index):
            setattr(self, fields_list[index_map.index(index)], value)
        return

# 🔵 Group the flanks for each member
@dataclass
class MemberFlankData:
    concave: MachineField = field(default_factory=MachineField)
    convex: MachineField = field(default_factory=MachineField)

    def __str__(self, tabstring = '', indent = 3):
        return log_dataclass(self)

@dataclass
class MemberCutterData:
    concave: CutterField = field(default_factory=CutterField)
    convex: CutterField = field(default_factory=CutterField)

    def __str__(self, tabstring = '', indent = 3):
        return log_dataclass(self)

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

        return machine_settings.extract_settings()

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
        gammaF = common_settings.FACEANGLE

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

        

        return tool_settings.extract_settings()
    
    def to_json(self, filename, indent = 4):

        data_dict = asdict(self)
        # Dump to JSON file
        with open(filename, "w") as f:
            json.dump(data_dict, f, indent=indent)
        return

    # LOG
    def __str__(self, tabstring = '', indent = 3):
        return log_dataclass(self)
    
    # getters
    def get_machine_field(self, member: Literal["gear", "pinion"], flank: Literal["concave", "convex"]):

        if member == 'pinion':
            machine_field = self.pinion_machine_settings
        else:
            machine_field = self.gear_machine_settings

        if flank == 'concave':
            machine_field_flank = machine_field.concave
        else:
            machine_field_flank = machine_field_flank.convex

        return machine_field_flank

    def get_tool_field(self, member: Literal["gear", "pinion"], flank: Literal["concave", "convex"]):

        if member == 'pinion':
            tool_field = self.pinion_cutter_data
        else:
            tool_field = self.gear_cutter_data

        if flank == 'concave':
            tool_field_flank = tool_field.concave
        else:
            tool_field_flank = tool_field.convex

        return tool_field_flank
    
    # setters
    def copy_machine_tool_settings(self, member: Literal["gear", "pinion", "both"], flank: Literal["concave", "convex", "both"], data: "DesignData"):
        
        if member.lower() == "both":
            self.copy_machine_tool_settings("gear", flank, data)
            self.copy_machine_tool_settings("pinion", flank, data)
            return

        if flank.lower() in ["both", "completing", "spread_blade", "spreadblade", "SB"]:
            self.copy_machine_tool_settings(member, "concave", data)
            self.copy_machine_tool_settings(member, "convex", data)
            return 
        
        if member.lower() == "gear":
            member_machine_field = self.gear_machine_settings
            member_cutter_field = self.gear_cutter_data
            other_member_machine_field = data.gear_machine_settings
            other_member_cutter_field = data.gear_cutter_data
        elif member.lower() == "pinion":
            member_machine_field = self.pinion_machine_settings
            member_cutter_field = self.pinion_cutter_data
            other_member_machine_field = data.pinion_machine_settings
            other_member_cutter_field = data.pinion_cutter_data

        if flank.lower() == "concave":
            machine_field = member_machine_field.concave
            cutter_field = member_cutter_field.concave
            other_machine_field = other_member_machine_field.concave
            other_cutter_field = other_member_cutter_field.concave
        elif flank.lower() == "convex":
            machine_field = member_machine_field.convex
            cutter_field = member_cutter_field.convex
            other_machine_field = other_member_machine_field.convex
            other_cutter_field = other_member_cutter_field.convex
        
        machine_field = deepcopy(other_machine_field)
        cutter_field = deepcopy(other_cutter_field)
        return
    
    @staticmethod
    def manage_machine_settings(member, systemHand, mode = 'gleason'):
        
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

    def update_settings(self, member: Literal["gear", "pinion"], flank: Literal["concave", "convex", "both"], x_index, x_values, return_copy = False):
        
        x_index = np.array(x_index)
        x_values = np.array(x_values)

        data = self
        if return_copy:
            data = deepcopy(self)

        if member.lower() == "gear":
            member_tool_settings = data.gear_cutter_data
            member_machine_settings = data.gear_machine_settings
        else:
            member_tool_settings = data.pinion_cutter_data
            member_machine_settings = data.pinion_machine_settings

        if flank.lower() == 'concave':
            tool_settings = member_tool_settings.concave
            machine_settings = member_machine_settings.concave
        else:
            tool_settings = member_tool_settings.convex
            machine_settings = member_machine_settings.convex

        # need also to implement "both" ("completing case"), where we just modify first the concae then the convex flank starting from mean cutter radius and point width
        # TODO...

        # splitting settings
        index_machine = x_index[x_index < 72]
        index_tool = x_index[x_index >= 72] - 72
        num_machine = max(index_machine.shape)
        machine_values = x_values[:num_machine] 
        tool_values = x_values[num_machine:]


        tool_settings.update_settings(index_tool, tool_values)
        machine_settings.update_settings(index_machine, machine_values)

        if return_copy:
            return data
        
        return

# dataclasses for gear/pinion concave/convex data attributes of Hypoid object
@dataclass
class flankData:
    concave: object = field(default_factory=list)
    convex: object = field(default_factory=list)
    both: object = field(default_factory=list) # structures that share data for both flanks (i.e. Nurbs for both flanks)
    def __str__(self, tabstring = '', indent = 3):
        return log_dataclass(self)

@dataclass
class FlankNumericalData:
    gear: flankData = field(default_factory=flankData)
    pinion: flankData = field(default_factory=flankData)

    def set_value(self, member, flank, result):
        if member.lower() == 'gear':
            member_data = self.gear
        else:
            member_data = self.pinion

        if flank.lower() == 'concave':
            member_data.concave = result
        elif flank.lower() == 'convex':
            member_data.convex = result
        else:
            member_data.both = result
        
        return
    
    def get_value(self, member, flank):
        if member.lower() == 'gear':
            member_data = self.gear
        else:
            member_data = self.pinion

        if flank.lower() == 'concave':
            return member_data.concave
        elif flank.lower() == 'convex':
            return member_data.convex
        else:
            return member_data.both
    
    def __str__(self, tabstring = '', indent = 3):
        return log_dataclass(self)

@dataclass
class MemberData:
    gear: object = field(default_factory=list)
    pinion: object = field(default_factory=list)
    
    def __str__(self, tabstring = '', indent = 3):
        return log_dataclass(self)

@dataclass
class identificationProblemData:
    designData: DesignData = field(default_factory=DesignData)
    gear: flankData = field(default_factory=flankData)
    pinion: flankData = field(default_factory=flankData)

    def get_value(self, member, flank):
        if member.lower() == 'gear':
            member_data = self.gear
        else:
            member_data = self.pinion

        if flank.lower() == 'concave':
            return member_data.concave
        elif flank.lower() == 'convex':
            return member_data.convex
        else:
            return member_data.both
    
    def set_value(self, member, flank, result):
        if member.lower() == 'gear':
            member_data = self.gear
        else:
            member_data = self.pinion

        if flank.lower() == 'concave':
            member_data.concave = result
        elif flank.lower() == 'convex':
            member_data.convex = result
        else:
            member_data.both = result
        
        return
    
    def __str__(self, tabstring = '', indent = 3):
        return log_dataclass(self)