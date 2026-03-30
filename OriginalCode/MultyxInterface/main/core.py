import ctypes
import sys
import os

import multiprocessing 
import time
import shutil
import numpy as np
from MultyxInterface.main.data_classes import *
from typing import List, Optional, Any
from hypoid.main.data_structs import DesignData

def build_LTCA(
    system: SystemData,
    side: np.ndarray,
    torque: np.ndarray,
    speed: np.ndarray,
    path: str,
    ses_filename: str,
    EPGalpha: Optional[List[List[float]]] = None,
    ses_type: str = "T3D",
    material_properties: List[List[float]] = [[204000, 0.3], [204000, 0.3]], # Young modulus, poisson ratio for member 1 and 2
    sim_options: List[int] = [5, 15, 11, 8, 1, 7, 1, 7],
) -> List[List[LTCA]]:
    """
    Build LTCA configuration data
    """
    torque = np.array(torque)
    speed = np.array(speed)

    ratio = system.gear_nteeth / system.pin_nteeth
    rows, cols = torque.shape
    LTCA_list = [[None for _ in range(cols)] for _ in range(rows)]

    if EPGalpha is None:
        EPGalpha = [[[0, 0, 0, 0] for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            signC = 1 if system.hand.upper() == "RIGHT" else -1
            if side[i][j].lower() == "coast":
                signC *= -1

            epg = EPGalpha[i][j]
            dir_name = os.path.join(
                path,
                f"worker_{i+1}_{j+1}_EPGa_[{int(epg[0]*1000)}_{int(epg[1]*1000)}_{int(epg[2]*1000)}_{int(epg[3]*1000)}]"
            )
            ses_sim = ses_filename[:-4] + "_sim.ses"

            # gear and pinion data
            gear_torque = signC * abs(torque[i, j]) * ratio
            gear_rpm = signC * abs(speed[i, j]) / ratio
            pin_torque = gear_torque * (system.pin_nteeth / system.gear_nteeth)
            pin_rpm = signC * speed[i, j]

            ntimesteps = sim_options[2]

            pin = MemberData()
            pin.member = "PINION" # member label inside T3D
            pin.rze = "RZEpin.dat" # filename where RZE is stored for T3D
            pin.nteeth = system.pin_nteeth
            pin.torque = pin_torque
            pin.rpm = pin_rpm
            pin.pattern = "ContactPatternPinion.dat"
            pin.patterndata = "PatternDataPinion.dat"
            pin.tooth_begin = sim_options[4]
            pin.tooth_end = sim_options[5]
            pin.contact = "Contact_pin.dat"
            pin.bending  = f"Bending_pin_T{j+1}.dat"
            pin.material = material_properties[0]

            gear = MemberData()
            gear.member = "GEAR"
            gear.nteeth = system.gear_nteeth
            gear.torque = gear_torque
            gear.rpm = gear_rpm
            gear.pattern = "ContactPatternGear.dat"
            gear.patterndata = "PatternDataGear.dat"
            gear.tooth_begin = sim_options[6]
            gear.tooth_end = sim_options[7]
            gear.contact = "Contact_gear.dat"
            gear.bending = f"Bending_gear_T{j+1}.dat"
            gear.pattern_no_edge = "ContactPatternGear_NO_EDGE.dat"
            gear.material = material_properties[1]


            deltatime = abs(60 / (pin.rpm * pin.nteeth * (ntimesteps - 1)))
            initialtime = -deltatime

            status = {
                "tipContactStatus": 0,
                "backContactStatus": 0,
                "pinPressureStatus": 0,
                "gearPressureStatus": 0,
                "pinBendingStatus": 1,
                "gearBendingStatus": 1,
                "pinContactStatus": 0,
                "gearContactStatus": 0,
                "LTEStatus": 1,
                "simStatus": 1,
                "pinRZE": 0,
                "gearRZE": 0,
                "msh": system.msh_geometry is not None,
            }

            LTCA_instance = LTCA()
            LTCA_instance.id = 1
            LTCA_instance.dir = dir_name
            LTCA_instance.orignal_dir = path
            LTCA_instance.ses = ses_sim
            LTCA_instance.script = "RunScript.txt" # may not be necessary anymore since we interface direclty with multyx session
            LTCA_instance.memory = 3
            LTCA_instance.ses_type = ses_type
            
            LTCA_instance.status_checks = status
            LTCA_instance.nprofdivs = sim_options[0]
            LTCA_instance.nfacedivs = sim_options[1]
            LTCA_instance.ntimesteps = sim_options[2]
            LTCA_instance.nthreads = sim_options[3]

            LTCA_instance.post = "postproc.dat"
            LTCA_instance.export = f"EXPORT_sim_{j+1}.DAT"
            LTCA_instance.calyx_export_script = f"postexportgearcontactresults_hookfns_{j+1}.cmd"
            LTCA_instance.backlash_file = "BACKLASH.DAT"
            LTCA_instance.thetaz = f"LTE{j+1}.DAT"
            
            LTCA_instance.system_hand = system.hand
            LTCA_instance.side = side[i][j]
            LTCA_instance.EPGalpha = epg
            LTCA_instance.pin = pin
            LTCA_instance.gear = gear
            LTCA_instance.deltatime = deltatime
            LTCA_instance.initialtime = initialtime

            # directory + file copies
            os.makedirs(dir_name, exist_ok=True)
            shutil.copy(os.path.join(path, ses_filename), os.path.join(dir_name, LTCA_instance.ses))
            if system.msh_geometry:
                for msh in system.msh_geometry:
                    src = os.path.join(path, "mshData", msh)
                    dst = os.path.join(dir_name, msh)
                    shutil.copy(src, dst)

            LTCA_list[i][j] = LTCA_instance

    return LTCA_list

def hook_fcn_copy(filename, newfilename, id):
    # Read the file content into a list, one element per line
    fileContent = [None] * 200

    with open(filename, 'r') as f:
        counter = 0
        for line in f:
            fileContent[counter] = line.rstrip('\n')
            counter += 1

    fileContent = fileContent[:counter]

    # Specify the line number and the new text
    lineNumber = 17  # MATLAB is 1-based
    newText = f'var ExportFileName="EXPORT_sim_{id}.DAT";'

    # Update the specific line
    if lineNumber <= len(fileContent):
        fileContent[lineNumber - 1] = newText
    else:
        raise ValueError("Line number exceeds the number of lines in the file.")

    # Write the modified content to the new file
    with open(newfilename, 'w') as f:
        for line in fileContent:
            f.write(line + '\n')


def set_hoofcn(LTCA_list:List[List[LTCA]]):
    # LTCA list is parallelized: 
    #   1st dimension -> decision variables parallelization
    #   2nd dimension -> load levels parallelization
    # Hook fcn needs to have a different handle for each load level

    path = LTCA_list[0][0].orignal_dir
    n = len(LTCA_list[0])
    for ii in range(0, n):
        src = rf"{path}/{'postexportgearcontactresults_hookfns.cmd'}"
        dst = rf"{LTCA_list[ii][0].dir}/{LTCA_list[ii][0].calyx_export_script}"
    return

def init_LTCA():
    """
    Instantiation of multyx interface and strings preparation for multyx commands
    """
    return

def set_design_data():
    """
    Copying of settings that define the design geometry (mainly for facemilled bevel gears)
    """
    return

def launch_LTCA():
    """
    Basically a "STARTANAL" of T3D. we ll see if it is really necessary
    """
    return

def exit_LTCA():
    """
    Close of multyx interface
    """
    return

def post_proc_T3D(LTCA_list, ii,
                    readPinContact=False, readGearContact=False,
                    pinBendingStatus=False, gearBendingStatus=False,
                    pinPressureStatus=False, gearPressureStatus=False,
                    LTEStatus=False):

    s = ""

    # POST-PROCESS
    s += f"POSTPROC\n"
    s += f"POSTPROCFILENAME {LTCA_list[ii].post} OK\n"

    # CONTACT
    if readPinContact:
        s += (
            f"CONTACT\n"
            f"MEMBER {LTCA_list[ii].pin.member}\n"
            f"OUTPUTTOFILE TRUE\n"
            f"EDGECONTACT TRUE\n"
            f"AUTOTOOTH TRUE\n"
            f"FILENAME {LTCA_list[ii].pin.contact}\n"
            f"BEGINSTEP 1\n"
            f"ENDSTEP {LTCA_list[ii].ntimesteps}\n"
            f"START\n"
            f"EXIT\n"
        )

    if readGearContact:
        s += (
            f"CONTACT\n"
            f"MEMBER {LTCA_list[ii].gear.member}\n"
            f"OUTPUTTOFILE TRUE\n"
            f"EDGECONTACT TRUE\n"
            f"AUTOTOOTH TRUE\n"
            f"FILENAME {LTCA_list[ii].gear.contact}\n"
            f"BEGINSTEP 1\n"
            f"ENDSTEP {LTCA_list[ii].ntimesteps}\n"
            f"START\n"
            f"EXIT\n"
        )

    # BENDING
    if pinBendingStatus:
        s += (
            f"SEARCHSTRESS\n"
            f"BODY {LTCA_list[ii].pin.member}\n"
            f"COMPONENT MAXPPLSTRESS\n"
        )
        if LTCA_list[ii].side.lower() == 'drive' and LTCA_list[ii].systemHand.lower() == 'right':
            s += "SURFACE HYPOIDFILL_1_1_2\n"
        else:
            s += "SURFACE HYPOIDFILL_1_1_1\n"

        s += (
            f"AUTOTOOTH TRUE\n"
            f"NUMSPROF 31\n"
            f"NUMTFACE 21\n"
            f"DISTMIN 3\n"
            f"BEGINSTEP 1\n"
            f"ENDSTEP {LTCA_list[ii].ntimesteps}\n"
            f"COMPONENT MAXPPLSTRESS\n"
            f"OUTPUTTOFILE TRUE\n"
            f"FILENAME {LTCA_list[ii].pin.bending}\n"
            f"SEPTEETH TRUE\n"
            f"START\n"
            f"EXIT\n"
        )

    if gearBendingStatus:
        s += (
            f"SEARCHSTRESS\n"
            f"BODY {LTCA_list[ii].gear.member}\n"
            f"COMPONENT MAXPPLSTRESS\n"
        )
        if LTCA_list[ii].side.lower() == 'drive' and LTCA_list[ii].systemHand.lower() == 'right':
            s += "SURFACE HYPOIDFILL_2_1_2\n"
        else:
            s += "SURFACE HYPOIDFILL_2_1_1\n"

        s += (
            f"AUTOTOOTH TRUE\n"
            f"NUMSPROF 31\n"
            f"NUMTFACE 21\n"
            f"DISTMIN 3\n"
            f"BEGINSTEP 1\n"
            f"ENDSTEP {LTCA_list[ii].ntimesteps}\n"
            f"COMPONENT MAXPPLSTRESS\n"
            f"OUTPUTTOFILE TRUE\n"
            f"FILENAME {LTCA_list[ii].gear.bending}\n"
            f"SEPTEETH TRUE\n"
            f"START\n"
            f"EXIT\n"
        )

    # PATTERNS
    s += "PATTERN\n"

    if pinPressureStatus:
        s += (
            f"PATTERNCOMPONENT CONTACTPRESSURE\n"
            f"EDGECONTACT TRUE\n"
            f"CONTOURS FALSE\n"
            f"MEMBER {LTCA_list[ii].pin.member}\n"
            f"TOOTHBEGIN {LTCA_list[ii].pin.tooth_begin}\n"
            f"TOOTHEND {LTCA_list[ii].pin.tooth_end}\n"
            f"BEGINSTEP 1\n"
            f"ENDSTEP {LTCA_list[ii].ntimesteps}\n"
            f"SMOOTH FALSE\n"
            f"OUTPUTTOFILE TRUE\n"
            f"FILENAME {LTCA_list[ii].pin.pattern}\n"
            f"START\n"
        )

    if gearPressureStatus:
        s += (
            f"PATTERNCOMPONENT CONTACTPRESSURE\n"
            f"EDGECONTACT TRUE\n"
            f"CONTOURS FALSE\n"
            f"MEMBER {LTCA_list[ii].gear.member}\n"
            f"TOOTHBEGIN {LTCA_list[ii].gear.tooth_begin}\n"
            f"TOOTHEND {LTCA_list[ii].gear.tooth_end}\n"
            f"BEGINSTEP 1\n"
            f"ENDSTEP {LTCA_list[ii].ntimesteps}\n"
            f"SMOOTH FALSE\n"
            f"OUTPUTTOFILE TRUE\n"
            f"FILENAME {LTCA_list[ii].gear.pattern}\n"
            f"START\n"
        )

    s += "EXIT\n"  # EXIT PATTERN

    # TRANSMISSION ERROR (BODY DEFLECTION)
    if LTEStatus:
        s += "BODYDEFLECTION\n"
        if LTCA_list[ii].side.lower() == 'drive':
            s += f"BODY {LTCA_list[ii].gear.member}_ROTOR\n"
        else:
            s += f"BODY {LTCA_list[ii].pin.member}_ROTOR\n"

        s += (
            f"COMPONENT THETAZ\n"
            f"OUTPUTTOFILE TRUE\n"
            f"FILENAME {LTCA_list[ii].thetaz}\n"
            f"START\n"
            f"EXIT\n"
        )

    # Exit
    s += "EXIT\n"  # EXIT POSTPROC

    return s

def settings_script_T3D(data:DesignData, LTCA_list: List[LTCA], shaft_export = False):

    pin_hand = data.system_data.hand
    gear_hand = 'LEFT'
    if pin_hand.lower() == 'LEFT':
        gear_hand = 'RIGHT'

    script = ""
    script += f"SESFILENAME \n {LTCA_list.ses}"
    indent_level = "\t"*1
    script += "EDIT \n"
    script += indent_level + "PAIRS \n"
    indent_level = "\t"*2
    script += indent_level + f"HAND_PINION {data.system_data.hand.upper()} \n"
    indent_level = "\t"*2
    script += indent_level + "EXIT \n"
    script += indent_level + "ROTOR \n"
    
    rotor_ID = {"PINION": 1, "GEAR": 2}
    HANDS = {"PINION": pin_hand, "GEAR": gear_hand}
    for member in ["PINION", "GEAR"]:

        common_field = data.get_common_field(member.lower())

        zA = common_field.zROOTTOE
        RA = common_field.RROOTTOE
        zB = common_field.zROOTHEEL
        RB = common_field.RROOTHEEL
        DiA = common_field.ShaftDiA
        DiB = common_field.ShaftDiB
        L = zB - zA
        DoA = RA*2 # external diameter of the shaft at the toe
        DoB = RB*2 # external diameter of the shaft at the heel

        script += indent_level + f"ROTOR {rotor_ID[member]} \n"

        if shaft_export:
            script += indent_level + "SHAFT \n"
            indent_level = "\t"*3

            script += (
                f"{indent_level}SHAFT 1\n"
                f"{indent_level}AXIALPOSNSHAFT {zA}\n"
                f"{indent_level}SEGMENT \n"
            )

            indent_level = "\t"*4

            script += (
                f"{indent_level}SEGMENT 1\n"
                f"{indent_level}TYPESEG DEFINEGEOMETRY\n"
                f"{indent_level}LENGTH {L}\n"
                f"{indent_level}OUTERSHAPE CONICAL\n"
                f"{indent_level}USECONEANGLEOUTER FALSE\n"
                f"{indent_level}OUTERDIA1 {DoA}\n"
                f"{indent_level}OUTERDIA2 {DoB}\n"
                f"{indent_level}INNERSHAPE CONICAL\n"
                f"{indent_level}USECONEANGLEINNER FALSE\n"
                f"{indent_level}INNERDIA1 {DiA}\n"
                f"{indent_level}INNERDIA2 {DiB}\n"
            )

            indent_level = "\t"*3

            script += indent_level + "EXIT\n"   # EXIT SEGMENT
        
        indent_level = "\t"*2
        script += indent_level + "EXIT\n"    # EXIT SHAFT

        script += indent_level + "HYPOID\n"
        
        indent_level = "\t"*3
        script += (
            f"{indent_level}NHYPOIDS 1\n"
            f"{indent_level}HYPOID 1\n"
            f"{indent_level}TYPE FACEMILLED_{member}\n"
        )

        if member == 'GEAR':
            script += indent_level + f"GEAR_TYPE {data.gear_common_data.gen_type}\n"
        
        script += indent_level + "COMMON \n"
        
        indent_level = "\t"*4
        script += (
            f"{indent_level}NTEETH {common_field.NTEETH}\n"
            f"{indent_level}HAND {HANDS[member]}\n"
            f"{indent_level}SPIRALANGLE {common_field.SPIRALANGLE}\n"
            f"{indent_level}OUTERCONEDIST {common_field.OUTERCONEDIST}\n"
            f"{indent_level}FACEWIDTH {common_field.FACEWIDTH}\n"
            f"{indent_level}FACEANGLE {common_field.FACEANGLE}\n"
            f"{indent_level}BACKANGLE {common_field.BACKANGLE}\n"
            f"{indent_level}FRONTANGLE {common_field.FRONTANGLE}\n"
            f"{indent_level}PITCHANGLE {common_field.PITCHANGLE}\n"
            f"{indent_level}PITCHAPEX {common_field.PITCHAPEX}\n"
            f"{indent_level}FACEAPEX {common_field.FACEAPEX}\n"
            f"{indent_level}ROOTAPEX {common_field.ROOTAPEX}\n"
            f"{indent_level}BASECONEANGLE {common_field.BASECONEANGLE}\n"
            f"{indent_level}BASECONEAPEX {common_field.BASECONEAPEX}\n"
        )

        if common_field.USE_SPRD_BLD_THICKNESS == False:
            script += indent_level + "USE_SPRD_BLD_THICKNESS FALSE\n"
            if common_field.NORMAL_THICKNESS is not None:
                script += indent_level + "TYPE_THICKNESS NORMALCHORDAL\n"
                script += indent_level + f"NORMAL_THICKNESS {common_field.NORMAL_THICKNESS}\n"
        else:
            script += indent_level + "USE_SPRD_BLD_THICKNESS TRUE\n"

        indent_level = "\t"*3
        script += indent_level + "EXIT  //EXIT COMMON \n"

        # cycle through concave and convex settings
        flanks = ["CONCAVE", "CONVEX"]

        for flank in flanks:

            machine_settings = data.get_machine_field(member, flank)
            indent_level = "\t"*4
            script += indent_level + "MACHINE \n"

            indent_level = "\t"*5

            if member == "GEAR" and common_field.gen_type.lower() == 'formate':
                script += (
                    f"{indent_level}HORIZONTAL {machine_settings.Horizontal}\n"
                    f"{indent_level}MACHCTRBACK {machine_settings.MACHCTRBACK}\n"
                    f"{indent_level}ROOTANGLE {machine_settings.ROOTANGLE}\n"
                )
            else: # generated member
                script += (
                    f"{indent_level}RADIALSETTING {machine_settings.RADIALSETTING}\n"
                    f"{indent_level}TILTANGLE {machine_settings.TILTANGLE}\n"
                    f"{indent_level}SWIVELANGLE {machine_settings.SWIVELANGLE}\n"
                    f"{indent_level}BLANKOFFSET {machine_settings.BLANKOFFSET}\n"
                    f"{indent_level}ROOTANGLE {machine_settings.ROOTANGLE}\n"
                    f"{indent_level}MACHCTRBACK {machine_settings.MACHCTRBACK}\n"
                    f"{indent_level}SLIDINGBASE {machine_settings.SLIDINGBASE}\n"
                    f"{indent_level}CRADLEANGLE {machine_settings.CRADLEANGLE}\n"
                    f"{indent_level}RATIOROLL {machine_settings.RATIOROLL}\n"
                    f"{indent_level}MODROLL_2C {machine_settings.C2}\n"
                    f"{indent_level}MODROLL_6D {machine_settings.D6}\n"
                    f"{indent_level}MODROLL_24E {machine_settings.E24}\n"
                    f"{indent_level}MODROLL_120F {machine_settings.F120}\n"
                    f"{indent_level}MODROLL_720G {machine_settings.G720}\n"
                    f"{indent_level}MODROLL_5040H {machine_settings.H5040}\n"
                    f"{indent_level}H1 {machine_settings.H1}\n"
                    f"{indent_level}H2 {machine_settings.H2}\n"
                    f"{indent_level}H3 {machine_settings.H3}\n"
                    f"{indent_level}H4 {machine_settings.H4}\n"
                    f"{indent_level}H5 {machine_settings.H5}\n"
                    f"{indent_level}H6 {machine_settings.H6}\n"
                    f"{indent_level}V1 {machine_settings.V1}\n"
                    f"{indent_level}V2 {machine_settings.V2}\n"
                    f"{indent_level}V3 {machine_settings.V3}\n"
                    f"{indent_level}V4 {machine_settings.V4}\n"
                    f"{indent_level}V5 {machine_settings.V5}\n"
                    f"{indent_level}V6 {machine_settings.V6}\n"
                    f"{indent_level}R1 {machine_settings.R1}\n"
                    f"{indent_level}R2 {machine_settings.R2}\n"
                    f"{indent_level}R3 {machine_settings.R3}\n"
                    f"{indent_level}R4 {machine_settings.R4}\n"
                    f"{indent_level}R5 {machine_settings.R5}\n"
                    f"{indent_level}R6 {machine_settings.R6}\n"
                )
            
            indent_level = "\t"*4
            script += (
                f"{indent_level} "
                f"{indent_level}EXIT\n"
                f"{indent_level}// EXIT MACHINE\n"
                f"{indent_level}CUTTER\n"
            )

            indent_level = "\t"*5
            cutter_settings = data.get_tool_field(member, flank)

            if member == 'GEAR' and data.gear_common_data.gen_type.lower() == 'formate':
                RPcnv = data.gear_cutter_data.concave.POINTRADIUS
                RPcvx = data.gear_cutter_data.convex.POINTRADIUS
                PRmean = (RPcnv + RPcvx)*0.5
                PW = RPcnv - RPcvx

                script += (
                    f"{indent_level}POINTRADIUS {cutter_settings.POINTRADIUS}\n"
                    f"{indent_level}BLADEANGLE {cutter_settings.BLADEANGLE}\n"
                    f"{indent_level}EDGERADIUS {cutter_settings.EDGERADIUS}\n"
                    f"{indent_level}POINTWIDTH {PW}\n"
                    f"{indent_level}USENEWCUTTER TRUE\n"
                    f"{indent_level}TYPE {cutter_settings.TYPE}\n"
                    f"{indent_level}TOPREM_OPTION {cutter_settings.topremTYPE}\n"
                    f"{indent_level}FLANKREM_OPTION {cutter_settings.flankremTYPE}\n"
                )

            else: # non formate data

                script += (
                    f"{indent_level}POINTRADIUS {cutter_settings.POINTRADIUS}\n"
                    f"{indent_level}BLADEANGLE {cutter_settings.BLADEANGLE}\n"
                    f"{indent_level}EDGERADIUS {cutter_settings.EDGERADIUS}\n"
                    f"{indent_level}USENEWCUTTER TRUE\n"
                    f"{indent_level}TYPE {cutter_settings.TYPE}\n"
                    f"{indent_level}TOPREM_OPTION {cutter_settings.topremTYPE}\n"
                    f"{indent_level}FLANKREM_OPTION {cutter_settings.flankremTYPE}\n"
                )

            # toprem data
            if cutter_settings.topremTYPE.lower() == 'blended':
                script += indent_level + f"TOPREM_DEPTH {cutter_settings.topremDEPTH}\n"
                script += indent_level + f"TOPREM_BLEND_RADIUS {cutter_settings.topremRADIUS}\n"
            elif cutter_settings.topremTYPE.lower() == 'straight':
                script += indent_level + f"TOPREM_DEPTH {cutter_settings.topremDEPTH}\n"
                script += indent_level + f"TOPREM_ANGLE {cutter_settings.topremANGLE}\n"

            # blade data
            if cutter_settings.TYPE.lower() == 'curved':
                script += indent_level + f"RHO {cutter_settings.RHO}\n"

            # flankrem data
            if cutter_settings.flankremTYPE.lower() == 'blended':
                script += indent_level + f"FLANKREM_DEPTH {cutter_settings.flankremDEPTH}\n"
                script += indent_level + f"FLANKREM_BLEND_RADIUS {cutter_settings.flankremRADIUS}\n"
            elif cutter_settings.topremTYPE.lower() == 'straight':
                script += indent_level + f"FLANKREM_DEPTH {cutter_settings.flankremDEPTH}\n"
                script += indent_level + f"FLANKREM_ANGLE {cutter_settings.flankremANGLE}\n"
            
            indent_level = "\t"*4
            script += indent_level + "EXIT\n"
            script += indent_level + f"// EXIT CUTTER\n"

            indent_level = "\t"*3
            script += indent_level + "EXIT\n"
            script += indent_level + f"// EXIT {flank}\n"

        # END CONCAVE-CONVEX cycle

        indent_level = "\t"*2
        script += indent_level + "EXIT\n"
        script += indent_level + "//EXIT HYPOID\n"

    # END MEMBER cycle
    indent_level = "\t"*1
    script += indent_level + "EXIT\n"
    script += indent_level + "//EXIT ROTOR\n"

    # ses saving will be performed outside

    return script

def settings_script_T3D_msh(data:DesignData, LTCA_list: List[LTCA], shaft_export = False):
    script = " "
    return script

def init_multyx_types(path, multyxlib):
    # multyx.dll is a shared library that exposes a few standard C functions.
    # ctypes is a python package for calling C functions.
    # Load multyx.dll using the python ctypes.cdll.LoadLibrary() function:
    
    # In order to tell python how to call each function in the DLL,
    # specify the python-C interface of thes functions:
    #
    # void* OpenMultyxSession(
    #   char* SessionFileName,
    #   MsgCallbackFunctionPointer ptr_infocallback,
    #   MsgCallbackFunctionPointer ptr_errorcallback,
    #   MsgCallbackFunctionPointer ptr_warningcallback
    # );
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)
    multyxlib.OpenMultyxSession.restype=ctypes.c_void_p
    multyxlib.OpenMultyxSession.argtypes=[
    ctypes.c_char_p,
    ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p),
    ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p),
    ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
    ]
    #
    #  Interface for setting values of multyx variables.
    # int SetValueFloatingPointVariable(void* SessionHandle,char* InputSpecifier,double Value);
    multyxlib.SetValueFloatingPointVariable.restype=ctypes.c_int
    multyxlib.SetValueFloatingPointVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_double]
    # int SetValueBoolVariable(void* SessionHandle,char* InputSpecifier,bool Value);
    multyxlib.SetValueBoolVariable.restype=ctypes.c_int
    multyxlib.SetValueBoolVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_bool]
    # int SetValueSwitchVariable(void* SessionHandle,char* InputSpecifier,int Value);
    multyxlib.SetValueSwitchVariable.restype=ctypes.c_int
    multyxlib.SetValueSwitchVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int]
    # int SetValueIntegerVariable(void* SessionHandle,char* InputSpecifier,int Value);
    multyxlib.SetValueIntegerVariable.restype=ctypes.c_int
    multyxlib.SetValueIntegerVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int]
    # int SetValueStringVariable(void* SessionHandle,char* InputSpecifier,char* Value);
    multyxlib.SetValueStringVariable.restype=ctypes.c_int
    multyxlib.SetValueStringVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_char_p]
    # int ExecuteScript(void* SessionHandle,char* Script,int ShowOutput);
    multyxlib.ExecuteScript.restype=ctypes.c_int
    multyxlib.ExecuteScript.argtypes=[ctypes.c_void_p,ctypes.c_char_p]
    #  Interface for getting values of multyx variables.
    # int GetValueFloatingPointTaggedItem(void* SessionHandle,char* OutputSpecifier,double* Value);
    multyxlib.GetValueFloatingPointTaggedItem.restype=ctypes.c_int
    multyxlib.GetValueFloatingPointTaggedItem.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_double)]
    # int GetValueBoolVariable(void* SessionHandle,char* OuputSpecifier,bool* Value);
    multyxlib.GetValueBoolVariable.restype=ctypes.c_int
    multyxlib.GetValueBoolVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_bool)]
    # int GetValueSwitchVariable(void* SessionHandle,char* OuputSpecifier,int* Value);
    multyxlib.GetValueSwitchVariable.restype=ctypes.c_int
    multyxlib.GetValueSwitchVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_int)]
    # int GetValueIntegerVariable(void* SessionHandle,char* OuputSpecifier,int* Value);
    multyxlib.GetValueIntegerVariable.restype=ctypes.c_int
    multyxlib.GetValueIntegerVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_int)]
    multyxlib.GetValueStringVariable.restype=ctypes.c_int
    # Call GetValueStringVariable() like this:
    # bufsize=1024
    # buf= (ctypes.c_char * buf_size)()
    # pbuf= (ctypes.POINTER(ctypes.c_char) * 1)(buf)
    # retval=multyxlib.GetValueStringVariable(SessionHandle,b"DESCRIPTION",pbuf,bufsize)
    multyxlib.GetValueStringVariable.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),ctypes.c_int]
    multyxlib.CloseMultyxSession.restype=ctypes.c_int
    multyxlib.CloseMultyxSession.argtypes=[ctypes.c_void_p]
    #
    # Default Null callback. Use this if you do not wish to process
    # Informational, Error, or Warning messages:
    # NullMsgCallback=ctypes.cast(None,ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p))    
    #
    # ShowOutput=1 if you want to see the output from multyx, =0 otherwise
    ShowOutput=ctypes.c_int(0) #Don't show output
    # os.chdir(current_path)
    return multyxlib

# Take a message as parameter, and return a bool DoAboort
@ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
def InfoCallBack(msg):
    #print(msg.decode("utf-8"))
    # Do not abort session:
    return ctypes.c_bool(False)

# Take a message as parameter, and return a bool DoAboort
@ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
def ErrorCallBack(msg):
    print(msg.decode("utf-8"))
    # Do not abort when an error message is received:
    return ctypes.c_bool(False)

# Take a message as parameter, and return a bool DoAboort
@ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
def WarningCallBack(msg):
    print(msg.decode("utf-8"))
    # Do not abort session:
    return ctypes.c_bool(False)

def init_multyx_session(path, ses_file, multyxlib):
    #sys.path.insert(0,'/opt/ansol/calyx')
    
    # Start a session by passing it the name of the session file to read.
    # It will acquire the license, open the session file and wait.
    # It will return a SessionHandle of type ctypes.c_voidp, which is needed
    # for all subsequent calls to the opened session.
    SessionFileName=bytes(ses_file, encoding='utf-8')
    SessionHandle=multyxlib.OpenMultyxSession(SessionFileName, InfoCallBack, ErrorCallBack, WarningCallBack)
    if SessionHandle==ctypes.c_voidp(None):
        print("Failed to start session")
        return 1
    return (multyxlib, SessionHandle)

def init_process(path, ses_file, ID):
    parent_conn, child_conn = multiprocessing.Pipe()
    worker = multiprocessing.Process(target=worker_process, args=(child_conn, ID, path, ses_file))

    # start worker
    worker.start()

    # Wait for worker to initialize
    init_msg = parent_conn.recv()

    return parent_conn, worker, init_msg

def worker_process(conn, worker_id, path, ses_file):
    """
    This function is run by each worker process. It initializes the multyx interface and
    session, and then waits for parameters to be sent through the pipe. When a parameter
    is received, it performs a task and sends the result back to the parent process.
    This process continues until the parent sends the 'exit' command, at which point the worker shuts down.
    """

    def LOG(msg):
        msg = f'Worker {worker_id}: ' + msg
        # print(msg)
        conn.send(msg)
        return
    # init hypoid
    ...
    # init path associated with the worker id
    ...
    # initialize multyx interfaces
    library=ctypes.cdll.LoadLibrary("C:/Program Files/Ansol/Transmission3Dx64/multyx.dll")
    library = init_multyx_types(path, library)
    interface, seshandle = init_multyx_session(path, ses_file, library)
    LOG(f"worker {worker_id} initialized")
    

    while True:
        # Wait to receive data from the pipe
        param = conn.recv()

        # identify hypoid machine-tool settings to best match the ease-off defined by the parameters
        
        if isinstance(param, str) and param.lower() == "exit":
            LOG("Shutting down.")
            interface.CloseMultyxSession(seshandle)

            break

        # Perform a task (e.g., square the parameter)
        log_string = f"Received parameter {param}\n"
        log_string += "Starting simulation\n"

        t = time.time()
        ret = interface.ExecuteScript(seshandle, param)

        log_string += "Sending result\n"
        log_string += f"Simulation completed successfully in {time.time() - t} seconds!"

        LOG(log_string)

        conn.send(ret)
    
    conn.close()  # Close the connection when done


