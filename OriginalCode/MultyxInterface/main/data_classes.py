import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np


@dataclass
class SystemData:
    hand: Any = None 
    pin_nteeth: Any = None
    gear_nteeth: Any = None
    design_data: Any = None 
    msh_geometry: List = None # ["pinion.msh", "gear.msh"]

@dataclass
class MemberData:
    member: Any = None
    rze: Any = None
    nteeth: Any = None
    torque: Any = None
    rpm: Any = None
    pattern: Any = None
    patterndata: Any = None
    tooth_begin: Any = None
    tooth_end: Any = None
    contact: Any = None
    bending: Any = None
    material: Any = None
    pattern_no_edge: Any = None

@dataclass
class MemberResults:
    pressure_pattern: Any = None
    contact: Any = None
    bending: Any = None
    

@dataclass
class Results:
    pinion: MemberResults = field(default_factory=MemberResults)
    gear: MemberResults = field(default_factory=MemberResults)
    LTE: dict = field(default_factory=dict)
    efficiency: dict = field(default_factory=dict)

@dataclass
class LTCA:
    id: int = None
    dir: str = None
    orignal_dir: str = None
    ses: str = None
    script: str = None
    memory: int = None
    ntimesteps: int = None
    nthreads: int = None
    nprofdivs: int = None
    nfacedivs: int = None
    ses_type: str = None
    EPGalpha: List[float] = None
    post: str = None
    export: str = None
    side: str = None
    calyx_export_script: str = None
    backlash_file: str = None
    thetaz: str = None
    system_hand: str = None
    pin: MemberData = field(default_factory=MemberData)
    gear: MemberData = field(default_factory=MemberData)
    deltatime: float = None
    initialtime: float = None
    status_checks: dict = None
    simulate_worker: bool = True
    results: Results = field(default_factory=Results)



