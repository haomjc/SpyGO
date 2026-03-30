from dataclasses import dataclass, field, asdict
from typing import Literal, Dict
import numpy as np
from math import atan, pi, sqrt, sin , asin
from general_utils import *
import json
from copy import deepcopy

@dataclass
class rackData:
    m: float = 1
    alpha: float = 20.0
    rc: float = 0.2
    prof_shift: float = 0.0
    rho_B: float = 1000.0
    rho_F: float = 1000.0
    s_F: float = 2.0
    alpha_F: float = 0.0
    cut_rack_tip: float = 1.25

@dataclass
class gearData:
    profile_shift: float = 0.0
    nZ: int = 20
    addendum_coeff: float = 1.0
    pitch_radius: float = 0.0
    root_radius: float = 0.0
    form_diameter_radius: float = 0.0
    outer_radius: float = 0.0
    base_radius: float = 0.0

@dataclass
class sampledGearProfile:
    """Sampled gear profile data for efficient plotting and analysis"""
    u_vec: np.ndarray = field(default_factory=lambda: np.array([]))
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    nx: np.ndarray = field(default_factory=lambda: np.array([]))
    ny: np.ndarray = field(default_factory=lambda: np.array([]))
    num_points: int = 0
    
    def __post_init__(self):
        """Update num_points after initialization"""
        if len(self.x) > 0:
            self.num_points = len(self.x)
    
    def is_empty(self) -> bool:
        """Check if the sampled data is empty"""
        return len(self.x) == 0