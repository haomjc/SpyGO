"""
Enhanced gear design workflow with flexible rack integration

This module provides a cleaner workflow for gear generation with complex rack cutters,
addressing the envelope theory requirements while maintaining flexibility.
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import casadi as ca

from gears.main.data_structs import rackData
from gears.main.core import rackCutter


@dataclass
class GearParameters:
    """Complete gear parameters for generation"""
    module: float
    teeth_count: int
    profile_shift: float = 0.0
    pressure_angle: float = 20.0  # degrees
    
    # Optional overrides
    addendum_coeff: float = 1.0
    dedendum_coeff: float = 1.25
    
    def __post_init__(self):
        # Derived properties
        self.pitch_radius = self.module * self.teeth_count / 2
        self.addendum_radius = self.pitch_radius + self.addendum_coeff * self.module
        self.dedendum_radius = self.pitch_radius - self.dedendum_coeff * self.module


class GearDesigner:
    """
    Main class for gear design workflow with rack cutter integration
    
    Usage:
        designer = GearDesigner(module=2.0, teeth_count=24)
        designer.set_custom_rack(my_rack_data)  # or use_standard_rack()
        gear = designer.generate_gear()
    """
    
    def __init__(self, module: float, teeth_count: int, profile_shift: float = 0.0):
        self.params = GearParameters(
            module=module, 
            teeth_count=teeth_count, 
            profile_shift=profile_shift
        )
        self._rack_cutter: Optional[rackCutter] = None
        self._generated_gear = None
        
    def use_standard_rack(self, pressure_angle: float = 20.0, **rack_kwargs) -> 'GearDesigner':
        """
        Configure with standard rack cutter
        
        Parameters:
        -----------
        pressure_angle : float
            Rack pressure angle in degrees
        **rack_kwargs : dict
            Additional rack parameters (rc, rho_B, rho_F, etc.)
        """
        rack_data = rackData(
            m=self.params.module,
            alpha=pressure_angle,
            prof_shift=self.params.profile_shift,
            **rack_kwargs
        )
        self._rack_cutter = rackCutter(rack_data)
        self._generated_gear = None  # Reset if parameters changed
        return self
        
    def set_custom_rack(self, rack_data: rackData) -> 'GearDesigner':
        """
        Configure with custom rack cutter data
        
        Parameters:
        -----------
        rack_data : rackData
            Complete rack geometry specification
        """
        # Validate compatibility
        if abs(rack_data.m - self.params.module) > 1e-6:
            raise ValueError(
                f"Rack module ({rack_data.m}) doesn't match gear module ({self.params.module})"
            )
            
        self._rack_cutter = rackCutter(rack_data)
        self._generated_gear = None  # Reset if parameters changed
        return self
        
    def set_rack_cutter(self, rack_cutter_instance: rackCutter) -> 'GearDesigner':
        """
        Use pre-configured rack cutter instance
        
        Parameters:
        -----------
        rack_cutter_instance : rackCutter
            Already configured rack cutter
        """
        # Validate compatibility
        if abs(rack_cutter_instance.data.m - self.params.module) > 1e-6:
            raise ValueError("Rack cutter module doesn't match gear module")
            
        self._rack_cutter = rack_cutter_instance
        self._generated_gear = None
        return self
        
    def preview_rack(self, plot_normals: bool = False):
        """Preview the current rack cutter geometry"""
        if self._rack_cutter is None:
            raise ValueError("No rack cutter configured. Call use_standard_rack() or set_custom_rack() first.")
        
        self._rack_cutter.plot(plot_normals=plot_normals)
        
    def generate_gear(self, method: str = 'envelope') -> 'EnhancedGear':
        """
        Generate gear using specified method
        
        Parameters:
        -----------
        method : str
            Generation method ('envelope', 'involute_direct')
        
        Returns:
        --------
        EnhancedGear
            Generated gear with surface and analysis methods
        """
        if self._rack_cutter is None:
            # Auto-create standard rack
            self.use_standard_rack()
            
        if method == 'envelope':
            self._generated_gear = self._generate_by_envelope()
        elif method == 'involute_direct':
            self._generated_gear = self._generate_involute_direct()
        else:
            raise ValueError(f"Unknown generation method: {method}")
            
        return self._generated_gear
        
    def _generate_by_envelope(self) -> 'EnhancedGear':
        """Generate gear using envelope theory with rack cutter"""
        return EnhancedGear(
            parameters=self.params,
            rack_cutter=self._rack_cutter,
            generation_method='envelope'
        )
        
    def _generate_involute_direct(self) -> 'EnhancedGear':
        """Generate standard involute gear directly (for comparison)"""
        return EnhancedGear(
            parameters=self.params,
            rack_cutter=self._rack_cutter,
            generation_method='involute_direct'
        )
        
    @property
    def gear(self) -> 'EnhancedGear':
        """Get generated gear (generates if not done yet)"""
        if self._generated_gear is None:
            return self.generate_gear()
        return self._generated_gear
        
    @property  
    def rack_cutter(self) -> rackCutter:
        """Get current rack cutter"""
        if self._rack_cutter is None:
            raise ValueError("No rack cutter configured")
        return self._rack_cutter


class EnhancedGear:
    """
    Enhanced gear class with surface generation and analysis capabilities
    
    This replaces the minimal Gear class with full functionality
    """
    
    def __init__(self, parameters: GearParameters, rack_cutter: rackCutter, 
                 generation_method: str = 'envelope'):
        self.params = parameters
        self.rack_cutter = rack_cutter
        self.generation_method = generation_method
        
        # Surface generation parameters
        self.nProf = 22  # Profile discretization
        self.nFace = 20  # Face width discretization  
        self.nFillet = 11  # Fillet discretization
        
        # Generate surfaces
        self._generate_tooth_surface()
        
    def _generate_tooth_surface(self):
        """Generate tooth surface using envelope theory"""
        if self.generation_method == 'envelope':
            self._envelope_generation()
        elif self.generation_method == 'involute_direct':
            self._involute_generation()
            
    def _envelope_generation(self):
        """
        Implement envelope theory for gear generation
        
        This is where your main mathematical work goes:
        1. Parameterize rack motion
        2. Apply envelope conditions
        3. Generate tooth surface points
        """
        # TODO: Implement your envelope theory calculations
        # This would involve:
        # - Rolling motion parameterization
        # - Envelope condition (∂r/∂φ · v = 0)
        # - Surface point calculation
        pass
        
    def _involute_generation(self):
        """Generate standard involute profile for comparison"""
        # Standard involute generation
        pass
        
    def plot_tooth_surface(self, tooth_number: int = 0):
        """Plot 3D tooth surface"""
        # Implementation for 3D surface plotting
        pass
        
    def plot_profile_comparison(self):
        """Compare generated profile with theoretical involute"""
        pass
        
    def analyze_contact_pattern(self, mating_gear: 'EnhancedGear' = None):
        """Analyze gear mesh contact patterns"""
        pass


# Convenience functions for common workflows
def create_standard_gear(module: float, teeth_count: int, profile_shift: float = 0.0) -> EnhancedGear:
    """Quick creation of standard involute gear"""
    return (GearDesigner(module, teeth_count, profile_shift)
            .use_standard_rack()
            .generate_gear())


def create_custom_gear(module: float, teeth_count: int, rack_data: rackData, 
                      profile_shift: float = 0.0) -> EnhancedGear:
    """Quick creation of gear with custom rack"""
    return (GearDesigner(module, teeth_count, profile_shift)
            .set_custom_rack(rack_data)
            .generate_gear())


# Example usage workflow
if __name__ == "__main__":
    # Example 1: Standard workflow
    designer = GearDesigner(module=2.0, teeth_count=24, profile_shift=0.1)
    designer.use_standard_rack(pressure_angle=20.0, rc=0.2)
    designer.preview_rack()
    gear = designer.generate_gear()
    
    # Example 2: Custom complex rack  
    custom_rack = rackData(
        m=2.0, alpha=25.0, rc=0.15, 
        rho_B=500.0, rho_F=800.0,  # Complex curvatures
        s_F=1.5, alpha_F=5.0       # Custom chamfer
    )
    
    custom_gear = create_custom_gear(
        module=2.0, teeth_count=30, 
        rack_data=custom_rack, profile_shift=0.05
    )