
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gears.main.core import rackCutter, Gear
from gears.main.data_structs import rackData
m = 1.5
prof_shift=0.2692#0.539#

rack_data = rackData(m=m, alpha=20.0, prof_shift=0, rc=0.35*m, rho_B=9000.0, rho_F=9000, alpha_F=0, s_F=0.5, cut_rack_tip=1.25)
rack = rackCutter().curved_geometry(rack_data)
rack2 = rackCutter().create_standard(module=2.0, pressure_angle=30.0)

# rack.plot(plot_normals=True, internal_gear=True)
# rack2.plot(plot_normals=True, internal_gear=True)

gear = Gear(rack, nZ = 25, profile_shift=prof_shift, internal_gear=False)
gear.plot_tooth(plot_normals=True, debug_colors=False)

cont = input("Continue design? [y/n]: ")
if cont.lower().strip() != "y":
    sys.exit(0)

gear.set_lengthwise_curve(lengthwise_curve_function=-25)

Ep_tip = 50*m/1.5; Ep_root = 0; Ef_toe = 50*m/1.5; Ef_heel = 50*m/1.5# in microns
facewidth = 20
gear.apply_crowning(Ep_tip, Ep_root, Ef_toe, Ef_heel, facewidth, order_prof = 3, order_face = 2)


gear.fit_nurbs(z_range=(-facewidth/2*1.1,facewidth/2*1.1), both_flanks=True, verbose=False, backlash=0.2)



gear.generate_gear_CAD(z_range=(-facewidth/2,facewidth/2), rim_thickness=2.0, filename = 'step_files\m1dot5_n_25.stp')

print(gear.data.outer_radius)
print(gear.data.form_diameter_radius)
print(gear.data.root_radius)

