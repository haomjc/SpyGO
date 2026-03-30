from gears.main.data_structs import rackData, gearData, sampledGearProfile
from gears.main.utilities import nonlinspace, find_curve_intersection, find_gear_end_parameter
from gears.main.ease_off import crowning_RZE
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import screwCalculus as sc
from computational_geometry import interp_arc

class rackCutter:

    def __init__(self, ):

        self.nProf = 22
        self.nFace = 20
        self.nFillet = 11
        
        return

    def curved_geometry(self, rack_data:rackData):
        """
        rack_data: datastruct
        """
        self.data = rack_data
        m = rack_data.m
        alpha = rack_data.alpha
        x = rack_data.prof_shift
        rc = rack_data.rc
        rho_B = rack_data.rho_B
        rho_F = rack_data.rho_F
        alpha_F = ca.fabs(rack_data.alpha_F)
        s_F = rack_data.s_F*m       # we use normalized champfer depth so that is scales
        height_coeff = rack_data.cut_rack_tip
        self.height_coeff = height_coeff

        height = height_coeff * m                  # height of the rack tooth above pitch line
        alpha = alpha * np.pi / 180.0            # convert to radians

        CxB = ca.pi * m / 4 + rho_B * ca.cos(alpha)
        CzB = rho_B * ca.sin(alpha) - x

        # careful: asin argument must be in domain; this matches your matlab expressions
        thetaBladeFin = ca.asin((CzB + (s_F + x)) / rho_B)
        thetaBladeIn  = ca.asin((CzB + rc + x - height) / (rho_B + rc))

        Xf = CxB - ca.cos(thetaBladeIn) * (rho_B + rc)

        thetaFlankin = thetaBladeFin + alpha_F

        CzF = CzB + ca.sin(thetaFlankin) * rho_F - ca.sin(thetaBladeFin) * rho_B
        CxF = CxB + ca.cos(thetaFlankin) * rho_F - ca.cos(thetaBladeFin) * rho_B

        thetaFlankFin = ca.asin((CzF + 1.25 * m + x) / rho_F)

        # curvilinear abscissas
        u1 = Xf
        u2 = u1 + rc * (ca.pi/2 - thetaBladeIn)
        u3 = u2 + rho_B * (thetaBladeFin - thetaBladeIn)
        u4 = u3 + rho_F * (thetaFlankFin - thetaFlankin)

        self.curves_ends = [u1, u2, u3, u4]

        # --- CasADi symbolic variable ---
        u = ca.SX.sym('u')

        # CasADi doesn't allow multiplying boolean SX arrays like MATLAB,
        # so we construct piecewise expressions using ca.if_else.
        # Helper for nested piecewise (returns expression depending on conditions).
        # We'll build from the most inner (topland -> fillet -> blade -> flankrem).

        # boolean conditions (SX boolean expressions)
        topland_cond   = ca.logic_and(u >= 0, u <= u1)
        fillet_cond    = ca.logic_and(u > u1, u <= u2)
        blade_cond     = ca.logic_and(u > u2, u <= u3)
        flankrem_cond  = u > u3  # this is the "else"

        # piecewise angle
        angle_topland = ca.SX(0)
        angle_fillet  = (u - u1) / rc
        angle_blade   = (u - u2) / rho_B + thetaBladeIn
        angle_flankrem= (u - u3) / rho_F + thetaFlankin

        # build nested if_else: if topland then angle_topland else (if fillet ... else (if blade ... else flankrem))
        angle = ca.if_else(topland_cond,
                        angle_topland,
                        ca.if_else(fillet_cond,
                                    angle_fillet,
                                    ca.if_else(blade_cond,
                                                angle_blade,
                                                angle_flankrem)))

        ca_s = ca.cos(angle)
        sa_s = ca.sin(angle)

        # point: 2x1 vector piecewise
        point_topland = ca.vertcat(u, height - x)
        point_fillet  = ca.vertcat(Xf + rc * sa_s, height - x + rc * (ca_s - 1))
        point_blade   = ca.vertcat(CxB - rho_B * ca_s, CzB - rho_B * sa_s)
        point_flankrem= ca.vertcat(CxF - rho_F * ca_s, CzF - rho_F * sa_s)

        point = ca.if_else(topland_cond,
                        point_topland,
                        ca.if_else(fillet_cond,
                                    point_fillet,
                                    ca.if_else(blade_cond,
                                                point_blade,
                                                point_flankrem)))

        # normal: note in your MATLAB blade and flankrem share same expression
        normal_topland = ca.vertcat(0, 1)
        normal_fillet  = ca.vertcat(sa_s, ca_s)
        normal_blade   = ca.vertcat(ca_s, sa_s)
        normal_flankrem= normal_blade

        normal = ca.if_else(topland_cond,
                            normal_topland,
                            ca.if_else(fillet_cond,
                                    normal_fillet,
                                    ca.if_else(blade_cond,
                                                normal_blade,
                                                normal_flankrem)))

        # keep the same named helper points/normals for fillet / flank / flankrem
        toplandP = point_topland
        toplandN = normal_topland

        filletP = point_fillet
        filletN = normal_fillet

        flankP = point_blade
        flankN = normal_blade

        flankremP = point_flankrem
        flankremN = flankN

        curves_points = [toplandP, filletP, flankP, flankremP]
        curves_normals = [toplandN, filletN, flankN, flankremN]
        curves_names = ['topland', 'fillet', 'flank', 'flankrem']

        # --- CasADi functions ---
        self.p_fun = ca.Function('point', [u], [point])
        self.n_fun = ca.Function('normal', [u], [normal])

        self.curves_point_fun = []
        self.curves_normal_fun = []
        for point_expr, normal_expr, name in zip(curves_points, curves_normals, curves_names):
            self.curves_point_fun.append(ca.Function(f'point_{name}', [u], [point_expr]))
            self.curves_normal_fun.append(ca.Function(f'normal_{name}', [u], [normal_expr]))

        return self

    def create_standard(self,module = 1, pressure_angle = 20*np.pi/180, rc = None, height_coeff = 1.25, profile_shift = 0, backlash = 0, parametric = False):
        """Factory method for standard rack"""
        if rc is None:
            rc = 0.2*module

        # standard_data = rackData(m=module, alpha=pressure_angle, prof_shift=0.0, rc=rc, rho_B=4000*module, rho_F=4000*module, alpha_F=0.0, s_F=1.25, cut_rack_tip=1.25)
        # self.curved_geometry(standard_data)
        
        u = ca.SX.sym('u')
        alpha = pressure_angle
        m = module
        x = profile_shift


        if parametric:
            alpha = ca.SX.sym('alpha')
            m = ca.SX.sym('m')
            rc = ca.SX.sym('rc')
            x = ca.SX.sym('x')
            backlash = ca.SX.sym('b')
            params = ca.vertcat(m, alpha, rc, x, backlash)

        calpha = ca.cos(alpha)
        salpha = ca.sin(alpha)
        # parameters
        x_shift = x*m
        height = height_coeff*m
        ybar = height + rc * (salpha - 1)
        u1 = m*np.pi/4 + backlash - rc*calpha - ybar*ca.tan(alpha)
        u2 = (ca.pi/2-alpha)*rc + u1
        u3 = (ybar + 1.0*m)/calpha + u2

        self.curves_ends = [u1, u2, u3]

        xbar = u1 + rc*ca.cos(alpha)
        theta = (u-u1)/rc
        topland_cond   = ca.logic_and(u >= 0, u <= u1)
        fillet_cond    = ca.logic_and(u > u1, u <= u2)
        blade_cond     = u > u2

        stheta = ca.sin(theta)
        ctheta = ca.cos(theta)

        point_topland = ca.vertcat(u,
                                    height-x_shift)
        point_fillet  = ca.vertcat(u1 + rc*stheta,
                                    height +rc*(ctheta - 1)-x_shift)
        point_blade   = ca.vertcat(xbar + (u-u2)*salpha,
                                   ybar - (u-u2)*calpha - x_shift)
        
        point = ca.if_else(topland_cond,
                    point_topland,
                    ca.if_else(fillet_cond,
                            point_fillet,
                            point_blade)
                            )
        
        normal_topland = ca.vertcat(0, 1)
        normal_fillet = ca.vertcat(stheta, ctheta)
        normal_blade = ca.vertcat(calpha, salpha)

        normal = ca.if_else(topland_cond,
                    normal_topland,
                    ca.if_else(fillet_cond,
                            normal_fillet,
                            normal_blade)
                            )

        # --- CasADi functions ---
        if parametric:
            self.p_fun = ca.Function('point', [params, u], [point])
            self.n_fun = ca.Function('normal', [params, u], [normal])
        else:
            self.p_fun = ca.Function('point', [u], [point])
            self.n_fun = ca.Function('normal', [u], [normal])

        return self
    
    def plot(self, plot_normals = False, internal_gear = False, left = False):

        u_end = self.curves_ends[-1]
        u_vec = np.linspace(0, u_end, 100)

        points = self.p_fun(u_vec.reshape(1,-1)).full(); x = points[0,:]; y = points[1,:]

        if internal_gear:
            x = x
            y = -y
        import matplotlib.pyplot as plt
        
        if left:
            x = -x

        plt.figure()  # Create new figure
        plt.plot(x, y, marker = '.')

        if plot_normals:
            normals = self.n_fun(u_vec.reshape(1,-1)).full(); nx = normals[0,:]; ny = normals[1,:]
            if internal_gear:
                nx = nx
                ny = -ny
            if left:
                nx = -nx
            plt.quiver(x, y, nx, ny)

        plt.axis("equal")
        plt.title('Rack Cutter Profile')
        plt.xlabel('X [mm]')
        plt.ylabel('Y [mm]') 
        plt.grid(True, alpha=0.3)
        
        plt.show(block = True)

    def apply_facewidth(self, face_width_function, face_width_ranges):
        """
        face_width_function: callable, takes (z) and returns face width at that z
        face_width_ranges: list of tuples (z_min, z_max) defining the ranges where the face width function is applied
        """
        # check if lengthwise_curve_function is callable
        if not callable(face_width_function) and (isinstance(face_width_function, float) or isinstance(face_width_function, int)):
            # then we assume it is the helix angle for a helical gear
            helix_angle = face_width_function* np.pi/180
            face_width_function = lambda z: z  * np.tan(helix_angle)
        
        # casadi symbolics
        z = ca.SX.sym('z', 1, 1); u = ca.SX.sym('u', 1, 1)
        deltaX_sym = face_width_function(z)
        x_der = ca.jacobian(deltaX_sym, z)


        pR = self.p_fun(u); nR = self.p_fun(u) # right flank profile
        pL = pR * ca.vertcat(-1, 1); nL = nR* ca.vertcat(-1, 1) # left flank profile

        new_point_R = pR+ca.vertcat(deltaX_sym,0)
        new_point_L = pL+ca.vertcat(deltaX_sym,0)

        self.point_fun_3D_R = ca.Function('point_3D_R', [u, z], [ca.vertcat(new_point_R, z)])
        NR = ca.vertcat(nR,
                       -x_der*(nR[1]*nR[0] - nR[0]*nR[1]))
        
        nR = NR / ca.norm_2(NR)
        self.normal_fun_3D_R = ca.Function('normal_3D_R', [u, z], [nR])

        self.point_fun_3D_L = ca.Function('point_3D_L', [u, z], [ca.vertcat(new_point_L, z)])
        NL = ca.vertcat(nL,
                       -x_der*(nL[1]*nL[0] - nL[0]*nL[1]))
        nL = NL / ca.norm_2(NL)
        self.normal_fun_3D_L = ca.Function('normal_3D_L', [u, z], [nL])
        return

    def fit_nurbs(self, z_range=(-10, 10), n_z_points=30, n_u_points=50, 
                  degree_u=4, degree_v=4, control_points_u=40, control_points_v=25,
                  both_flanks=False, verbose=True, backlash = 0.05):
        """
        Fit NURBS surface to 3D rack
        
        Parameters:
        -----------
        z_range : tuple
            (z_min, z_max) for rack width
        n_u_points : int
            Number of points along profile
        n_z_points : int
            Number of points along length
        degree_u, degree_v : int
            NURBS degrees
        control_points_u, control_points_v : int
            Number of control points
        verbose : bool
            Print progress
            
        Returns:
        --------
        dict : Results with fitted NURBS surface
        """

        from nurbs import Nurbs
        if verbose:
            print(f"Fitting NURBS surface to rack...")
        
        # Sample points
        u_vec = np.linspace(0, self.curves_ends[-1]*1.1, n_u_points)
        z_vec = np.linspace(z_range[0], z_range[1], n_z_points)
        
        # Create meshgrid
        U_grid, Z_grid = np.meshgrid(u_vec, z_vec, indexing='ij')
        u_flat = U_grid.reshape(1, -1)
        z_flat = Z_grid.reshape(1, -1)

        if verbose:   
            print("Sampling points...")

        # Use CasADi vectorization - evaluate all points at once
        points_3d_R_flat = self.point_fun_3D_R(u_flat, z_flat).full()
        normals_3d_R_flat = self.normal_fun_3D_R(u_flat, z_flat).full()

        # Extract x, y, z components and reshape to grid
        points_R = np.zeros((3, n_u_points, n_z_points))
        points_R[0, :, :] = points_3d_R_flat[0, :].reshape(n_u_points, n_z_points)  # X
        points_R[1, :, :] = points_3d_R_flat[1, :].reshape(n_u_points, n_z_points)  # Y  
        points_R[2, :, :] = points_3d_R_flat[2, :].reshape(n_u_points, n_z_points)  # Z

        # Generate 3D surface points for left flank using vectorization
        if verbose:
            print("Generating left flank surface points (vectorized)...")
        # Use CasADi vectorization - evaluate all points at once
        points_3d_L_flat = self.point_fun_3D_L(u_flat, z_flat).full()
        normals_3d_L_flat = self.normal_fun_3D_L(u_flat, z_flat).full()
        # Applying backlash
        points_3d_L_flat[0,:] -= backlash

        # Extract x, y, z components and reshape to grid
        points_L = np.zeros((3, n_u_points, n_z_points))
        points_L[0, :, :] = points_3d_L_flat[0, :].reshape(n_u_points, n_z_points)  # X
        points_L[1, :, :] = points_3d_L_flat[1, :].reshape(n_u_points, n_z_points)  # Y
        points_L[2, :, :] = points_3d_L_flat[2, :].reshape(n_u_points, n_z_points)  # Z

        
        if both_flanks:
            # Combine both flanks into a single continuous surface
            if verbose:
                print("Combining flanks into single continuous surface...")
            
            # Flip left flank points in U direction to create continuity
            points_L_flipped = points_L[:, ::-1, :]
            
            # Combine points: right flank + flipped left flank
            # This creates a continuous surface from right flank through root to left flank
            combined_points = np.concatenate([points_L_flipped, points_R], axis=1)
            # we need now to resample the points uniformly
            for ii in range(n_z_points):
                x, y, z = interp_arc(n_u_points*2, combined_points[0,:,ii], combined_points[1,:,ii], combined_points[2,:,ii])
                combined_points[0,:,ii] = x
                combined_points[1,:,ii] = y
                combined_points[2,:,ii] = z

            if verbose:
                print("Plotting points to fit")
                import easy_plot as ep
                X = combined_points[0,:,:].squeeze(); Y = combined_points[1,:,:].squeeze(); Z = combined_points[2,:,:].squeeze()
                F = ep.Figure()
                S = ep.scatter(F, X.reshape(-1,1),Y.reshape(-1,1),Z.reshape(-1,1))
                F.show()


            combined_control_points_u = min(combined_points.shape[1] // 2, control_points_u * 2)
            
            # Fit single NURBS surface to combined points
            if verbose:
                print("Fitting NURBS to combined tooth surface...")
            nurbs_combined = Nurbs()
            nurbs_combined.fit(combined_points, degree_u, degree_v, (combined_control_points_u, control_points_v))
            
                        # Calculate fitting quality metrics
            rms_error = np.sqrt(np.mean(nurbs_combined.fit_residuals**2)) if hasattr(nurbs_combined, 'fit_residuals') else 0
            max_error = np.max(np.abs(nurbs_combined.fit_residuals)) if hasattr(nurbs_combined, 'fit_residuals') else 0
            
            if verbose:
                print(f"Combined surface - RMS error: {rms_error:.6f}, Max error: {max_error:.6f}")
            
            # Store NURBS surface in gear object
            self.nurbs_surfaces = nurbs_combined

        else:
            raise KeyError("Not implemented...")

        return
    
    def generate_CAD(self, z_range=(-10, 10), rim_thickness=None, filename='test_gear.STEP'):

        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.Interface import Interface_Static
        from nurbs import Nurbs_to_OCC_surface

        nurbs_OCC = Nurbs_to_OCC_surface(self.nurbs_surfaces)
        tooth_spacing = BRepBuilderAPI_MakeFace(nurbs_OCC, 1e-6).Face()


        
        step_writer_body = STEPControl_Writer()
        Interface_Static.SetCVal("write.step.schema", "AP203")
        # step_writer_body.Transfer(revolved_shape, STEPControl_AsIs)
        step_writer_body.Transfer(tooth_spacing, STEPControl_AsIs)
        step_writer_body.Write(filename)
        return


class Gear:
    
    def __init__(self, rack_cutter:rackCutter, nZ, profile_shift, internal_gear = False):

        self.nProf = 22
        self.nFace = 20
        self.nFillet = 11

        
        self.rackCutter = rack_cutter
        self.rack_data = self.rackCutter.data
        self.data = gearData()
        self.data.nZ = nZ
        self.internal_gear = internal_gear
        m = self.rack_data.m
        alpha = self.rack_data.alpha
        x = profile_shift*m
        self.data.profile_shift = profile_shift

        r = m*nZ/2     # pitch radius
        r_a = r + x + 1.0*m # addendum radius
        r_r = r + x - self.rackCutter.height_coeff*m # root radius

        if internal_gear:
            self.internal_gear = True
            r_a = r - x - 1.0*m
            r_r = r - x + self.rackCutter.height_coeff*m

        self.data.pitch_radius = r
        self.data.outer_radius = r_a
        self.data.root_radius = r_r
        self.data.base_radius = r * ca.cos(alpha * np.pi / 180.0)
        
        # kinematics
        u = ca.SX.sym('u')      # curvilinear abscissa of rack
        p = self.rackCutter.p_fun(u) - ca.vertcat(0, x)  # shift by profile shift
        n = -self.rackCutter.n_fun(u)

        p_curves = []; n_curves = []
        for curve_point, curve_normal in zip(self.rackCutter.curves_point_fun, self.rackCutter.curves_normal_fun):

            curveP = curve_point(u) - ca.vertcat(0, x)
            curveN = -curve_normal(u)
            if not internal_gear:
                curveP *= ca.vertcat(1, -1)
                curveN *= ca.vertcat(1, -1)
            p_curves.append(curveP)
            n_curves.append(curveN)

        if not internal_gear:
            # External gear: flip coordinates and normal (matching MATLAB)
            p = p * ca.vertcat(1, -1)
            n = n * ca.vertcat(1, -1)

        phi  = (n[1]*p[0] - n[0]*p[1])/n[1]/r # equation of meshing
        s = r*phi

        cp = ca.cos(phi)
        sp = ca.sin(phi)

        Gfg = sc.TrotZ(phi)
        Gfr = sc.TtY(r) @ sc.TtX(-s)
        Ggr = sc.rigidInverse(Gfg) @ Gfr
        Rgr = Ggr[0:2, 0:2]
        dgr = Ggr[0:2, 3]
        p_gear = Rgr @ p[0:2] + dgr
        n_gear = Rgr @ n[0:2]

        self.curves_point_fun = []; self.curves_normal_fun = []; curves_names = ['root_circ', 'fillet', 'flank', 'tip_fillet']
        for curve_point_expr, curve_normal_expr, curve_name in zip(p_curves, n_curves, curves_names):
            self.curves_point_fun.append(ca.Function(f'{curve_name}_point', [u], [Rgr @ curve_point_expr[0:2] + dgr]))
            self.curves_normal_fun.append(ca.Function(f'{curve_name}_normal', [u], [Rgr @ curve_normal_expr[0:2]]))

        self.point_fun = ca.Function('point_gear', [u], [p_gear])
        self.normal_fun = ca.Function('normal_gear', [u], [n_gear])

        # Setup gear curve geometry and check for undercuts
        self._setup_gear_geometry()

        self.crowning = None # placeholder for crowining 
        return
    
    def _setup_gear_geometry(self):
        """Setup complete gear geometry: calculate curve ends, initialize intervals, and fix undercuts"""
        
        # 1. Calculate gear-specific curve ends (corrected for actual gear outer radius)
        self.curves_ends = list(self.rackCutter.curves_ends)
        
        # Find actual gear end where profile intersects outer radius
        u_min = float(self.curves_ends[2])  # Start from flank section
        u_max = float(self.curves_ends[-1]) # Original rack end
        
        # For internal gears, check if outer radius is physically achievable
        if self.internal_gear:
            if self.data.outer_radius < self.data.base_radius:
                # Outer radius is below base radius - use base radius instead
                effective_outer_radius = self.data.base_radius
                print(f"Warning: Internal gear outer radius ({self.data.outer_radius:.3f}) is below base radius ({self.data.base_radius:.3f}). Using base radius.")
            else:
                effective_outer_radius = self.data.outer_radius
                
            # Extend search range moderately for internal gears
            u_max = u_max * 1.2  # Extend search range by 20%
        else:
            effective_outer_radius = self.data.outer_radius
            
        gear_end_u = find_gear_end_parameter(
            self.point_fun, 
            self.data.outer_radius, 
            u_min, 
            u_max,
            self.internal_gear
        )
        
        self.curves_ends[-1] = gear_end_u
        # 2. Initialize curve intervals for each section based on curve_ends
        # curves_intervals: [(start, end), (start, end), (start, end), (start, end)]
        # for [topland, fillet, flank, tip_fillet] sections
        self.curves_intervals = [
            (0, self.curves_ends[0]),                           # topland: 0 to u1
            (self.curves_ends[0], self.curves_ends[1]),         # fillet: u1 to u2  
            (self.curves_ends[1], self.curves_ends[2]),         # flank: u2 to u3
            (self.curves_ends[2], self.curves_ends[3])          # tip_fillet: u3 to u4
        ]
        
        # 3. Check for undercut between fillet and flank curves and fix intervals if needed
        u1, u2, u3 = float(self.curves_ends[0]), float(self.curves_ends[1]), float(self.curves_ends[2])
        
        # Find intersection between fillet and flank curves
        u_fillet_intersect, u_flank_intersect = find_curve_intersection(
            self.curves_point_fun[1], self.curves_point_fun[2], u1, u2, u3
        )
        
        if u_fillet_intersect is not None:
            # Update intervals to avoid the undercut loop
            self.curves_intervals[1] = (self.curves_intervals[1][0], u_fillet_intersect)  # fillet: u1 to intersection
            self.curves_intervals[2] = (u_flank_intersect, self.curves_intervals[2][1])   # flank: intersection to u3
        
        # update form diameter radius
        self.data.form_diameter_radius = np.linalg.norm(self.point_fun(self.curves_ends[1]).full())
        
        # 4. Pre-compute sampled points and normals for efficient plotting
        self._sample_gear_profile()
    
    def _sample_gear_profile(self, num_points=1000):
        """Pre-compute sampled points and normals for efficient plotting"""
        
        # Use gear-specific curve intervals (corrected for undercut if needed)
        intervals = self.curves_intervals
        
        # Distribute points proportionally across each section
        section_points = [
            max(5, int(num_points * 0.15)),   # topland: 15% of points
            max(10, int(num_points * 0.25)),  # fillet: 25% of points  
            max(12, int(num_points * 0.30)),  # flank: 30% of points
            max(12, int(num_points * 0.30))   # tip_fillet: 30% of points
        ]
        
        # Generate u_vec with non-linear spacing using curve intervals
        u_sections = []
        
        for i, (section_start, section_end) in enumerate(intervals):
            section_len = section_end - section_start
            if section_len > 1e-6:  # Only if meaningful length
                if i == 0:  # Topland - linear spacing
                    u_section = np.linspace(section_start, section_end, section_points[i], endpoint=False)
                elif i == 3:  # Tip fillet - include endpoint
                    u_section = nonlinspace(section_start, section_end, section_points[i], 
                                          power=0.7, endpoint=True)
                else:  # Fillet and flank sections
                    power = 0.6 if i == 1 else 0.5  # fillet: 0.6, flank: 0.5
                    u_section = nonlinspace(section_start, section_end, section_points[i], 
                                          power=power, endpoint=False)
                
                u_sections.append(u_section)
        
        # Combine all sections
        u_total = intervals[-1][1]  # End of last interval
        u_vec = np.concatenate(u_sections) if u_sections else np.linspace(0, float(u_total), num_points)

        # Sample points and normals
        try:
            points_spacing = self.point_fun(u_vec.reshape(1,-1)).full()
            normals_spacing = self.normal_fun(u_vec.reshape(1,-1)).full()
            
            # Store sampled data in dataclass
            self.sampled_profile = sampledGearProfile(
                u_vec=u_vec,
                x=points_spacing[0,:],
                y=points_spacing[1,:],
                nx=normals_spacing[0,:],
                ny=normals_spacing[1,:],
                num_points=len(u_vec)
            )
            
        except Exception as e:
            # Fallback: store empty data
            self.sampled_profile = sampledGearProfile()
    
    def apply_crowning(self, Ep_tip, Ep_root, Ef_toe, Ef_heel, b, order_prof, order_face):
        
        ra = self.data.outer_radius
        rff = self.data.form_diameter_radius
        rf = self.data.root_radius
        
        E_fun = crowning_RZE(ra, rff, rf, b, Ep_tip, Ep_root, Ef_toe, Ef_heel, order_prof, order_face)
        self.crowning = E_fun
        return
    
    def plot_tooth(self, plot_normals=False, debug_colors=False):
        """
        Plot one complete gear tooth using pre-computed sampled points
        """
        
        # Use pre-computed sampled points
        if not hasattr(self, 'sampled_profile') or self.sampled_profile.is_empty():
            print("Warning: No sampled points available for plotting")
            return
            
        x_spacing = self.sampled_profile.x
        y_spacing = self.sampled_profile.y
        u_vec = self.sampled_profile.u_vec
        
        if plot_normals:
            nx_spacing = self.sampled_profile.nx
            ny_spacing = self.sampled_profile.ny
        
        # Mirror the spacing flank to get the other flank
        # Convert to polar coordinates for mirroring
        r_spacing = np.sqrt(x_spacing**2 + y_spacing**2)
        theta_spacing = np.arctan2(y_spacing, x_spacing)
        
        # Mirror angles around y-axis (negate x-coordinates)
        theta_spacing_mirror = np.pi - theta_spacing
        x_spacing_mirror = r_spacing * np.cos(theta_spacing_mirror)
        y_spacing_mirror = r_spacing * np.sin(theta_spacing_mirror)
        
        # No rotation - show tooth spacing directly
        # Right flank (original spacing)
        x_right = x_spacing
        y_right = y_spacing
        
        # Left flank (mirrored spacing)
        x_left = x_spacing_mirror
        y_left = y_spacing_mirror
        
        # Plot the tooth
        plt.figure(figsize=(10, 10))
        
        if debug_colors:
            # Plot different sections in different colors for debugging
            # Get curve intervals for color coding
            intervals = self.curves_intervals
            section_names = ['Topland', 'Fillet', 'Flank', 'Tip_fillet']
            section_colors = ['red', 'orange', 'blue', 'green']
            
            print(f"Curve intervals: {intervals}")
            print(f"Curve ends: {self.curves_ends}")
            
            # Plot right flank with color coding
            for i, ((u_start, u_end), name, color) in enumerate(zip(intervals, section_names, section_colors)):
                # Find indices in u_vec that correspond to this interval
                mask = (u_vec >= u_start) & (u_vec <= u_end)
                if np.any(mask):
                    x_section = x_right[mask]
                    y_section = y_right[mask]
                    plt.plot(x_section, y_section, color=color, linewidth=3, 
                            label=f'Right {name} (u:{u_start:.3f}-{u_end:.3f})', marker='o', markersize=3)
            
            # Plot left flank with lighter colors
            for i, ((u_start, u_end), name, color) in enumerate(zip(intervals, section_names, section_colors)):
                mask = (u_vec >= u_start) & (u_vec <= u_end)
                if np.any(mask):
                    x_section = x_left[mask]
                    y_section = y_left[mask]
                    plt.plot(x_section, y_section, color=color, linewidth=2, alpha=0.7,
                            label=f'Left {name}', linestyle='--', marker='s', markersize=2)
        else:
            # Original single-color plotting
            plt.plot(x_right, y_right, 'b-', linewidth=2, label='Right flank')
            plt.plot(x_left, y_left, 'r-', linewidth=2, label='Left flank')
        
        # Add reference circles
        circle_pitch = plt.Circle((0, 0), self.data.pitch_radius, 
                                fill=False, linestyle='--', color='gray', alpha=0.7, label='Pitch circle')
        circle_outer = plt.Circle((0, 0), self.data.outer_radius, 
                                fill=False, linestyle=':', color='green', alpha=0.7, label='Outer circle')
        circle_root = plt.Circle((0, 0), self.data.root_radius, 
                               fill=False, linestyle=':', color='orange', alpha=0.7, label='Root circle')
        circle_base = plt.Circle((0, 0), self.data.base_radius, 
                               fill=False, linestyle='-.', color='purple', alpha=0.7, label='Base circle')
        
        ax = plt.gca()
        ax.add_patch(circle_pitch)
        ax.add_patch(circle_outer)
        ax.add_patch(circle_root)
        ax.add_patch(circle_base)
        
        # Plot normals if requested
        if plot_normals:
            # Right flank normals (original spacing)
            nx_right = nx_spacing
            ny_right = ny_spacing
            
            # Left flank normals (mirrored spacing)
            nx_left = -nx_spacing  # Mirror x-component
            ny_left = ny_spacing   # Keep y-component same
            
            # Show every 5th normal for cleaner visualization
            step = 5
            # Scale arrows to be 0.25*module in length
            arrow_length = 0.25 * self.rack_data.m
            plt.quiver(x_right[::step], y_right[::step], 
                      nx_right[::step], ny_right[::step], 
                      color='blue', alpha=0.7, scale_units='xy', scale=1/arrow_length, 
                      width=0.002, headwidth=3, headlength=4)
            plt.quiver(x_left[::step], y_left[::step], 
                      nx_left[::step], ny_left[::step], 
                      color='red', alpha=0.7, scale_units='xy', scale=1/arrow_length,
                      width=0.002, headwidth=3, headlength=4)
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Check if undercut correction was applied by comparing intervals with original curve ends
        original_ends = self.curves_ends  # These remain the original boundaries
        original_intervals = [(0, original_ends[0]), (original_ends[0], original_ends[1]), 
                            (original_ends[1], original_ends[2]), (original_ends[2], original_ends[3])]
        undercut_corrected = any(abs(self.curves_intervals[i][0] - original_intervals[i][0]) > 1e-6 or 
                               abs(self.curves_intervals[i][1] - original_intervals[i][1]) > 1e-6 
                               for i in range(len(self.curves_intervals)))
        
        title = f'Gear Tooth Spacing (Z={self.data.nZ}, m={self.rack_data.m}mm)'
        if undercut_corrected:
            title += ' [Undercut Corrected]'
        plt.title(title)
        plt.xlabel('X [mm]')
        plt.ylabel('Y [mm]')
        
        # Set appropriate axis limits
        max_radius = max(self.data.outer_radius, abs(self.data.root_radius))
        margin = max_radius * 0.1
        plt.xlim(-max_radius - margin, max_radius + margin)
        plt.ylim(-margin, max_radius + margin)
        
        plt.show(block=True)

    def set_lengthwise_curve(self, lengthwise_curve_function = 25, lengthwise_curve_ranges = (-10, 10)):
        """
        lengthwise_curve_function: callable, takes (z) and returns lengthwise curve offset at that z
        lengthwise_curve_ranges: list of tuples (z_min, z_max) defining the ranges where the lengthwise curve function is applied
        """

        # check if lengthwise_curve_function is callable
        if not callable(lengthwise_curve_function) and (isinstance(lengthwise_curve_function, float) or isinstance(lengthwise_curve_function, int)):
            # then we assume it is the helix angle for a helical gear
            helix_angle = lengthwise_curve_function* np.pi/180
            lengthwise_curve_function = lambda z: z  / np.tan(np.pi/2-helix_angle) / self.data.pitch_radius

        z = ca.SX.sym('z', 1, 1); u = ca.SX.sym('u', 1, 1)
        psi_sym = lengthwise_curve_function(z)
        psi_der = ca.jacobian(psi_sym, z)

        R = sc.rotZ2D(psi_sym)
        pR = self.point_fun(u); nR = self.normal_fun(u) # right flank profile
        pL = pR * ca.vertcat(-1, 1); nL = nR* ca.vertcat(-1, 1) # left flank profile

        rotated_point_R = R @ pR; rotated_normal_R = R @ nR
        rotated_point_L = R @ pL; rotated_normal_L = R @ nL

        self.point_fun_3D_R = ca.Function('point_3D_R', [u, z], [ca.vertcat(rotated_point_R, z)])
        NR = ca.vertcat(rotated_normal_R,
                       -psi_der*(rotated_normal_R[1]*rotated_point_R[0] - rotated_normal_R[0]*rotated_point_R[1]))
        
        nR = NR / ca.norm_2(NR)
        self.normal_fun_3D_R = ca.Function('normal_3D_R', [u, z], [nR])

        self.point_fun_3D_L = ca.Function('point_3D_L', [u, z], [ca.vertcat(rotated_point_L, z)])
        NL = ca.vertcat(rotated_normal_L,
                       -psi_der*(rotated_normal_L[1]*rotated_point_L[0] - rotated_normal_L[0]*rotated_point_L[1]))
        nL = NL / ca.norm_2(NL)
        self.normal_fun_3D_L = ca.Function('normal_3D_L', [u, z], [nL])
        return

    def plot3D(self, z_range=(-10, 10), n_z_points=20, n_u_points=None, show_edges=True, 
               edge_color='black', face_color='lightblue', show_single_tooth=False, 
               auto_close=False, block=True):
        """
        Plot the 3D gear surface using easy_plot functionality
        
        Parameters:
        -----------
        z_range : tuple, optional
            Range of z-coordinates for the gear (z_min, z_max). Default: (-10, 10)
        n_z_points : int, optional
            Number of points along the z-axis. Default: 20
        n_u_points : int, optional
            Number of points along the profile (u parameter). If None, uses sampled profile points
        show_edges : bool, optional
            Whether to show mesh edges. Default: True
        edge_color : str, optional
            Color of the mesh edges. Default: 'black'
        face_color : str, optional
            Color of the mesh faces. Default: 'lightblue'
        show_single_tooth : bool, optional
            If True, shows only one tooth. If False, shows complete gear. Default: False
        auto_close : bool, optional
            Whether to automatically close the plot window. Default: False
        block : bool, optional
            Whether to block execution until plot is closed. Default: True
        """
        
        # Check if 3D functions are available
        if not hasattr(self, 'point_fun_3D_R'):
            raise ValueError("3D gear functions not available. Call set_lengthwise_curve() first.")
        
        # Import easy_plot functionality
        try:
            from easy_plot import Figure, surface
        except ImportError:
            raise ImportError("easy_plot module not available. Cannot create 3D plot.")
        
        # Use sampled profile points if n_u_points not specified
        if n_u_points is None:
            if hasattr(self, 'sampled_profile') and not self.sampled_profile.is_empty():
                u_vec = self.sampled_profile.u_vec
            else:
                # Fallback: create uniform sampling
                u_vec = np.linspace(0, float(self.curves_ends[-1]), 100)
        else:
            u_vec = np.linspace(0, float(self.curves_ends[-1]), n_u_points)
        
        # Create z coordinates
        z_min, z_max = z_range
        z_vec = np.linspace(z_min, z_max, n_z_points)
        
        # Create meshgrid for evaluation
        U, Z = np.meshgrid(u_vec, z_vec)
        
        # Evaluate 3D points for right flank
        points_R = np.zeros((n_z_points, len(u_vec), 3))
        for i, z_val in enumerate(z_vec):
            for j, u_val in enumerate(u_vec):
                try:
                    point_3d = self.point_fun_3D_R(u_val, z_val).full().flatten()
                    points_R[i, j, :] = point_3d
                except:
                    # Handle potential evaluation errors
                    points_R[i, j, :] = [0, 0, z_val]
        
        # Extract X, Y, Z coordinates for right flank
        X_R = points_R[:, :, 0]
        Y_R = points_R[:, :, 1] 
        Z_R = points_R[:, :, 2]
        
        # Evaluate 3D points for left flank
        points_L = np.zeros((n_z_points, len(u_vec), 3))
        for i, z_val in enumerate(z_vec):
            for j, u_val in enumerate(u_vec):
                try:
                    point_3d = self.point_fun_3D_L(u_val, z_val).full().flatten()
                    points_L[i, j, :] = point_3d
                except:
                    # Handle potential evaluation errors  
                    points_L[i, j, :] = [0, 0, z_val]
        
        # Extract X, Y, Z coordinates for left flank
        X_L = points_L[:, :, 0]
        Y_L = points_L[:, :, 1]
        Z_L = points_L[:, :, 2]
        
        # Create figure
        fig = Figure(f'3D Gear Surface (Z={self.data.nZ}, m={self.rack_data.m}mm)')
        
        # Add right flank surface
        surface_R = surface(fig, X_R, Y_R, Z_R, 
                           show_edges=show_edges, 
                           edge_color=edge_color, 
                           face_color=face_color)
        
        # Add left flank surface  
        surface_L = surface(fig, X_L, Y_L, Z_L,
                           show_edges=show_edges,
                           edge_color=edge_color, 
                           face_color=face_color)
        
        if not show_single_tooth:
            # Create complete gear by rotating tooth around z-axis
            tooth_angle = 2 * np.pi / self.data.nZ
            
            for tooth_num in range(1, self.data.nZ):  # Already have tooth 0
                angle = tooth_num * tooth_angle
                
                # Rotation matrix for this tooth
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                
                # Rotate right flank coordinates
                X_R_rot = cos_a * X_R - sin_a * Y_R
                Y_R_rot = sin_a * X_R + cos_a * Y_R
                
                # Rotate left flank coordinates
                X_L_rot = cos_a * X_L - sin_a * Y_L  
                Y_L_rot = sin_a * X_L + cos_a * Y_L
                
                # Add rotated surfaces
                surface(fig, X_R_rot, Y_R_rot, Z_R,
                       show_edges=show_edges,
                       edge_color=edge_color, 
                       face_color=face_color)
                       
                surface(fig, X_L_rot, Y_L_rot, Z_L,
                       show_edges=show_edges,
                       edge_color=edge_color,
                       face_color=face_color)
        
        # Show the 3D plot with proper blocking behavior
        if block:
            # Keep the plot open and wait for user interaction
            fig.show()
            try:
                # For PyVistaQt BackgroundPlotter, keep it interactive
                if hasattr(fig.figure, 'app') and hasattr(fig.figure.app, 'exec_'):
                    fig.figure.app.exec_()
                else:
                    # Fallback: simple input to keep script running
                    input("Press Enter to close the 3D plot...")
            except (KeyboardInterrupt, EOFError):
                print("\nPlot closed.")
        else:
            # Non-blocking show
            fig.show()
        
        return fig

    def fit_nurbs(self, z_range=(-10, 10), n_z_points=30, n_u_points=50, 
                  degree_u=4, degree_v=4, control_points_u=40, control_points_v=25,
                  both_flanks=False, verbose=True, backlash = 0.05):
        """
        Fit 3D gear surface to NURBS using basic least squares method
        
        Parameters:
        -----------
        z_range : tuple
            Range of z-coordinates for the gear (z_min, z_max)
        n_z_points : int
            Number of points along the z-axis for surface sampling
        n_u_points : int
            Number of points along the profile direction
        degree_u, degree_v : int
            NURBS degrees in U (profile) and V (lengthwise) directions
        control_points_u, control_points_v : int, optional
            Number of control points. If None, auto-calculated
        both_flanks : bool, optional
            If True, fit both flanks as a single continuous surface by flipping
            and joining points at root circumference. Default is False.
        verbose : bool, optional
            If True, print progress and diagnostic information. Default is True.
            
        Returns:
        --------
        dict : Dictionary containing fitted NURBS surfaces
        """
        # Import NURBS fitting functionality
        try:
            from nurbs import Nurbs
        except ImportError:
            raise ImportError("NURBS module not available. Cannot perform surface fitting.")
        
        # Check if 3D functions are available
        if not hasattr(self, 'point_fun_3D_R'):
            raise ValueError("3D gear functions not available. Call set_lengthwise_curve() first.")
        
        if verbose:
            print(f"Fitting NURBS surfaces to 3D gear (Z={self.data.nZ})...")
        
        # Use profile parameter range from curves_ends (full tooth profile)
        u_min = 0.0  # Start from beginning of profile
        u_max = float(self.curves_ends[-1])*1.2  # End of tooth profile
        u_vec = np.linspace(u_min, u_max, n_u_points)
        z_vec = np.linspace(z_range[0], z_range[1], n_z_points)
        
        # Auto-calculate control points if not specified
        if control_points_u is None:
            control_points_u = min(n_u_points // 2, 25)
        if control_points_v is None:
            control_points_v = min(n_z_points // 2, 20)
        
        if verbose:
            print(f"Surface resolution: {n_u_points}×{n_z_points} points")
            print(f"NURBS degrees: U={degree_u}, V={degree_v}")
            print(f"Control points: {control_points_u}×{control_points_v}")
        
        # Create meshgrid for vectorized evaluation
        if verbose:
            print("Creating parameter meshgrid for vectorized evaluation...")
        U_grid, Z_grid = np.meshgrid(u_vec, z_vec, indexing='ij')
        
        # Flatten grids for vectorized CasADi evaluation
        u_flat = U_grid.reshape(1, -1)
        z_flat = Z_grid.reshape(1, -1)
        
        # Generate 3D surface points for right flank using vectorization
        if verbose:
            print("Generating right flank surface points (vectorized)...")

        # Use CasADi vectorization - evaluate all points at once
        points_3d_R_flat = self.point_fun_3D_R(u_flat, z_flat).full()
        normals_3d_R_flat = self.normal_fun_3D_R(u_flat, z_flat).full()

        if self.crowning is not None:
            if verbose:
                print("Found crowning function: applying micro-geometry modifications on the right flank...")
            R_flat = np.sqrt(points_3d_R_flat[0,:]**2 + points_3d_R_flat[1,:]**2)
            E_flat = self.crowning(z_flat, R_flat)
            points_3d_R_flat -= E_flat/1000*normals_3d_R_flat
        
        # Extract x, y, z components and reshape to grid
        points_R = np.zeros((3, n_u_points, n_z_points))
        points_R[0, :, :] = points_3d_R_flat[0, :].reshape(n_u_points, n_z_points)  # X
        points_R[1, :, :] = points_3d_R_flat[1, :].reshape(n_u_points, n_z_points)  # Y  
        points_R[2, :, :] = points_3d_R_flat[2, :].reshape(n_u_points, n_z_points)  # Z
        
        # Generate 3D surface points for left flank using vectorization
        if verbose:
            print("Generating left flank surface points (vectorized)...")
        # Use CasADi vectorization - evaluate all points at once
        points_3d_L_flat = self.point_fun_3D_L(u_flat, z_flat).full()
        normals_3d_L_flat = self.normal_fun_3D_L(u_flat, z_flat).full()
        
        if self.crowning is not None:
            if verbose:
                print("Found crowning function: applying micro-geometry modifications on the right flank...")
            R_flat = np.sqrt(points_3d_L_flat[0,:]**2 + points_3d_L_flat[1,:]**2)
            E_flat = self.crowning(z_flat, R_flat)
            points_3d_L_flat -= E_flat/1000*normals_3d_L_flat

        if verbose:
            print("Applying backlash on the left flank")
        points_3d_L_flat = sc.rotZ(backlash/self.data.pitch_radius)@points_3d_L_flat
        
        # Extract x, y, z components and reshape to grid
        points_L = np.zeros((3, n_u_points, n_z_points))
        points_L[0, :, :] = points_3d_L_flat[0, :].reshape(n_u_points, n_z_points)  # X
        points_L[1, :, :] = points_3d_L_flat[1, :].reshape(n_u_points, n_z_points)  # Y
        points_L[2, :, :] = points_3d_L_flat[2, :].reshape(n_u_points, n_z_points)  # Z

        if both_flanks:
            # Combine both flanks into a single continuous surface
            if verbose:
                print("Combining flanks into single continuous surface...")
            
            # Flip left flank points in U direction to create continuity
            points_L_flipped = points_L[:, ::-1, :]
            
            # Combine points: right flank + flipped left flank
            # This creates a continuous surface from right flank through root to left flank
            combined_points = np.concatenate([points_L_flipped, points_R], axis=1)
            # we need now to resample the points uniformly
            for ii in range(n_z_points):
                x, y, z = interp_arc(n_u_points*2, combined_points[0,:,ii], combined_points[1,:,ii], combined_points[2,:,ii])
                combined_points[0,:,ii] = x
                combined_points[1,:,ii] = y
                combined_points[2,:,ii] = z

            import easy_plot as ep
            X = combined_points[0,:,:].squeeze(); Y = combined_points[1,:,:].squeeze(); Z = combined_points[2,:,:].squeeze()
            F = ep.Figure()
            S = ep.scatter(F, X.reshape(-1,1),Y.reshape(-1,1),Z.reshape(-1,1))
            F.show()

            if verbose:
                print(f"Combined surface dimensions: {combined_points.shape[1]}×{combined_points.shape[2]} points")
            
            # Adjust control points for wider U dimension
            combined_control_points_u = min(combined_points.shape[1] // 2, control_points_u * 2)
            
            # Fit single NURBS surface to combined points
            if verbose:
                print("Fitting NURBS to combined tooth surface...")
            nurbs_combined = Nurbs()
            nurbs_combined.fit(combined_points, degree_u, degree_v, (combined_control_points_u, control_points_v))
            
            # Calculate fitting quality metrics
            rms_error = np.sqrt(np.mean(nurbs_combined.fit_residuals**2)) if hasattr(nurbs_combined, 'fit_residuals') else 0
            max_error = np.max(np.abs(nurbs_combined.fit_residuals)) if hasattr(nurbs_combined, 'fit_residuals') else 0
            
            if verbose:
                print(f"Combined surface - RMS error: {rms_error:.6f}, Max error: {max_error:.6f}")
            
            # Store NURBS surface in gear object
            self.nurbs_surfaces = nurbs_combined
            
        else:
            # Fit separate surfaces for each flank (original behavior)
            if verbose:
                print("Fitting NURBS to right flank...")
            nurbs_R = Nurbs()
            nurbs_R.fit(points_R, degree_u, degree_v, (control_points_u, control_points_v))
            
            if verbose:
                print("Fitting NURBS to left flank...")
            nurbs_L = Nurbs()
            nurbs_L.fit(points_L, degree_u, degree_v, (control_points_u, control_points_v))
            
            # Compute basic fitting errors
            if verbose:
                print("Computing fitting residuals...")
            
            # Simple RMS error calculation
            rms_error_R = np.sqrt(np.mean(nurbs_R.fit_residuals**2)) if hasattr(nurbs_R, 'fit_residuals') else 0
            max_error_R = np.max(np.abs(nurbs_R.fit_residuals)) if hasattr(nurbs_R, 'fit_residuals') else 0
            
            rms_error_L = np.sqrt(np.mean(nurbs_L.fit_residuals**2)) if hasattr(nurbs_L, 'fit_residuals') else 0
            max_error_L = np.max(np.abs(nurbs_L.fit_residuals)) if hasattr(nurbs_L, 'fit_residuals') else 0
            
            if verbose:
                print(f"Right flank - RMS error: {rms_error_R:.6f}, Max error: {max_error_R:.6f}")
                print(f"Left flank  - RMS error: {rms_error_L:.6f}, Max error: {max_error_L:.6f}")
            
            # Store NURBS surfaces in gear object
            self.nurbs_surfaces = {
                'right_flank': nurbs_R,
                'left_flank': nurbs_L
                }
            
        return

    def generate_gear_CAD(self, z_range=(-10, 10), rim_thickness=None, filename='test_gear.STEP'):

        from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax1, gp_Vec
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol,BRepPrimAPI_MakePrism
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeThickSolid
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
        from OCC.Core.TopTools import TopTools_ListOfShape
        from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir

        outer_radius = self.data.outer_radius
        root_radius = self.data.root_radius
        internal = self.internal_gear

        sign_rim = -1
        if internal:
            sign_rim = 1

        if rim_thickness is None:
            inner_radius = 0
        else:
            inner_radius = root_radius + sign_rim*rim_thickness

        points = [
            gp_Pnt(inner_radius, 0, z_range[0]),
            gp_Pnt(outer_radius, 0, z_range[0]),
            gp_Pnt(outer_radius, 0, z_range[1]),
            gp_Pnt(inner_radius, 0, z_range[1]),
            gp_Pnt(inner_radius, 0, z_range[0])
        ]
        edges = [BRepBuilderAPI_MakeEdge(points[i], points[i + 1]).Edge() for i in range(len(points) - 1)]

        wire_builder = BRepBuilderAPI_MakeWire()
        for edge in edges:
            wire_builder.Add(edge)
        profile_wire = wire_builder.Wire()
        face = BRepBuilderAPI_MakeFace(profile_wire).Face()

        revolution_axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))

        revolved_shape = BRepPrimAPI_MakeRevol(face, revolution_axis).Shape()

        from nurbs import Nurbs_to_OCC_surface

        nurbs_OCC = Nurbs_to_OCC_surface(self.nurbs_surfaces)

        u_min, u_max, v_min, v_max = nurbs_OCC.Bounds()

        # Extract edges
        iso_u0 = nurbs_OCC.UIso(u_min)   # curve at u = u_min (one boundary)
        iso_umax = nurbs_OCC.UIso(u_max) # curve at u = u_max
        iso_v0 = nurbs_OCC.VIso(v_min)   # curve at v = v_min
        iso_vmax = nurbs_OCC.VIso(v_max) # curve at v = v_max
        edge_u0 = BRepBuilderAPI_MakeEdge(iso_u0).Edge()
        edge_umax = BRepBuilderAPI_MakeEdge(iso_umax).Edge()
        edge_v0 = BRepBuilderAPI_MakeEdge(iso_v0).Edge()
        edge_vmax = BRepBuilderAPI_MakeEdge(iso_vmax).Edge()
        tooth_spacing = BRepBuilderAPI_MakeFace(nurbs_OCC, 1e-6).Face()

        pts = []; sampling_values = [[0,0], [1,0], [0,1], [1,1]]
        for values in sampling_values:
            pts.append(nurbs_OCC.Value(values[0], values[1]))

        # tip edges
        edge_tip_toe = BRepBuilderAPI_MakeEdge(pts[0], pts[1]).Edge()
        edge_tip_heel = BRepBuilderAPI_MakeEdge(pts[2], pts[3]).Edge()

        mk_wire = BRepBuilderAPI_MakeWire()
        mk_wire.Add(edge_u0)
        mk_wire.Add(edge_vmax)
        mk_wire.Add(edge_umax)
        mk_wire.Add(edge_v0)
        wire = mk_wire.Wire()


        # make a filler
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
        from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_G1, GeomAbs_G2
        from OCC.Core.TopoDS import topods_Face, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound
        from OCC.Core.BRep import BRep_Builder

        filler_top = BRepOffsetAPI_MakeFilling()
        filler_toe = BRepOffsetAPI_MakeFilling()
        filler_heel = BRepOffsetAPI_MakeFilling()

        edge_list= [edge_tip_toe, edge_umax, edge_tip_heel, edge_u0]
        for edge in edge_list:
            filler_top.Add(edge, GeomAbs_C0, True)

        filler_top.Build()
        face_top = topods_Face(filler_top.Shape())

        edge_list = [edge_vmax, edge_tip_heel]
        for edge in edge_list:
            filler_toe.Add(edge, GeomAbs_C0, True)

        filler_toe.Build()
        face_toe = topods_Face(filler_toe.Shape())

        edge_list = [edge_v0, edge_tip_toe]
        for edge in edge_list:
            filler_heel.Add(edge, GeomAbs_C0, True)

        filler_heel.Build()
        face_heel = topods_Face(filler_heel.Shape())

        faces = [tooth_spacing, face_top, face_toe, face_heel]

        builder = BRep_Builder()
        shell = TopoDS_Shell()
        builder.MakeShell(shell)

        for face in faces:  # assume faces is a list of TopoDS_Face
            builder.Add(shell, face)

        cutter_solid = TopoDS_Solid()
        builder.MakeSolid(cutter_solid)
        builder.Add(cutter_solid, shell)

        # check faces
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.BRepCheck import _BRepCheck, BRepCheck_Analyzer
        analyzer_cutter = BRepCheck_Analyzer(cutter_solid, True)
        analyzer_blank = BRepCheck_Analyzer(cutter_solid, True)
        if not analyzer_cutter.IsValid():
            print("Cutter shape is invalid")

        if not analyzer_blank.IsValid():
            print("Blank shape is invalid")
        else:
            print("Blank shape si OK")

        cutting_function = BRepAlgoAPI_Cut
        if internal:
            cutting_function = BRepAlgoAPI_Common

        # for ii in range(0, self.data.nZ):
        #     angle = ii * 360.0 / self.data.nZ
        #     trsf = gp_Trsf()
        #     trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), angle * np.pi / 180.0)
        #     brep_trsf = BRepBuilderAPI_Transform(cutter_solid, trsf, True).Shape()
        #     revolved_shape = cutting_function(revolved_shape, brep_trsf).Shape()
        #     print(f"Cut {ii+1}/{self.data.nZ}-th tooth")
        


        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.Interface import Interface_Static
        
        step_writer_body = STEPControl_Writer()
        Interface_Static.SetCVal("write.step.schema", "AP203")
        # step_writer_body.Transfer(revolved_shape, STEPControl_AsIs)
        step_writer_body.Transfer(tooth_spacing, STEPControl_AsIs)
        step_writer_body.Write(filename)

        # from OCC.Display.SimpleGui import init_display
        # display, start_display, add_menu, add_function_to_menu = init_display()
        # # display.DisplayShape(cutter_solid)
        # # display.DisplayShape(cut.Shape())
        # # display.DisplayShape(face_top)
        # display.DisplayShape(tooth_spacing)
        # # display.DisplayShape(revolved_shape)
        # start_display()


