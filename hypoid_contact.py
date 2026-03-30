# -*- coding: utf-8 -*-
"""
准双曲面齿轮接触分析
读取齿面数据并进行接触分析
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_gear_surfaces(filename='all_surfaces.npz'):
    """
    读取齿面数据
    
    Returns:
        surfaces: dict, 包含4个齿面的点和法向量
        meta: dict, 元数据 (n_profile, n_face) 和装配参数
    """
    data = np.load(filename, allow_pickle=True)
    
    surfaces = {}
    for member in ['gear', 'pinion']:
        for flank in ['concave', 'convex']:
            key = f"{member}_{flank}"
            surfaces[key] = {
                'points': data[f"{key}_pts"],
                'normals': data[f"{key}_nrm"],
            }
    
    meta = {
        'n_profile': int(data['n_profile']),
        'n_face': int(data['n_face']),
    }
    
    
    # 尝试读取装配参数
    try:
        meta['shaft_angle'] = float(data['shaft_angle'])
        meta['hypoid_offset'] = float(data['hypoid_offset'])
        meta['hand'] = str(data['hand'])
        if 'EPGalpha' in data:
           meta['EPGalpha'] = data['EPGalpha']
        if 'N_gear' in data:
            meta['N_gear'] = int(data['N_gear'])
            meta['N_pinion'] = int(data['N_pinion'])
        print(f"  装配参数: 轴交角={meta['shaft_angle']}°, 偏置={meta['hypoid_offset']}mm")
    except:
        print("  (无装配参数)")
    
    print(f"已加载齿面数据:")
    for key in surfaces:
        print(f"  {key}: {surfaces[key]['points'].shape}")
    
    return surfaces, meta


def get_pinion_transform(meta):
    """
    计算小齿轮到大齿轮坐标系的变换矩阵
    基于 gear_to_pinion_kinematics 函数
    
    EPGalpha: [E, P, G, alpha]
    - E: 偏置调整
    - P: 小齿轮轴向位置
    - G: 大齿轮轴向位置  
    - alpha: 轴交角调整
    """
    if 'shaft_angle' not in meta:
        return np.eye(4)
    
    SIGMA = np.deg2rad(meta['shaft_angle'])
    offset = meta['hypoid_offset']
    hand = meta['hand']
    
    # 从 EPGalpha 获取位置参数
    epga = meta.get('EPGalpha', [0, 0, 0, 0])
    E = epga[0] if len(epga) > 0 else 0  # 偏置调整
    P = epga[1] if len(epga) > 1 else 0  # 小齿轮轴向
    G = epga[2] if len(epga) > 2 else 0  # 大齿轮轴向
    alpha = epga[3] if len(epga) > 3 else 0  # 轴交角调整
    
    s = 1 if hand.lower() == 'left' else -1
    
    # 变换矩阵：小齿轮 -> 大齿轮坐标系
    # Tpg = TtZ(-P) @ TtY((offset+E)*s) @ TrotY(SIGMA+alpha) @ TtZ(G) @ TrotZ(pi)
    
    # 构建变换矩阵
    total_sigma = SIGMA + alpha
    cos_s, sin_s = np.cos(total_sigma), np.sin(total_sigma)
    total_offset = (offset + E) * s
    
    # 组合变换
    T = np.array([
        [-cos_s, 0,  sin_s, G*sin_s],
        [0,      -1, 0,     total_offset],
        [sin_s,  0,  cos_s, -P + G*cos_s],
        [0,      0,  0,     1]
    ])
    
    return T


def transform_points(pts_3d, T):
    """
    应用齐次变换到点云
    pts_3d: (3, m, n) 形状的点
    T: 4x4 齐次变换矩阵
    """
    shape = pts_3d.shape
    pts_flat = pts_3d.reshape(3, -1)
    pts_homo = np.vstack([pts_flat, np.ones((1, pts_flat.shape[1]))])
    pts_transformed = T @ pts_homo
    return pts_transformed[:3, :].reshape(shape)


def plot_surfaces(surfaces, title="齿面可视化"):
    """
    可视化齿面（曲面显示）
    """
    fig = plt.figure(figsize=(14, 10))
    
    # 颜色映射
    cmaps = {
        'gear_concave': 'Blues',
        'gear_convex': 'Greens',
        'pinion_concave': 'Reds',
        'pinion_convex': 'Oranges'
    }
    
    # 2x2 子图布局
    for idx, (key, surf) in enumerate(surfaces.items()):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        pts = surf['points']
        
        # 数据形状: (3, n_face, n_profile) - 直接使用不转置
        X = pts[0, :, :]
        Y = pts[1, :, :]
        Z = pts[2, :, :]
        
        ax.plot_surface(X, Y, Z, cmap=cmaps[key], alpha=0.8, 
                       linewidth=0.3, edgecolor='gray', antialiased=True)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(key)
        set_axes_equal(ax)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return fig


def rotate_z(pts, angle):
    """绕Z轴旋转点云"""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    shape = pts.shape
    pts_flat = pts.reshape(3, -1)
    pts_rot = R @ pts_flat
    return pts_rot.reshape(shape)


def calculate_xyz_grid(pts_flat, original_shape):
    """辅助函数：将展平的点云恢复为网格用于 plot_surface"""
    return pts_flat.reshape(original_shape)


def find_best_meshing_pair_indices(pts_g, pts_p, N_gear, N_pinion, T_pin_to_gear, hand='right'):
    """
    寻找最佳啮合的齿索引对 (g_idx, p_idx)
    策略（用户指定）：
    1. 遍历所有小轮位置，找到齿面中点 Z 坐标最大的那个作为 p_idx（啮合齿面）
    2. 遍历所有大轮位置，找到离该小轮齿面最近的那个作为 g_idx
    """
    # 1. 基础中心点
    mid_face = pts_g.shape[1] // 2
    mid_prof = pts_g.shape[2] // 2
    center_g_base = pts_g[:, mid_face, mid_prof]
    center_p_base = pts_p[:, mid_face, mid_prof]
    
    s_gear = -1 if hand.lower() == 'right' else 1
    s_pinion = -s_gear
    
    pitch_g = 2 * np.pi / N_gear
    pitch_p = 2 * np.pi / N_pinion
    
    # ---------------------------------------------------------
    # 第一步：找到小齿轮啮合面 (Z 轴最大)
    # ---------------------------------------------------------
    best_p = 0
    max_z_p = -float('inf')
    center_p_best = None # 记录最佳小轮在空间的位置
    
    for k in range(N_pinion):
        angle = pitch_p * k * s_pinion
        c, s = np.cos(angle), np.sin(angle)
        R_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        center_rot = R_mat @ center_p_base
        
        # 变换到大齿轮坐标系 (绝对空间)
        center_homo = np.append(center_rot, 1)
        center_trans = (T_pin_to_gear @ center_homo)[:3]
        
        # 判断 Z 值
        z_val = center_trans[2]
        if z_val > max_z_p:
            max_z_p = z_val
            best_p = k
            center_p_best = center_trans
            
    # ---------------------------------------------------------
    # 第二步：找到大齿轮配对 (离最佳小轮最近)
    # ---------------------------------------------------------
    best_g = 0
    min_dist_g = float('inf')
    
    for i in range(N_gear):
        angle = pitch_g * i * s_gear
        c, s = np.cos(angle), np.sin(angle)
        R_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        center_rot = R_mat @ center_g_base
        
        # 计算与最佳小轮中心的距离
        dist = np.linalg.norm(center_rot - center_p_best)
        
        if dist < min_dist_g:
            min_dist_g = dist
            best_g = i
            
    return best_g, best_p


def plot_meshing_pair(surfaces, meta, gear_flank='concave', pinion_flank='convex',
                      n_gear_teeth=None, n_pinion_teeth=None):
    """
    绘制多齿啮合对（应用装配变换）
    
    参数:
        n_gear_teeth: 大齿轮显示齿数 (None=全部)
        n_pinion_teeth: 小齿轮显示齿数 (None=全部)
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 齿数
    N_gear = meta.get('N_gear', 41)
    N_pinion = meta.get('N_pinion', 9)
    
    # 如果未指定，则显示全部齿
    if n_gear_teeth is None:
        n_gear_teeth = N_gear
    if n_pinion_teeth is None:
        n_pinion_teeth = N_pinion
    
    # 旋向
    hand = meta.get('hand', 'right')
    s_gear = -1 if hand.lower() == 'right' else 1
    s_pinion = -s_gear
    
    # 大齿轮（多齿）
    # 获取基础齿面
    pts_g = surfaces[f'gear_{gear_flank}']['points']
    pts_p = surfaces[f'pinion_{pinion_flank}']['points']
    T = get_pinion_transform(meta)
    
    # 寻找最佳匹配对 (g_idx, p_idx)
    g_best, p_best = find_best_meshing_pair_indices(pts_g, pts_p, N_gear, N_pinion, T, hand)
    print(f"最佳啮合对 (Z-max策略): Gear={g_best}, Pinion={p_best}")
    
    # 确定显示范围：只显示最佳匹配及其左右邻居 (共3个)
    # 如果用户没有强制指定数量，默认为 3
    if n_gear_teeth is None:
        num_g = 3
    else:
        num_g = n_gear_teeth
        
    start_g = g_best - num_g // 2
    g_range = range(start_g, start_g + num_g)
    
    if n_pinion_teeth is None:
        num_p = 3
    else:
        num_p = n_pinion_teeth
        
    start_p = p_best - num_p // 2
    p_range = range(start_p, start_p + num_p)
    
    # 大齿轮绘图
    for i in g_range:
        angle = 2 * np.pi / N_gear * i * s_gear
        pts_rot = rotate_z(pts_g, angle)
        
        # 判断是否为最佳匹配齿 (考虑周期性)
        is_best = (i % N_gear == g_best % N_gear)
        
        # 样式：最佳匹配齿不透明且有黑边，其他半透明
        alpha = 0.9 if is_best else 0.4
        edge = 'black' if is_best else 'gray'
        lw = 0.5 if is_best else 0.2
        
        ax.plot_surface(pts_rot[0], pts_rot[1], pts_rot[2], 
                        cmap='Blues', alpha=alpha, linewidth=lw, edgecolor=edge)
    
    # 小齿轮绘图
    for i in p_range:
        angle = 2 * np.pi / N_pinion * i * s_pinion
        # 先旋转再变换
        pts_rot = rotate_z(pts_p, angle)
        pts_transformed = transform_points(pts_rot, T)
        
        # 判断是否为最佳匹配齿
        is_best = (i % N_pinion == p_best % N_pinion)
        
        alpha = 0.9 if is_best else 0.4
        edge = 'black' if is_best else 'gray'
        lw = 0.5 if is_best else 0.2
        
        ax.plot_surface(pts_transformed[0], pts_transformed[1], pts_transformed[2], 
                        cmap='Reds', alpha=alpha, linewidth=lw, edgecolor=edge)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Meshing: Gear({n_gear_teeth}T) - Pinion({n_pinion_teeth}T)')
    
    # 设置各轴比例一致
    set_axes_equal(ax)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def set_axes_equal(ax):
    """
    设置 3D 绘图轴比例一致
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_single_surface(surfaces, member='gear', flank='concave'):
    """
    绘制单个齿面
    """
    key = f"{member}_{flank}"
    pts = surfaces[key]['points']
    nrm = surfaces[key]['normals']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面
    X = pts[0, :, :]
    Y = pts[1, :, :]
    Z = pts[2, :, :]
    
    ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', 
                   linewidth=0.3, edgecolor='gray', antialiased=True)
    
    # 绘制法向量（每隔几个点画一个）
    step = 2
    x = X[::step, ::step].flatten()
    y = Y[::step, ::step].flatten()
    z = Z[::step, ::step].flatten()
    u = nrm[0, ::step, ::step].flatten()
    v = nrm[1, ::step, ::step].flatten()
    w = nrm[2, ::step, ::step].flatten()
    
    ax.quiver(x, y, z, u, v, w, length=0.5, color='red', alpha=0.5)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'{member} {flank}')
    
    set_axes_equal(ax)
    
    plt.tight_layout()
    plt.show()
    
    return fig


# ============================================
# 主程序
# ============================================
if __name__ == '__main__':
    # 读取齿面数据 (优先尝试优化后的数据，如果不存在则使用旧数据)
    try:
        surfaces, meta = load_gear_surfaces('optimized_surfaces.npz')
        print("成功加载优化后的齿面数据: optimized_surfaces.npz")
    except FileNotFoundError:
        print("未找到优化后的数据，加载初始数据: all_surfaces.npz")
        surfaces, meta = load_gear_surfaces('all_surfaces.npz')
    
    print(f"\n网格尺寸: {meta['n_profile']} x {meta['n_face']}")
    
    # 可视化所有齿面（2x2 布局，各自坐标系）
    # 可视化所有齿面（2x2 布局，各自坐标系）
    # plot_surfaces(surfaces)
    
    # 可视化 Drive 侧啮合对 (gear concave - pinion convex)
    # plot_meshing_pair(surfaces, meta, gear_flank='concave', pinion_flank='convex', n_gear_teeth=3, n_pinion_teeth=3)




    # 可视化 Coast 侧啮合对 (gear convex - pinion concave)
    # plot_meshing_pair(surfaces, meta, gear_flank='convex', pinion_flank='concave', n_gear_teeth=3, n_pinion_teeth=3)

    # -------------------------------------------------------------
    # 静态接触分析 (Static Contact Analysis)
    # -------------------------------------------------------------
    try:
        from scipy.optimize import brentq
        import contact_physics
        from nurbs_surface import refine_surface_mesh  # 从独立模块导入

        def run_static_analysis(surfaces, meta, gear_flank, pinion_flank, threshold=0.025, gear_offset_deg=0.0):
            print(f"\n[静态接触分析 (有符号距离): {gear_flank} vs {pinion_flank}] 开始...")
            
            # 1. 准备数据 (Initial Coarse Data)
            pts_g_coarse = surfaces[f'gear_{gear_flank}']['points']
            nrm_g_coarse = surfaces[f'gear_{gear_flank}']['normals']
            pts_p_coarse = surfaces[f'pinion_{pinion_flank}']['points']
            nrm_p_coarse = surfaces[f'pinion_{pinion_flank}']['normals']
            
            T = get_pinion_transform(meta)
            hand = meta.get('hand', 'right')
            N_gear = meta.get('N_gear', 75)
            N_pinion = meta.get('N_pinion', 5)
            
            # 2. 找到最佳初始对齐位置 (使用粗网格快速搜索)
            g_best, p_best = find_best_meshing_pair_indices(pts_g_coarse, pts_p_coarse, N_gear, N_pinion, T, hand)
            
            # =========================================================================
            # Phase 1: Coarse Search (使用原始粗网格)
            # =========================================================================
            print(f"  - [Phase 1 Setup] 准备粗网格数据 ({pts_g_coarse.shape[1]}x{pts_g_coarse.shape[2]})...")
            
            # 3a. 准备粗网格大轮源点
            pitch_g = 2 * np.pi / N_gear
            s_gear = -1 if hand.lower() == 'right' else 1
            gear_offset_rad = np.deg2rad(gear_offset_deg)
            angle_g = pitch_g * g_best * s_gear + gear_offset_rad
            print(f"  - [用户偏移] 大轮附加转角: {gear_offset_deg:.2f} 度")
            
            pts_g_aligned_coarse = rotate_z(pts_g_coarse, angle_g)
            nrm_g_aligned_coarse = rotate_z(nrm_g_coarse, angle_g)
            pts_g_flat_coarse = pts_g_aligned_coarse.reshape(3, -1)
            nrm_g_flat_coarse = nrm_g_aligned_coarse.reshape(3, -1)
            
            # 4a. 定义粗网格小轮变换
            pitch_p = 2 * np.pi / N_pinion
            s_pinion = -s_gear
            # 小轮也要按传动比旋转 (Kinematic Coupling)
            # 传动比: i = N_gear / N_pinion
            # 小轮转角增量 = 大轮转角增量 * i * (s_pinion/s_gear)
            ratio = N_gear / N_pinion
            pinion_offset_rad = gear_offset_rad * ratio * (s_pinion / s_gear)
            base_angle_p = pitch_p * p_best * s_pinion + pinion_offset_rad
            print(f"  - [传动耦合] 小轮对应转角: {np.rad2deg(pinion_offset_rad):.2f} 度 (传动比 i={ratio:.2f})")
            cols_p_coarse = pts_p_coarse.shape[2]
            rows_p_coarse = pts_p_coarse.shape[1]
            R_only = T[:3, :3]
            
            def get_pinion_mesh_coarse(d_phi):
                total_angle = base_angle_p + d_phi
                # 变换点
                pts_p_rot = rotate_z(pts_p_coarse, total_angle)
                pts_flat = pts_p_rot.reshape(3, -1)
                pts_homo = np.vstack((pts_flat, np.ones((1, pts_flat.shape[1]))))
                pts_trans_flat = (T @ pts_homo)[:3, :]
                pts_trans_grid = pts_trans_flat.reshape(3, rows_p_coarse, cols_p_coarse)
                
                # 变换法向量
                nrm_p_rot = rotate_z(nrm_p_coarse, total_angle)
                nrm_flat = nrm_p_rot.reshape(3, -1)
                nrm_trans_flat = R_only @ nrm_flat
                nrm_trans_grid = nrm_trans_flat.reshape(3, rows_p_coarse, cols_p_coarse)
                
                # 三角化
                tris, tri_nrms = contact_physics.triangulate_structured_grid_with_normals(
                    pts_trans_grid[0], pts_trans_grid[1], pts_trans_grid[2],
                    nrm_trans_grid[0], nrm_trans_grid[1], nrm_trans_grid[2]
                )
                return tris, tri_nrms
                
            def compute_gap_coarse(d_phi):
                pinion_tris, pinion_nrms = get_pinion_mesh_coarse(d_phi)
                gaps, is_valid, _ = contact_physics.compute_gap_with_validity(
                    pts_g_flat_coarse, nrm_g_flat_coarse, pinion_tris, pinion_nrms
                )
                if not np.any(is_valid): return float('inf')
                valid_gaps = gaps[is_valid]
                # 穿透(负)优先，否则最小间隙(正)
                if np.any(valid_gaps < 0): return np.min(valid_gaps)
                return np.min(valid_gaps)

            # 执行粗搜索
            search_range = np.deg2rad(12.0)
            n_coarse = 25
            test_angles = np.linspace(-search_range, search_range, n_coarse)
            print(f"  - [Phase 1 Run] 粗搜索: ±{np.rad2deg(search_range):.1f}°, {n_coarse}步")
            
            vals = []
            for angle in test_angles:
                val = compute_gap_coarse(angle)
                vals.append(val if val != float('inf') else 100.0)
                
            # 寻找 Bracket
            bracket = None
            for i in range(len(vals) - 1):
                v1 = vals[i]
                v2 = vals[i+1]
                if v1 != 100.0 and v2 != 100.0 and (v1 * v2 < 0):
                    bracket = (test_angles[i], test_angles[i+1])
                    print(f"    找到交界区间: [{np.rad2deg(bracket[0]):.2f}, {np.rad2deg(bracket[1]):.2f}] 度")
                    break
            
            # 如果没找到 Bracket，退回到粗结果最佳值作为中心及小范围
            if not bracket:
                valid_vals = [(test_angles[i], vals[i]) for i in range(len(vals)) if vals[i] != 100.0]
                if not valid_vals:
                    print("错误: 无有效接触!")
                    return
                # 找绝对值最小
                best_coarse = min(valid_vals, key=lambda x: abs(x[1]))
                center_angle = best_coarse[0]
                # 就在这个附近 +/- 1度 细搜
                bracket = (center_angle - np.deg2rad(1.0), center_angle + np.deg2rad(1.0))
                print(f"    未找到符号翻转，使用最佳点附近区间: [{np.rad2deg(bracket[0]):.2f}, {np.rad2deg(bracket[1]):.2f}]")

            # =========================================================================
            # Phase 2: Refinement & Fine Optimization
            # =========================================================================
            refine_factor = 4
            print(f"  - [Phase 2 Setup] 执行网格细化 (Factor={refine_factor})...")
            pts_g, nrm_g = refine_surface_mesh(pts_g_coarse, nrm_g_coarse, factor=refine_factor)
            pts_p, nrm_p = refine_surface_mesh(pts_p_coarse, nrm_p_coarse, factor=refine_factor)
            print(f"    细化后网格: {pts_g.shape[1]}x{pts_g.shape[2]}")
            
            # 3b. 准备细化后的大轮 (Final Gear Data)
            pts_g_aligned = rotate_z(pts_g, angle_g)
            nrm_g_aligned = rotate_z(nrm_g, angle_g)
            
            pts_g_flat = pts_g_aligned.reshape(3, -1)
            nrm_g_flat = nrm_g_aligned.reshape(3, -1)
            
            X_g = pts_g_aligned[0, :, :]
            Y_g = pts_g_aligned[1, :, :]
            Z_g = pts_g_aligned[2, :, :]
            NX_g = nrm_g_aligned[0, :, :]
            NY_g = nrm_g_aligned[1, :, :]
            NZ_g = nrm_g_aligned[2, :, :] # Needed for viz
            
            # 4b. 定义细化后的小轮变换
            rows_p, cols_p = pts_p.shape[1], pts_p.shape[2]
            
            def get_transformed_pinion_mesh(d_phi):
                """获取变换后的 *高分辨率* 小齿轮网格"""
                total_angle = base_angle_p + d_phi
                
                # 变换点
                pts_p_rot = rotate_z(pts_p, total_angle)
                pts_flat = pts_p_rot.reshape(3, -1)
                pts_homo = np.vstack((pts_flat, np.ones((1, pts_flat.shape[1]))))
                pts_trans_flat = (T @ pts_homo)[:3, :]
                pts_trans_grid = pts_trans_flat.reshape(3, rows_p, cols_p)
                
                # 变换法向量
                nrm_p_rot = rotate_z(nrm_p, total_angle)
                nrm_flat = nrm_p_rot.reshape(3, -1)
                nrm_trans_flat = R_only @ nrm_flat
                nrm_trans_grid = nrm_trans_flat.reshape(3, rows_p, cols_p)
                
                # 三角化
                tris, tri_nrms = contact_physics.triangulate_structured_grid_with_normals(
                    pts_trans_grid[0], pts_trans_grid[1], pts_trans_grid[2],
                    nrm_trans_grid[0], nrm_trans_grid[1], nrm_trans_grid[2]
                )
                return tris, tri_nrms, pts_trans_grid

            def compute_valid_min_gap(d_phi):
                """高分辨率间隙计算"""
                pinion_tris, pinion_tri_normals, _ = get_transformed_pinion_mesh(d_phi)
                gaps, is_valid, _ = contact_physics.compute_gap_with_validity(
                    pts_g_flat, nrm_g_flat, pinion_tris, pinion_tri_normals
                )
                
                valid_mask = is_valid
                invalid_mask = ~is_valid
                
                if not np.any(valid_mask):
                    return float('inf'), True, 0, len(pts_g_flat[0])
                
                valid_gaps = gaps[valid_mask]
                has_interference = np.any(valid_gaps < -0.01) # 10um tol
                
                if has_interference:
                    min_gap = np.min(valid_gaps)
                else:
                    min_gap = np.min(valid_gaps)
                    
                return min_gap, has_interference, np.sum(valid_mask), np.sum(invalid_mask)
            
            # 5. 执行 精确优化 (Phase 2 Run)
            best_d_phi = None
            min_gap_final = float('inf')
            
            if bracket:
                print(f"  - [Phase 2 Run] 精确求根 (Brent Optimization)...")
                try:
                    def objective(phi):
                        # 确保返回非inf
                        g, _, _, _ = compute_valid_min_gap(phi)
                        return g if g != float('inf') else 100.0
                        
                    best_d_phi, res = brentq(objective, bracket[0], bracket[1], full_output=True, xtol=1e-6)
                    min_gap_final = 0.0
                    print(f"    收敛! 迭代次数: {res.iterations}")
                except Exception as e:
                    print(f"    优化失败: {e}")
                    best_d_phi = (bracket[0]+bracket[1])/2
            else:
                 # fallback if no bracket from Phase 1
                 best_d_phi = bracket[0] if bracket else 0.0

            print(f"  - 最佳微调角: {np.rad2deg(best_d_phi):.4f} 度")
            
            # --- 最终计算 (Final Calculation) ---
            # 获取最终的高分辨率接触斑
            pinion_tris, pinion_tri_normals, pts_p_grid = get_transformed_pinion_mesh(best_d_phi)
            gaps, is_valid, _ = contact_physics.compute_gap_with_validity(
                pts_g_flat, nrm_g_flat, pinion_tris, pinion_tri_normals
            )
            
            # 统计有效接触点
            n_valid = np.sum(is_valid)
            n_invalid = np.sum(~is_valid)
            
            contact_mask = is_valid & (gaps < threshold)
            contact_points = pts_g_flat[:, contact_mask] # Refined contact points
            
            # 计算最小间隙用于显示 (优先显示穿透深度)
            if np.any(is_valid):
                valid_gaps = gaps[is_valid]
                min_gap_display = np.min(valid_gaps)
            else:
                min_gap_display = min_gap_final
            
            print(f"  - 最小有效间隙: {min_gap_display*1000:.2f} um")
            print(f"  - 有效接触点: {n_valid}, 边缘点: {n_invalid}")
            
            # 统计穿透情况 (gap < 0 表示穿透)
            n_penetrating = np.sum((gaps < 0) & is_valid)
            max_penetration = -np.min(gaps[is_valid]) if n_penetrating > 0 else 0
            
            print(f"  - 接触斑点数: {contact_points.shape[1]} (阈值 < {threshold*1000:.1f} um, 仅有效接触)")
            if n_penetrating > 0:
                print(f"  - 穿透点数: {n_penetrating}, 最大穿透深度: {max_penetration*1000:.2f} um")
            
            # ---- 接触点坐标详细输出 ----
            if contact_points.shape[1] > 0:
                # 接触点质心
                centroid = np.mean(contact_points, axis=1)
                print(f"\n  ===== 接触点坐标分析 =====")
                print(f"  接触质心 (X, Y, Z): ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}) mm")
                
                # 接触点包围盒
                cp_min = np.min(contact_points, axis=1)
                cp_max = np.max(contact_points, axis=1)
                print(f"  接触范围 X: [{cp_min[0]:.4f}, {cp_max[0]:.4f}] mm (跨度 {cp_max[0]-cp_min[0]:.4f})")
                print(f"  接触范围 Y: [{cp_min[1]:.4f}, {cp_max[1]:.4f}] mm (跨度 {cp_max[1]-cp_min[1]:.4f})")
                print(f"  接触范围 Z: [{cp_min[2]:.4f}, {cp_max[2]:.4f}] mm (跨度 {cp_max[2]-cp_min[2]:.4f})")
                
                # 计算接触点在齿面上的相对位置 (用柱坐标 R 和 Z 表示)
                # R = sqrt(X² + Y²) 对应齿高方向 (大齿轮: R大=齿顶, R小=齿根)
                # Z 对应齿宽方向 (大齿轮: Z大=大端heel, Z小=小端toe)
                R_contact = np.sqrt(contact_points[0]**2 + contact_points[1]**2)
                Z_contact = contact_points[2]
                
                # 齿面全部点的 R 和 Z 范围 (作为参考)
                R_all = np.sqrt(pts_g_flat[0]**2 + pts_g_flat[1]**2)
                Z_all = pts_g_flat[2]
                
                R_min_tooth, R_max_tooth = np.min(R_all), np.max(R_all)
                Z_min_tooth, Z_max_tooth = np.min(Z_all), np.max(Z_all)
                
                # 接触质心的相对位置 (0%=root/toe, 100%=tip/heel)
                R_centroid = np.mean(R_contact)
                Z_centroid = np.mean(Z_contact)
                
                R_range = R_max_tooth - R_min_tooth
                Z_range = Z_max_tooth - Z_min_tooth
                
                if R_range > 1e-6:
                    R_pct = (R_centroid - R_min_tooth) / R_range * 100
                else:
                    R_pct = 50.0
                if Z_range > 1e-6:
                    Z_pct = (Z_centroid - Z_min_tooth) / Z_range * 100
                else:
                    Z_pct = 50.0
                
                print(f"\n  齿面范围 R(齿宽): [{R_min_tooth:.3f}, {R_max_tooth:.3f}] mm")
                print(f"  齿面范围 Z(齿高): [{Z_min_tooth:.3f}, {Z_max_tooth:.3f}] mm")
                print(f"  接触质心 R={R_centroid:.3f} mm → 齿宽位置: {R_pct:.1f}% (0%=小端toe, 100%=大端heel)")
                print(f"  接触质心 Z={Z_centroid:.3f} mm → 齿高位置: {Z_pct:.1f}% (0%=齿根, 100%=齿顶)")
                
                # 判断接触位置
                if Z_pct < 30:
                    pos_profile = "偏齿根 ⚠"
                elif Z_pct > 70:
                    pos_profile = "偏齿顶 ⚠"
                else:
                    pos_profile = "居中 ✓"
                    
                if R_pct < 30:
                    pos_face = "偏小端(toe) ⚠"
                elif R_pct > 70:
                    pos_face = "偏大端(heel) ⚠"
                else:
                    pos_face = "居中 ✓"
                
                print(f"\n  >>> 齿高方向: {pos_profile}")
                print(f"  >>> 齿宽方向: {pos_face}")
                print(f"  ============================\n")
            
            # 7. 可视化接触斑 (Contact Pattern) - 使用 PyVista
            import easy_plot as ep
            
            # 创建 PyVista 图形窗口
            gear_label = f"Gear-{gear_flank}"
            pinion_label = f"Pinion-{pinion_flank}"
            status_text = f"接触点: {contact_points.shape[1]}" if contact_points.shape[1] > 0 else "无接触点"
            fig = ep.Figure(title=f"接触斑分析: {gear_label} vs {pinion_label} ({status_text})")
            
            print(f"  - 正在生成全齿阵列可视化 (PyVista)...")
            
            # 1. 绘制大轮 Active Tooth (黑色线框)
            ep.surface(fig, X_g, Y_g, Z_g, show_edges=True, edge_color='black', 
                      face_color='lightgray', opacity=0.3, style='wireframe')
            
            # 2. 绘制大轮 Inactive Teeth (淡灰色)
            for i in range(N_gear):
                if i == g_best: 
                    continue
                delta_angle = (i - g_best) * pitch_g * s_gear
                cos_a, sin_a = np.cos(delta_angle), np.sin(delta_angle)
                X_i = X_g * cos_a - Y_g * sin_a
                Y_i = X_g * sin_a + Y_g * cos_a
                Z_i = Z_g
                ep.surface(fig, X_i, Y_i, Z_i, face_color='gray', opacity=0.1)
            
            # 3. 绘制小轮 Active Tooth (蓝色半透明)
            ep.surface(fig, pts_p_grid[0], pts_p_grid[1], pts_p_grid[2], 
                      face_color='blue', opacity=0.4)
            
            # 4. 绘制小轮 Inactive Teeth (淡青色)
            pts_p_local = surfaces[f'pinion_{pinion_flank}']['points']
            rows_p_local, cols_p_local = pts_p_local.shape[1], pts_p_local.shape[2]
            
            for i in range(N_pinion):
                if i == p_best: 
                    continue
                target_local_angle = (pitch_p * i * s_pinion) + pinion_offset_rad + best_d_phi
                pts_rot = rotate_z(pts_p_local, target_local_angle)
                pts_flat = pts_rot.reshape(3, -1)
                pts_homo = np.vstack((pts_flat, np.ones((1, pts_flat.shape[1]))))
                pts_trans = (T @ pts_homo)[:3, :]
                X_pi = pts_trans[0].reshape(rows_p_local, cols_p_local)
                Y_pi = pts_trans[1].reshape(rows_p_local, cols_p_local)
                Z_pi = pts_trans[2].reshape(rows_p_local, cols_p_local)
                ep.surface(fig, X_pi, Y_pi, Z_pi, face_color='cyan', opacity=0.2)
            
            # 5. 绘制另一侧齿面 (非接触侧, 更透明) - 所有齿
            # 确定另一侧齿面名称
            other_gear_flank = 'convex' if gear_flank == 'concave' else 'concave'
            other_pinion_flank = 'concave' if pinion_flank == 'convex' else 'convex'
            
            # 获取另一侧齿面数据
            pts_g_other = surfaces[f'gear_{other_gear_flank}']['points']
            pts_p_other = surfaces[f'pinion_{other_pinion_flank}']['points']
            rows_p_other, cols_p_other = pts_p_other.shape[1], pts_p_other.shape[2]
            
            # 绘制大轮另一侧 - 所有齿 (绿色, 非常透明)
            for i in range(N_gear):
                delta_angle = (i - g_best) * pitch_g * s_gear + angle_g
                pts_g_other_aligned = rotate_z(pts_g_other, delta_angle)
                X_g_other = pts_g_other_aligned[0]
                Y_g_other = pts_g_other_aligned[1]
                Z_g_other = pts_g_other_aligned[2]
                ep.surface(fig, X_g_other, Y_g_other, Z_g_other, 
                          face_color='lightgreen', opacity=0.08)
            
            # 绘制小轮另一侧 - 所有齿 (橙色, 非常透明)
            for i in range(N_pinion):
                target_local_angle = (pitch_p * i * s_pinion) + pinion_offset_rad + best_d_phi
                pts_p_other_rot = rotate_z(pts_p_other, target_local_angle)
                pts_p_other_flat = pts_p_other_rot.reshape(3, -1)
                pts_p_other_homo = np.vstack((pts_p_other_flat, np.ones((1, pts_p_other_flat.shape[1]))))
                pts_p_other_trans = (T @ pts_p_other_homo)[:3, :]
                X_p_other = pts_p_other_trans[0].reshape(rows_p_other, cols_p_other)
                Y_p_other = pts_p_other_trans[1].reshape(rows_p_other, cols_p_other)
                Z_p_other = pts_p_other_trans[2].reshape(rows_p_other, cols_p_other)
                ep.surface(fig, X_p_other, Y_p_other, Z_p_other, 
                          face_color='orange', opacity=0.08)
            
            print(f"  - 另一侧齿面: Gear-{other_gear_flank} (绿x{N_gear}), Pinion-{other_pinion_flank} (橙x{N_pinion})")
            
            # 5. 绘制接触斑 (红色)
            if contact_points.shape[1] > 0:
                rows, cols = pts_g_aligned.shape[1], pts_g_aligned.shape[2]
                
                # 构建接触区域的三角形网格
                contact_faces = []
                contact_pts_list = []
                
                for r in range(rows - 1):
                    for c in range(cols - 1):
                        i00 = r * cols + c
                        i10 = (r + 1) * cols + c
                        i01 = r * cols + (c + 1)
                        i11 = (r + 1) * cols + (c + 1)
                        
                        if contact_mask[i00] or contact_mask[i10] or contact_mask[i11]:
                            p00 = pts_g_flat[:, i00]
                            p10 = pts_g_flat[:, i10]
                            p11 = pts_g_flat[:, i11]
                            base_idx = len(contact_pts_list)
                            contact_pts_list.extend([p00, p10, p11])
                            contact_faces.append([3, base_idx, base_idx+1, base_idx+2])
                        
                        if contact_mask[i00] or contact_mask[i11] or contact_mask[i01]:
                            p00 = pts_g_flat[:, i00]
                            p11 = pts_g_flat[:, i11]
                            p01 = pts_g_flat[:, i01]
                            base_idx = len(contact_pts_list)
                            contact_pts_list.extend([p00, p11, p01])
                            contact_faces.append([3, base_idx, base_idx+1, base_idx+2])
                
                if len(contact_faces) > 0:
                    import pyvista as pv
                    contact_pts_arr = np.array(contact_pts_list)
                    faces_arr = np.hstack(contact_faces)
                    contact_mesh = pv.PolyData(contact_pts_arr, faces_arr)
                    fig.figure.add_mesh(contact_mesh, color='red', opacity=0.9, show_edges=True, edge_color='darkred')
                    print(f"  - 绘制接触三角形数量: {len(contact_faces)}")
            else:
                print("  - 未检测到接触斑 (可能是阈值太小或间隙过大)")
            
            # 显示图形
            fig.show()

        # 交互式角度输入
        print("\n" + "="*60)
        print("交互式接触分析")
        print("  输入格式: [D/C] 角度")
        print("    D = Drive侧 (Gear Concave vs Pinion Convex)")
        print("    C = Coast侧 (Gear Convex vs Pinion Concave)")
        print("  示例: D 0    (Drive侧, 0度偏移)")
        print("        C 1.5  (Coast侧, 1.5度偏移)")
        print("        1.5    (默认Drive侧, 1.5度偏移)")
        print("  输入 'q' 退出")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n请输入 [D/C] 角度 [或 'q' 退出]: ").strip()
                if user_input.lower() == 'q':
                    print("退出交互式分析。")
                    break
                
                # 解析输入
                parts = user_input.split()
                if len(parts) == 2:
                    side = parts[0].upper()
                    gear_offset = float(parts[1])
                elif len(parts) == 1:
                    # 只有数字，默认 Drive 侧
                    side = 'D'
                    gear_offset = float(parts[0])
                else:
                    print("[错误] 输入格式不正确，请使用: D 0 或 C 1.5")
                    continue
                
                # 确定齿面配对
                if side == 'D':
                    # Drive Side: Gear Concave vs Pinion Convex
                    run_static_analysis(surfaces, meta, 'concave', 'convex', 
                                       threshold=0.025, gear_offset_deg=gear_offset)
                elif side == 'C':
                    # Coast Side: Gear Convex vs Pinion Concave
                    run_static_analysis(surfaces, meta, 'convex', 'concave', 
                                       threshold=0.025, gear_offset_deg=gear_offset)
                else:
                    print(f"[错误] 未知侧面 '{side}'，请使用 D 或 C")
                    continue
                
            except ValueError:
                print("[错误] 请输入有效数字，格式: D 0 或 C 1.5")
            except KeyboardInterrupt:
                print("\n用户中断。")
                break
            
    except ImportError as e:
        print(f"\n[错误] 无法运行静态接触分析: {e}")
        print("请确保已安装 scipy 和 numba")
    except Exception as e:
        print(f"\n[错误] 静态接触分析运行时出错: {e}")
        import traceback
        traceback.print_exc()

    # 可视化单个齿面（带法向量）
    # plot_single_surface(surfaces, 'gear', 'concave')

