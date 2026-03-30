# -*- coding: utf-8 -*-
"""非交互式接触分析 - 输出接触坐标到文件"""
import sys, os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

import numpy as np

# 复用 hypoid_contact 中的工具函数 (这些在模块级别定义)
from hypoid_contact import (load_gear_surfaces, get_pinion_transform, 
                            rotate_z, transform_points, 
                            find_best_meshing_pair_indices)

from scipy.optimize import brentq
import contact_physics
from nurbs_surface import refine_surface_mesh

# 输出到文件
log_file = open('contact_result.txt', 'w', encoding='utf-8')
def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

# 加载数据
try:
    surfaces, meta = load_gear_surfaces('optimized_surfaces.npz')
    log("成功加载: optimized_surfaces.npz")
except FileNotFoundError:
    surfaces, meta = load_gear_surfaces('all_surfaces.npz')
    log("成功加载: all_surfaces.npz")

gear_flank = 'concave'
pinion_flank = 'convex'
threshold = 0.025
gear_offset_deg = 0.0

log(f"\n[分析: {gear_flank} vs {pinion_flank}]")

pts_g_coarse = surfaces[f'gear_{gear_flank}']['points']
nrm_g_coarse = surfaces[f'gear_{gear_flank}']['normals']
pts_p_coarse = surfaces[f'pinion_{pinion_flank}']['points']
nrm_p_coarse = surfaces[f'pinion_{pinion_flank}']['normals']

T = get_pinion_transform(meta)
hand = meta.get('hand', 'right')
N_gear = meta.get('N_gear', 75)
N_pinion = meta.get('N_pinion', 5)

g_best, p_best = find_best_meshing_pair_indices(pts_g_coarse, pts_p_coarse, N_gear, N_pinion, T, hand)
log(f"最佳啮合对: Gear={g_best}, Pinion={p_best}")

pitch_g = 2 * np.pi / N_gear
s_gear = -1 if hand.lower() == 'right' else 1
s_pinion = -s_gear
gear_offset_rad = np.deg2rad(gear_offset_deg)
angle_g = pitch_g * g_best * s_gear + gear_offset_rad

ratio = N_gear / N_pinion
pinion_offset_rad = gear_offset_rad * ratio * (s_pinion / s_gear)
base_angle_p = 2 * np.pi / N_pinion * p_best * s_pinion + pinion_offset_rad
R_only = T[:3, :3]

# Coarse search
pts_g_aligned_c = rotate_z(pts_g_coarse, angle_g)
nrm_g_aligned_c = rotate_z(nrm_g_coarse, angle_g)
pts_g_flat_c = pts_g_aligned_c.reshape(3, -1)
nrm_g_flat_c = nrm_g_aligned_c.reshape(3, -1)
rows_c, cols_c = pts_p_coarse.shape[1], pts_p_coarse.shape[2]

def get_pinion_c(d_phi):
    tot = base_angle_p + d_phi
    pr = rotate_z(pts_p_coarse, tot)
    pf = pr.reshape(3, -1)
    ph = np.vstack((pf, np.ones((1, pf.shape[1]))))
    pt = (T @ ph)[:3, :].reshape(3, rows_c, cols_c)
    nr = rotate_z(nrm_p_coarse, tot)
    nf = nr.reshape(3, -1)
    nt = (R_only @ nf).reshape(3, rows_c, cols_c)
    tris, tn = contact_physics.triangulate_structured_grid_with_normals(pt[0], pt[1], pt[2], nt[0], nt[1], nt[2])
    return tris, tn

def gap_c(d_phi):
    tris, tn = get_pinion_c(d_phi)
    gaps, iv, _ = contact_physics.compute_gap_with_validity(pts_g_flat_c, nrm_g_flat_c, tris, tn)
    if not np.any(iv): return float('inf')
    return np.min(gaps[iv])

sr = np.deg2rad(12.0)
test_angles = np.linspace(-sr, sr, 25)
vals = []
for a in test_angles:
    v = gap_c(a)
    vals.append(v if v != float('inf') else 100.0)

bracket = None
for i in range(len(vals)-1):
    if vals[i] != 100 and vals[i+1] != 100 and vals[i]*vals[i+1] < 0:
        bracket = (test_angles[i], test_angles[i+1])
        break
if not bracket:
    vv = [(test_angles[i], vals[i]) for i in range(len(vals)) if vals[i] != 100]
    if vv:
        best = min(vv, key=lambda x: abs(x[1]))
        bracket = (best[0] - np.deg2rad(1), best[0] + np.deg2rad(1))

log(f"Bracket: [{np.rad2deg(bracket[0]):.2f}, {np.rad2deg(bracket[1]):.2f}] deg")

# Refine
rf = 4
pts_g, nrm_g = refine_surface_mesh(pts_g_coarse, nrm_g_coarse, factor=rf)
pts_p, nrm_p = refine_surface_mesh(pts_p_coarse, nrm_p_coarse, factor=rf)
log(f"细化网格: {pts_g.shape[1]}x{pts_g.shape[2]}")

pts_g_aligned = rotate_z(pts_g, angle_g)
nrm_g_aligned = rotate_z(nrm_g, angle_g)
pts_g_flat = pts_g_aligned.reshape(3, -1)
nrm_g_flat = nrm_g_aligned.reshape(3, -1)
rows_f, cols_f = pts_p.shape[1], pts_p.shape[2]

def get_pinion_f(d_phi):
    tot = base_angle_p + d_phi
    pr = rotate_z(pts_p, tot)
    pf = pr.reshape(3, -1)
    ph = np.vstack((pf, np.ones((1, pf.shape[1]))))
    pt = (T @ ph)[:3, :].reshape(3, rows_f, cols_f)
    nr = rotate_z(nrm_p, tot)
    nf = nr.reshape(3, -1)
    nt = (R_only @ nf).reshape(3, rows_f, cols_f)
    tris, tn = contact_physics.triangulate_structured_grid_with_normals(pt[0], pt[1], pt[2], nt[0], nt[1], nt[2])
    return tris, tn, pt

def gap_f(d_phi):
    tris, tn, _ = get_pinion_f(d_phi)
    gaps, iv, _ = contact_physics.compute_gap_with_validity(pts_g_flat, nrm_g_flat, tris, tn)
    if not np.any(iv): return float('inf')
    return np.min(gaps[iv])

try:
    best_phi, _ = brentq(gap_f, bracket[0], bracket[1], full_output=True, xtol=1e-6)
except:
    best_phi = (bracket[0]+bracket[1])/2

log(f"最佳微调角: {np.rad2deg(best_phi):.4f} deg")

tris, tn, pts_p_grid = get_pinion_f(best_phi)
gaps, is_valid, _ = contact_physics.compute_gap_with_validity(pts_g_flat, nrm_g_flat, tris, tn)

contact_mask = is_valid & (gaps < threshold)
contact_points = pts_g_flat[:, contact_mask]

n_contact = contact_points.shape[1]
log(f"\n接触斑点数: {n_contact}")

if n_contact > 0:
    centroid = np.mean(contact_points, axis=1)
    log(f"\n===== 接触点坐标分析 =====")
    log(f"接触质心 (X, Y, Z): ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}) mm")
    
    cp_min = np.min(contact_points, axis=1)
    cp_max = np.max(contact_points, axis=1)
    log(f"接触范围 X: [{cp_min[0]:.4f}, {cp_max[0]:.4f}] (跨度 {cp_max[0]-cp_min[0]:.4f})")
    log(f"接触范围 Y: [{cp_min[1]:.4f}, {cp_max[1]:.4f}] (跨度 {cp_max[1]-cp_min[1]:.4f})")
    log(f"接触范围 Z: [{cp_min[2]:.4f}, {cp_max[2]:.4f}] (跨度 {cp_max[2]-cp_min[2]:.4f})")
    
    R_contact = np.sqrt(contact_points[0]**2 + contact_points[1]**2)
    Z_contact = contact_points[2]
    R_all = np.sqrt(pts_g_flat[0]**2 + pts_g_flat[1]**2)
    Z_all = pts_g_flat[2]
    
    R_min_t, R_max_t = np.min(R_all), np.max(R_all)
    Z_min_t, Z_max_t = np.min(Z_all), np.max(Z_all)
    R_cent = np.mean(R_contact)
    Z_cent = np.mean(Z_contact)
    
    R_pct = (R_cent - R_min_t) / (R_max_t - R_min_t) * 100 if (R_max_t - R_min_t) > 1e-6 else 50
    Z_pct = (Z_cent - Z_min_t) / (Z_max_t - Z_min_t) * 100 if (Z_max_t - Z_min_t) > 1e-6 else 50
    
    log(f"\n齿面范围 R(齿高): [{R_min_t:.3f}, {R_max_t:.3f}] mm")
    log(f"齿面范围 Z(齿宽): [{Z_min_t:.3f}, {Z_max_t:.3f}] mm")
    log(f"接触质心 R={R_cent:.3f} mm -> 齿高位置: {R_pct:.1f}% (0%=齿根, 100%=齿顶)")
    log(f"接触质心 Z={Z_cent:.3f} mm -> 齿宽位置: {Z_pct:.1f}% (0%=小端toe, 100%=大端heel)")

    
    if R_pct < 30: pp = "偏齿根"
    elif R_pct > 70: pp = "偏齿顶"
    else: pp = "居中 OK"
    if Z_pct < 30: pf = "偏小端(toe)"
    elif Z_pct > 70: pf = "偏大端(heel)"
    else: pf = "居中 OK"

    
    log(f"\n>>> 齿高方向: {pp}")
    log(f">>> 齿宽方向: {pf}")
    log(f"============================")
else:
    log("无接触点!")

log_file.close()
print("结果已写入 contact_result.txt")
