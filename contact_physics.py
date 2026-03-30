import numpy as np
from numba import njit, prange
import math

# 常量定义
FACE_TRI_INTERIOR = 0
FACE_EDGE = 1
FACE_VERTEX = 2

@njit(fastmath=True)
def makef3(x, y, z):
    return np.array([x, y, z], dtype=np.float64)

@njit(fastmath=True)
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit(fastmath=True)
def sub(a, b):
    return np.array([a[0]-b[0], a[1]-b[1], a[2]-b[2]], dtype=np.float64)

@njit(fastmath=True)
def add(a, b):
    return np.array([a[0]+b[0], a[1]+b[1], a[2]+b[2]], dtype=np.float64)

@njit(fastmath=True)
def mul(a, s):
    return np.array([a[0]*s, a[1]*s, a[2]*s], dtype=np.float64)

@njit(fastmath=True)
def len_sq(a):
    return dot(a, a)

@njit(fastmath=True)
def length(a):
    return math.sqrt(dot(a, a))

@njit(fastmath=True)
def cross(a, b):
    """向量叉积"""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ], dtype=np.float64)

@njit(fastmath=True)
def closest_point_triangle(p, a, b, c):
    """
    计算点 p 到三角形 abc 的最近点及距离。
    移植自 OGC ogc_physics.cu
    返回: (closest_point, distance_sq)
    """
    ab = sub(b, a)
    ac = sub(c, a)
    ap = sub(p, a)
    
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    
    # 区域 1: A 顶点区域
    if d1 <= 0.0 and d2 <= 0.0:
        c_pt = a
        return c_pt, len_sq(sub(p, c_pt))
        
    bp = sub(p, b)
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    
    # 区域 2: B 顶点区域
    if d3 >= 0.0 and d4 <= d3:
        c_pt = b
        return c_pt, len_sq(sub(p, c_pt))
        
    vc = d1*d4 - d3*d2
    
    # 区域 3: AB 边区域
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        c_pt = add(a, mul(ab, v))
        return c_pt, len_sq(sub(p, c_pt))
        
    cp = sub(p, c)
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)
    
    # 区域 4: C 顶点区域
    if d6 >= 0.0 and d5 <= d6:
        c_pt = c
        return c_pt, len_sq(sub(p, c_pt))
        
    vb = d5*d2 - d1*d6
    
    # 区域 5: AC 边区域
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        c_pt = add(a, mul(ac, w))
        return c_pt, len_sq(sub(p, c_pt))
        
    va = d3*d6 - d5*d4
    
    # 区域 6: BC 边区域
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        c_pt = add(b, mul(sub(c, b), w))
        return c_pt, len_sq(sub(p, c_pt))
        
    # 区域 0: 面内部
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    # u = 1 - v - w
    # c_pt = a + ab*v + ac*w
    c_pt = add(a, add(mul(ab, v), mul(ac, w)))
    return c_pt, len_sq(sub(p, c_pt))

@njit(parallel=True, fastmath=True)
def compute_gap_field(target_points, triangles):
    """
    计算 target_points (3, N) 到 三角形集合 triangles (M, 3, 3) 的最小距离
    返回: dists (N,)
    """
    n_pts = target_points.shape[1]
    n_tris = triangles.shape[0]
    
    dists = np.empty(n_pts, dtype=np.float64)
    
    for i in prange(n_pts):
        p = target_points[:, i]
        min_dist_sq = np.inf
        
        # 暴力搜索最近的三角形
        # 注意: 对于每一点遍历所有三角形，复杂度 O(N*M)
        # 优化: 实际 OGC 使用 BVH/Grid 加速，这里利用 CPU 并行硬算 (适合 N, M < 10000)
        for j in range(n_tris):
            a = triangles[j, 0, :]
            b = triangles[j, 1, :]
            c = triangles[j, 2, :]
            
            _, d_sq = closest_point_triangle(p, a, b, c)
            
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
        
        dists[i] = math.sqrt(min_dist_sq)
        
    return dists


@njit(fastmath=True)
def closest_point_triangle_signed(p, a, b, c, flip_normal=False):
    """
    计算点 p 到三角形 abc 的有符号距离
    
    符号定义:
      正值 = 点在法向量方向外侧 (未接触/间隙)
      负值 = 点在法向量反方向 (穿透)
    
    参数:
      flip_normal: 如果为True，翻转法向量方向
    
    返回: (closest_point, signed_distance)
    """
    # 1. 计算三角形法向量 (右手定则: ab × ac)
    ab = sub(b, a)
    ac = sub(c, a)
    n = cross(ab, ac)
    n_len = length(n)
    
    if n_len > 1e-12:
        n = mul(n, 1.0 / n_len)  # 归一化
    else:
        # 退化三角形，无法定义法向量
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    # 翻转法向量
    if flip_normal:
        n = mul(n, -1.0)
    
    # 2. 计算最近点和无符号距离
    c_pt, dist_sq = closest_point_triangle(p, a, b, c)
    dist = math.sqrt(dist_sq)
    
    # 3. 判断符号
    # 如果点到最近点的向量与法向量同向 → 正 (外侧/间隙)
    # 如果反向 → 负 (穿透)
    to_point = sub(p, c_pt)
    sign = 1.0 if dot(to_point, n) >= 0 else -1.0
    
    return c_pt, sign * dist


@njit(parallel=True, fastmath=True)
def compute_signed_gap_field(target_points, triangles, flip_normal=False):
    """
    计算有符号距离场
    
    参数:
        target_points: (3, N) 待查询点
        triangles: (M, 3, 3) 三角形网格
        flip_normal: 是否翻转法向量方向
        
    返回:
        signed_dists: (N,) 有符号距离
            正值 = 间隙 (未接触)
            负值 = 穿透
    """
    n_pts = target_points.shape[1]
    n_tris = triangles.shape[0]
    
    signed_dists = np.empty(n_pts, dtype=np.float64)
    
    for i in prange(n_pts):
        p = target_points[:, i]
        min_abs_dist = np.inf
        best_signed_dist = np.inf
        
        for j in range(n_tris):
            a = triangles[j, 0, :]
            b = triangles[j, 1, :]
            c = triangles[j, 2, :]
            
            _, signed_d = closest_point_triangle_signed(p, a, b, c, flip_normal)
            
            # 找绝对值最小的（最近的三角形）
            abs_d = abs(signed_d)
            if abs_d < min_abs_dist:
                min_abs_dist = abs_d
                best_signed_dist = signed_d
        
        signed_dists[i] = best_signed_dist
        
    return signed_dists


@njit(fastmath=True)
def closest_point_triangle_with_normal(p, a, b, c, n):
    """
    计算点到三角形的最近点和间隙值
    
    使用连接向量判断:
    - 连接向量 v = p - closest_point (从齿面最近点指向查询点)
    - 如果 dot(v, n) < 0: 查询点在法向量背面 → 间隙 (正值)
    - 如果 dot(v, n) >= 0: 查询点在法向量正面 → 穿透 (负值)
    
    参数:
        p: 查询点 (小齿轮)
        a, b, c: 三角形顶点 (大齿轮)
        n: 法向量 (指向齿体内部)
    
    返回: (closest_point, gap)
        gap > 0: 间隙
        gap = 0: 接触
        gap < 0: 穿透
    """
    # 计算最近点和距离
    c_pt, dist_sq = closest_point_triangle(p, a, b, c)
    dist = math.sqrt(dist_sq)
    
    # 连接向量: 从齿面最近点指向查询点
    to_point = sub(p, c_pt)
    
    # 判断间隙/穿透
    # 法向量指向齿体内部，如果连接向量与法向量反向，说明查询点在外侧（间隙）
    if dot(to_point, n) < 0:
        gap = dist   # 间隙 (正值)
    else:
        gap = -dist  # 穿透 (负值)
    
    return c_pt, gap


@njit(parallel=True, fastmath=True)
def compute_signed_gap_field_with_normals(target_points, triangles, tri_normals, flip_normal=False):
    """
    使用原始法向量计算有符号距离场
    
    参数:
        target_points: (3, N) 待查询点
        triangles: (M, 3, 3) 三角形网格
        tri_normals: (M, 3) 每个三角形的原始法向量
        flip_normal: 是否翻转法向量方向
        
    返回:
        signed_dists: (N,) 有符号距离
            正值 = 间隙 (未接触)
            负值 = 穿透
    """
    n_pts = target_points.shape[1]
    n_tris = triangles.shape[0]
    
    signed_dists = np.empty(n_pts, dtype=np.float64)
    
    for i in prange(n_pts):
        p = target_points[:, i]
        min_abs_dist = np.inf
        best_signed_dist = np.inf
        
        for j in range(n_tris):
            a = triangles[j, 0, :]
            b = triangles[j, 1, :]
            c = triangles[j, 2, :]
            
            # 使用原始法向量
            n = tri_normals[j, :]
            if flip_normal:
                n = mul(n, -1.0)
            
            _, signed_d = closest_point_triangle_with_normal(p, a, b, c, n)
            
            # 找绝对值最小的（最近的三角形）
            abs_d = abs(signed_d)
            if abs_d < min_abs_dist:
                min_abs_dist = abs_d
                best_signed_dist = signed_d
        
        signed_dists[i] = best_signed_dist
        
    return signed_dists


@njit(parallel=True, fastmath=True)
def compute_gap_with_validity(target_points, target_normals, triangles, tri_normals):
    """
    计算间隙，同时检查法向量有效性来判断是否为有效接触
    
    使用连接向量法:
    - gap > 0: 间隙 (分开)
    - gap = 0: 接触
    - gap < 0: 穿透
    
    参数:
        target_points: (3, N) 小齿轮点
        target_normals: (3, N) 小齿轮法向量
        triangles: (M, 3, 3) 大齿轮三角形网格
        tri_normals: (M, 3) 大齿轮三角形法向量
        
    返回:
        gaps: (N,) 间隙值 (正=间隙, 负=穿透)
        is_valid: (N,) 是否为有效接触几何 (法向量面对面)
        closest_gear_normals: (3, N) 最近点处的大齿轮法向量
    """
    n_pts = target_points.shape[1]
    n_tris = triangles.shape[0]
    
    gaps = np.empty(n_pts, dtype=np.float64)
    is_valid = np.empty(n_pts, dtype=np.bool_)
    closest_gear_normals = np.empty((3, n_pts), dtype=np.float64)
    
    for i in prange(n_pts):
        p = target_points[:, i]
        n_p = target_normals[:, i]  # 小齿轮法向量
        
        min_abs_dist = np.inf
        best_gap = np.inf
        best_gear_normal = np.zeros(3)
        
        for j in range(n_tris):
            a = triangles[j, 0, :]
            b = triangles[j, 1, :]
            c = triangles[j, 2, :]
            n_g = tri_normals[j, :]  # 大齿轮法向量
            
            c_pt, gap = closest_point_triangle_with_normal(p, a, b, c, n_g)
            
            # 找绝对值最小的（最近的三角形）
            abs_d = abs(gap)
            if abs_d < min_abs_dist:
                min_abs_dist = abs_d
                best_gap = gap
                best_gear_normal = n_g
        
        gaps[i] = best_gap
        closest_gear_normals[:, i] = best_gear_normal
        
        # 检查法向量有效性: 两个法向量应该"面对面"
        # 即 dot(n_gear, n_pinion) < 0
        # 如果 > 0, 说明是边缘碰撞
        normal_dot = dot(best_gear_normal, n_p)
        is_valid[i] = (normal_dot < 0)
        
    return gaps, is_valid, closest_gear_normals


def triangulate_structured_grid(X, Y, Z):
    """
    将结构化网格 (Rows, Cols) 转换为三角形列表 (N_tri, 3, 3)
    """
    rows, cols = X.shape
    # 每个 Quad 切分为 2 个 Tri
    n_quads = (rows - 1) * (cols - 1)
    n_tris = n_quads * 2
    
    triangles = np.zeros((n_tris, 3, 3), dtype=np.float64)
    
    idx = 0
    for i in range(rows - 1):
        for j in range(cols - 1):
            # 提取四个顶点
            p00 = np.array([X[i, j],     Y[i, j],     Z[i, j]])
            p10 = np.array([X[i+1, j],   Y[i+1, j],   Z[i+1, j]])
            p01 = np.array([X[i, j+1],   Y[i, j+1],   Z[i, j+1]])
            p11 = np.array([X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]])
            
            # 三角形 1: (00, 10, 11)
            triangles[idx, 0, :] = p00
            triangles[idx, 1, :] = p10
            triangles[idx, 2, :] = p11
            idx += 1
            
            # 三角形 2: (00, 11, 01)
            triangles[idx, 0, :] = p00
            triangles[idx, 1, :] = p11
            triangles[idx, 2, :] = p01
            idx += 1
            
    return triangles


def triangulate_structured_grid_with_normals(X, Y, Z, NX, NY, NZ):
    """
    将结构化网格 (Rows, Cols) 转换为三角形列表，同时存储原始法向量
    
    参数:
        X, Y, Z: (Rows, Cols) 点坐标
        NX, NY, NZ: (Rows, Cols) 原始法向量分量
        
    返回:
        triangles: (N_tri, 3, 3) 三角形顶点
        tri_normals: (N_tri, 3) 每个三角形的平均法向量 (来自原始数据)
    """
    rows, cols = X.shape
    n_quads = (rows - 1) * (cols - 1)
    n_tris = n_quads * 2
    
    triangles = np.zeros((n_tris, 3, 3), dtype=np.float64)
    tri_normals = np.zeros((n_tris, 3), dtype=np.float64)
    
    idx = 0
    for i in range(rows - 1):
        for j in range(cols - 1):
            # 提取四个顶点
            p00 = np.array([X[i, j],     Y[i, j],     Z[i, j]])
            p10 = np.array([X[i+1, j],   Y[i+1, j],   Z[i+1, j]])
            p01 = np.array([X[i, j+1],   Y[i, j+1],   Z[i, j+1]])
            p11 = np.array([X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]])
            
            # 提取四个顶点的原始法向量
            n00 = np.array([NX[i, j],     NY[i, j],     NZ[i, j]])
            n10 = np.array([NX[i+1, j],   NY[i+1, j],   NZ[i+1, j]])
            n01 = np.array([NX[i, j+1],   NY[i, j+1],   NZ[i, j+1]])
            n11 = np.array([NX[i+1, j+1], NY[i+1, j+1], NZ[i+1, j+1]])
            
            # 三角形 1: (00, 10, 11) - 使用三个顶点法向量的平均
            triangles[idx, 0, :] = p00
            triangles[idx, 1, :] = p10
            triangles[idx, 2, :] = p11
            avg_n1 = (n00 + n10 + n11) / 3.0
            norm1 = np.linalg.norm(avg_n1)
            if norm1 > 1e-12:
                avg_n1 /= norm1
            tri_normals[idx, :] = avg_n1
            idx += 1
            
            # 三角形 2: (00, 11, 01)
            triangles[idx, 0, :] = p00
            triangles[idx, 1, :] = p11
            triangles[idx, 2, :] = p01
            avg_n2 = (n00 + n11 + n01) / 3.0
            norm2 = np.linalg.norm(avg_n2)
            if norm2 > 1e-12:
                avg_n2 /= norm2
            tri_normals[idx, :] = avg_n2
            idx += 1
            
    return triangles, tri_normals

def transform_points(points, d_phi, axis_origin=np.zeros(3), axis_dir=np.array([0,0,1])):
    """
    旋转点云
    points: (3, N)
    d_phi: 旋转角度 (弧度)
    """
    # 简单的绕 Z 轴旋转 (假设 axis_dir 为 Z，根据实际情况调整)
    # 实际应用中需要通用的 Rodrigues 旋转
    
    c = np.cos(d_phi)
    s = np.sin(d_phi)
    
    # 绕 Z 轴旋转矩阵
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    # 目前假设 pinion 是绕自身 Z 轴转，且中心在原点 (如果不经过原点需要平移)
    # 根据 hypoid 代码，pinion 通常已经转换到 global 坐标系。
    # 如果是绕 pinion 轴转，需要先逆变换回 pinion 局部坐标系，转完再变换回来
    # 或者如果我们知道 Pinion 轴在 Global 的向量。
    
    # 简化: 这里的 points 已经是 Global 坐标。我们需要正确的旋转轴。
    # 这是一个占位，需要在主程序中传入正确的旋转逻辑。
    return R @ points
