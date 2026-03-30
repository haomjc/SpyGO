"""
NURBS/B-Spline 曲面工具模块

提供齿面细化、法向量计算等功能，用于接触分析。
支持两种后端:
  1. scipy (默认): 使用 RectBivariateSpline，无需额外依赖
  2. geomdl (可选): 使用真正的 NURBS 曲面，需要安装 geomdl 库

Author: Refactored for HypoidGear contact analysis
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline


class BSplineSurface:
    """
    B样条曲面类 (基于 scipy)
    
    用于齿面的插值、细化和法向量计算
    """
    
    def __init__(self, pts, degree=3):
        """
        初始化 B样条曲面
        
        :param pts: (3, rows, cols) 点云数组
        :param degree: 样条阶数 (默认3, 即 Bicubic)
        """
        self.pts_original = pts
        self.degree = degree
        self.rows, self.cols = pts.shape[1], pts.shape[2]
        
        # 原始参数网格
        u = np.arange(self.rows)
        v = np.arange(self.cols)
        
        # 创建各坐标的 B样条插值器
        self._spl_x = RectBivariateSpline(u, v, pts[0], kx=degree, ky=degree)
        self._spl_y = RectBivariateSpline(u, v, pts[1], kx=degree, ky=degree)
        self._spl_z = RectBivariateSpline(u, v, pts[2], kx=degree, ky=degree)
    
    def evaluate(self, u, v):
        """
        求值曲面上的点
        
        :param u: u方向参数值 (数组或标量)
        :param v: v方向参数值 (数组或标量)
        :return: (3, ...) 点坐标
        """
        x = self._spl_x(u, v)
        y = self._spl_y(u, v)
        z = self._spl_z(u, v)
        return np.array([x, y, z])
    
    def evaluate_with_normals(self, u, v):
        """
        同时计算曲面点和法向量
        
        :param u: u方向参数值 (数组或标量)
        :param v: v方向参数值 (数组或标量)
        :return: (pts, normals) 各为 (3, ...) 数组
        """
        # 点坐标
        x = self._spl_x(u, v)
        y = self._spl_y(u, v)
        z = self._spl_z(u, v)
        
        # 偏导数 (切向量)
        dx_du = self._spl_x(u, v, dx=1, dy=0)
        dy_du = self._spl_y(u, v, dx=1, dy=0)
        dz_du = self._spl_z(u, v, dx=1, dy=0)
        
        dx_dv = self._spl_x(u, v, dx=0, dy=1)
        dy_dv = self._spl_y(u, v, dx=0, dy=1)
        dz_dv = self._spl_z(u, v, dx=0, dy=1)
        
        # 法向量 = tu × tv
        nx = dy_du * dz_dv - dz_du * dy_dv
        ny = dz_du * dx_dv - dx_du * dz_dv
        nz = dx_du * dy_dv - dy_du * dx_dv
        
        # 归一化
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-12
        nx /= norm
        ny /= norm
        nz /= norm
        
        return np.array([x, y, z]), np.array([nx, ny, nz])
    
    def refine(self, factor=3, reference_normals=None):
        """
        网格细化
        
        :param factor: 细化倍数 (整数, >=1)
        :param reference_normals: (3, rows, cols) 参考法向量，用于方向一致性校正
        :return: (pts_new, nrm_new) 细化后的点云和法向量
        """
        if factor <= 1:
            pts, nrm = self.evaluate_with_normals(
                np.arange(self.rows), np.arange(self.cols)
            )
            return pts.squeeze(), nrm.squeeze()
        
        # 细化后的参数网格
        u_new = np.linspace(0, self.rows - 1, int(self.rows * factor))
        v_new = np.linspace(0, self.cols - 1, int(self.cols * factor))
        
        pts_new, nrm_new = self.evaluate_with_normals(u_new, v_new)
        
        # 压缩维度
        pts_new = pts_new.squeeze()
        nrm_new = nrm_new.squeeze()
        
        # 方向一致性校正
        if reference_normals is not None:
            c_r, c_c = self.rows // 2, self.cols // 2
            ref_n = reference_normals[:, c_r, c_c]
            
            nc_r, nc_c = len(u_new) // 2, len(v_new) // 2
            new_n = nrm_new[:, nc_r, nc_c]
            
            if np.dot(ref_n, new_n) < 0:
                nrm_new = -nrm_new
        
        return pts_new, nrm_new


def refine_surface_mesh(pts, nrm_original=None, factor=3):
    """
    便捷函数: 使用 B样条对网格进行细化
    
    :param pts: (3, rows, cols) 原始点云
    :param nrm_original: (3, rows, cols) 原始法向量 (用于方向校正)
    :param factor: 细化倍数
    :return: (pts_new, nrm_new)
    """
    if factor <= 1:
        return pts, nrm_original
    
    surface = BSplineSurface(pts, degree=3)
    return surface.refine(factor, nrm_original)


# ============================================
# NURBS 扩展功能 (可选, 需要 geomdl 库)
# ============================================

def create_nurbs_surface(pts, degree_u=3, degree_v=3):
    """
    创建真正的 NURBS 曲面 (需要 geomdl 库)
    
    :param pts: (3, rows, cols) 控制点
    :param degree_u: U方向阶数
    :param degree_v: V方向阶数
    :return: geomdl BSpline.Surface 对象
    """
    try:
        from geomdl import BSpline
        from geomdl import knotvector
    except ImportError:
        raise ImportError("需要安装 geomdl 库: pip install geomdl")
    
    rows, cols = pts.shape[1], pts.shape[2]
    
    # 创建曲面
    surf = BSpline.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    
    # 转换控制点格式
    ctrlpts = []
    for i in range(rows):
        for j in range(cols):
            ctrlpts.append([pts[0, i, j], pts[1, i, j], pts[2, i, j]])
    
    surf.set_ctrlpts(ctrlpts, rows, cols)
    
    # 自动生成节点向量
    surf.knotvector_u = knotvector.generate(degree_u, rows)
    surf.knotvector_v = knotvector.generate(degree_v, cols)
    
    return surf


def export_to_step(pts, filename, degree=3):
    """
    导出曲面到 STEP 文件 (需要 PythonOCC)
    
    :param pts: (3, rows, cols) 点云
    :param filename: 输出文件名
    :param degree: 样条阶数
    """
    try:
        from OCC.Core.Geom import Geom_BSplineSurface
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.Interface import Interface_Static
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.TColgp import TColgp_Array2OfPnt
        from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    except ImportError:
        print("导出STEP需要 PythonOCC 库，跳过导出")
        return False
    
    rows, cols = pts.shape[1], pts.shape[2]
    
    # 创建控制点数组
    ctrl_pts = TColgp_Array2OfPnt(0, rows - 1, 0, cols - 1)
    for i in range(rows):
        for j in range(cols):
            ctrl_pts.SetValue(i, j, gp_Pnt(pts[0, i, j], pts[1, i, j], pts[2, i, j]))
    
    # 创建节点向量
    n_knots_u = rows - degree + 1
    n_knots_v = cols - degree + 1
    
    knots_u = TColStd_Array1OfReal(0, n_knots_u - 1)
    knots_v = TColStd_Array1OfReal(0, n_knots_v - 1)
    mult_u = TColStd_Array1OfInteger(0, n_knots_u - 1)
    mult_v = TColStd_Array1OfInteger(0, n_knots_v - 1)
    
    # 设置节点和重复度
    for i in range(n_knots_u):
        knots_u.SetValue(i, i / (n_knots_u - 1))
        mult_u.SetValue(i, degree + 1 if i == 0 or i == n_knots_u - 1 else 1)
    
    for i in range(n_knots_v):
        knots_v.SetValue(i, i / (n_knots_v - 1))
        mult_v.SetValue(i, degree + 1 if i == 0 or i == n_knots_v - 1 else 1)
    
    # 创建 B-Spline 曲面
    bspline_surf = Geom_BSplineSurface(
        ctrl_pts, knots_u, knots_v, mult_u, mult_v, degree, degree
    )
    
    # 创建面并导出
    face = BRepBuilderAPI_MakeFace(bspline_surf, 1e-6)
    
    writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP203")
    writer.Transfer(face.Shape(), STEPControl_AsIs)
    writer.Write(filename)
    
    print(f"STEP 文件已导出: {filename}")
    return True


# ============================================
# 测试代码
# ============================================
if __name__ == '__main__':
    # 创建测试曲面
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    U, V = np.meshgrid(u, v, indexing='ij')
    
    R = 10
    X = R * np.cos(U) * np.sin(V)
    Y = R * np.sin(U) * np.sin(V)
    Z = R * np.cos(V)
    
    pts = np.array([X, Y, Z])
    print(f"原始网格: {pts.shape}")
    
    # 测试细化
    pts_refined, nrm_refined = refine_surface_mesh(pts, factor=2)
    print(f"细化后网格: {pts_refined.shape}")
    print(f"法向量: {nrm_refined.shape}")
    
    # 验证法向量单位长度
    norms = np.linalg.norm(nrm_refined, axis=0)
    print(f"法向量长度范围: [{norms.min():.6f}, {norms.max():.6f}]")
