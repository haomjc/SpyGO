# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:52:00 2020

This is a test file to test the Hypoid class and its methods.

The Hypoid class is defined in the hypoid/main/core.py file.

"""

# Suppress VTK warnings to avoid terminal flooding
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()
from hypoid import *
from general_utils import dictprint, dataclass_print
import numpy as np
import screwCalculus as sc
import matplotlib.pyplot as plt

# Enable interactive plotting for scripts
plt.ion()

np.set_printoptions(precision=4)

# %%

SystemData = {
    'HAND': "Left",
    'taper' : "Standard",
    'hypoidOffset' : 27,  # J4-2: 偏置距 27mm
    'gearGenType' : "generated"
}

met = 2.0  # J4-2: 大端端面模数 2mm

coneData = {
    'SIGMA' : 90,           # 轴交角
    'a' : SystemData['hypoidOffset'],  # 27mm 偏置距
    'z1' : 5,               # J4-2: 小齿轮齿数
    'z2' : 75,              # J4-2: 大齿轮齿数
    'b2' : 11,              # J4-2: 大齿轮齿宽
    'betam1' : 50,          # J4-2: 中点螺旋角 50°
    'rc0' : 57.15,          # J4-2: 刀具半径 57.15mm
    'gearBaseThick' : 0.303,  # J4-2: 大齿轮齿顶高
    'pinBaseThick' : 2.868,   # J4-2: 小齿轮齿顶高
}

coneData['de2'] = coneData['z2'] * met
coneData['u'] = coneData['z2']/coneData['z1']

toothData = {
    'alphaD' : 20,
    'alphaC' : 20,
    'falphalim' : 1,
    'khap' : 1,
    'khfp' : 1.25,
    'xhm1' : 0.05,
    'jen' : 0.1,
    'xsmn' : 0.0649,
    'thetaa2' : None,
    'thetaf2' : None
}

H = Hypoid().from_macro_geometry(SystemData, toothData, coneData)
H.plot('pinion', 'both', whole_gear=True)
H.plot('gear', 'both', whole_gear=True)

# %%
# testing design data saving and loading functionality
filename = 'basic_data.json'
H.save_design_data_json(filename, 'basic')

# load the design data from json file and instantiate a new Hypoid object
H2 = Hypoid().from_file(filename)
H2.plot('pinion', 'both')


# %%
# print(H.zRbounds.pinion.concave)
# print(H.zRwithRoot.pinion)

# print(H.designData.pinion_machine_settings.concave)

dictprint(H.get_machine_settings_names())

# %%
np.set_printoptions(precision=4)

# ------------------------------------------------------------
# 方案 B: 宏观参数微调 (Macro-Geometry Adjustment)
# ------------------------------------------------------------
print(f"原始小齿轮螺旋角: {H.designData.pinion_common_data.SPIRALANGLE:.2f}")
H.designData.pinion_common_data.SPIRALANGLE += 10.0  # 增加 10 度以实现跨越式重心移动
print(f"调整后小齿轮螺旋角: {H.designData.pinion_common_data.SPIRALANGLE:.2f}")
H.compute_parameters(H.designData) # 关键：宏观几何改变后必须重算系统参数

H.identifyConjugatePinion()

# %% 保存所有齿面数据到 npz 文件
def save_all_surfaces(H, filename='all_surfaces.npz'):
    """
    保存所有齿面数据 (使用原始 surfPoints)
    包含装配参数用于正确可视化啮合位置
    """
    data = {}
    n_face = H.nFace
    
    for member, flank in [('gear', 'concave'), ('gear', 'convex'),
                          ('pinion', 'concave'), ('pinion', 'convex')]:
        # 直接使用已计算的 surfPoints
        pts = H.surfPoints.get_value(member, flank)
        nrm = H.surfNormals.get_value(member, flank)
        
        # 计算 n_profile
        n_total = pts.shape[1]
        n_profile = n_total // n_face
        
        # 重塑为 (3, n_face, n_profile) 格式
        pts_3d = pts[0:3, :].reshape(3, n_face, n_profile, order='F')
        nrm_3d = nrm[0:3, :].reshape(3, n_face, n_profile, order='F')
        
        data[f"{member}_{flank}_pts"] = pts_3d
        data[f"{member}_{flank}_nrm"] = nrm_3d
        
        print(f"  {member}_{flank}: {pts_3d.shape}")
    
    # 保存装配参数
    data['n_profile'] = n_profile
    data['n_face'] = n_face
    data['shaft_angle'] = H.designData.system_data.shaft_angle  # 轴交角 (deg)
    data['hypoid_offset'] = H.designData.system_data.hypoid_offset  # 偏置距离 (mm)
    data['hand'] = H.designData.system_data.hand  # 旋向
    data['N_gear'] = H.designData.gear_common_data.NTEETH  # 大齿轮齿数
    data['N_pinion'] = H.designData.pinion_common_data.NTEETH  # 小齿轮齿数
    
    # EPGalpha 包含位置调整参数 [E, P, G, alpha]
    # E: 偏置调整, P: 小齿轮轴向, G: 大齿轮轴向, alpha: 轴交角调整
    epga = H.EPGalpha if hasattr(H, 'EPGalpha') and H.EPGalpha else [0, 0, 0, 0]
    data['EPGalpha'] = np.array(epga)
    
    np.savez(filename, **data)
    print(f"已保存: {filename}")
    print(f"  轴交角: {data['shaft_angle']}°, 偏置: {data['hypoid_offset']}mm, 齿数: {data['N_gear']}/{data['N_pinion']}")

save_all_surfaces(H)

# %%
H.plotToolProfile("gear", "convex")

# %%
# if you dont remember the indexing just call
# dictprint(H.get_machine_settings_names())
x_index = [0,1,2,3,4,5,7,15,24,33,
           72,74,75]
# x_index = [0,1,2,3,4,5,7,15,
#            72,73,74,77]
x_index.sort()
print(x_index)
lb, ub = H.compute_identification_bounds('pinion', 'concave', x_index)

idx_tilt = x_index.index(1) if 1 in x_index else -1
idx_swiv = x_index.index(2) if 2 in x_index else -1

if idx_tilt != -1:
    val_tilt = H.designData.pinion_machine_settings.concave.TILTANGLE
    lb[idx_tilt] = max(lb[idx_tilt], val_tilt - 1.0)
    ub[idx_tilt] = min(ub[idx_tilt], val_tilt + 1.0)
if idx_swiv != -1:
    val_swiv = H.designData.pinion_machine_settings.concave.SWIVELANGLE
    lb[idx_swiv] = max(lb[idx_swiv], val_swiv - 1.0)
    ub[idx_swiv] = min(ub[idx_swiv], val_swiv + 1.0)

solver, settings = H.buildIdentificationProblem('pinion', 'concave', x_index, lb, ub, zR=None, problem_type='ease-off', bound_points_tol=1)
if isinstance(settings, dict):
    print("SETTINGS KEYS:", list(settings.keys())[:5])
else:
    print("SETTINGS IS NOT DICT. TYPE:", type(settings))


# %%
# import matlab.engine
# eng = matlab.engine.start_matlab()

# compute the target points
v5DoF = np.array([0, 0, 50, 25, 0])/1000 # Concave/Coast side 维持原参数避免优化发散
# v5DoF = np.array([1,0,2,4,0]) # PA, SA, PC, LC, TW
E_fun = ease_off_5DoF(v5DoF)

num_profile, num_face = 11, 22
U, V = np.meshgrid(np.linspace(-1,1,num_face), np.linspace(-1, 1, num_profile))
E = E_fun(U, V)

Z, R = H.compute_zr_grid('pinion', 'concave', 11, 22)

base_points = H.identificationProblemEaseOff.pinion.concave['base_points'].squeeze()
base_normals = H.identificationProblemEaseOff.pinion.concave['base_normals'].squeeze()
target_points = base_points - (E.flatten(order = 'F')) * base_normals

EO = np.sum((base_points - target_points) * base_normals, axis = 0)
plot_ease_off(EO.reshape(Z.shape, order = 'F'), Z, R, aspect_ratio=[1,1,0.010], labels=['z (mm)', 'R (mm)', r'E ($\mu$m)'])
root_points = H.identificationProblemEaseOff.pinion.concave['root_constraint']['points']

target_points = np.concatenate((target_points[0:3,:], root_points[0:3,:]), 1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(target_points[0,:], target_points[1,:], target_points[2,:])
ax.scatter(base_points[0,:], base_points[1,:], base_points[2,:])
# set axis equal
ax.axis('equal')

# set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()



# %%
data = H.identificationProblemEaseOff.get_value('pinion', 'concave')


for key in data.keys():
    print(key)

# ------------------------------------------------------------
# 启用优化求解 (User Requested)
# ------------------------------------------------------------
print("\n开始执行共轭齿面识别优化中...")
# evaluate_identification_problem returns (settings, residuals)
new_settings, residuals = evaluate_identification_problem(solver, settings, target_points)
print(f"Concave Maximum Residual: {np.max(np.abs(residuals)):.4f} mm")

# 更新机床参数 (Concave)
print("更新小齿轮机床参数 (Concave)...")
H.designData.update_settings('pinion', 'concave', x_index, new_settings)
# 重新计算齿面点 (update_settings 仅修改 designData，需要重新采样以更新 surfPoints)
print("重新采样 pinion concave 齿面...")
H.sample_surface('pinion', 'concave')

# %%
# ============================================================
# Pinion Convex 侧 ease-off 优化 (Drive Side Contact)
# ============================================================
print("\n" + "="*60)
print("开始 Pinion Convex 侧 ease-off 优化...")
print("="*60)

# 1. 构建 convex 识别问题 (使用相同的 x_index)
x_index_cvx = x_index.copy()
lb_cvx, ub_cvx = H.compute_identification_bounds('pinion', 'convex', x_index_cvx)

# 强制限制 TILTANGLE (1) 和 SWIVELANGLE (2) 防止求解器发生大幅度索引位移跑飞
idx_tilt = x_index_cvx.index(1) if 1 in x_index_cvx else -1
idx_swiv = x_index_cvx.index(2) if 2 in x_index_cvx else -1

if idx_tilt != -1:
    val_tilt = H.designData.pinion_machine_settings.convex.TILTANGLE
    lb_cvx[idx_tilt] = max(lb_cvx[idx_tilt], val_tilt - 1.0)
    ub_cvx[idx_tilt] = min(ub_cvx[idx_tilt], val_tilt + 1.0)
if idx_swiv != -1:
    val_swiv = H.designData.pinion_machine_settings.convex.SWIVELANGLE
    lb_cvx[idx_swiv] = max(lb_cvx[idx_swiv], val_swiv - 1.0)
    ub_cvx[idx_swiv] = min(ub_cvx[idx_swiv], val_swiv + 1.0)

solver_cvx, settings_cvx = H.buildIdentificationProblem(
    'pinion', 'convex', x_index_cvx, lb_cvx, ub_cvx, 
    zR=None, problem_type='ease-off', bound_points_tol=1
)

# 2. 计算 convex ease-off 目标点
# v5DoF_cvx: [PA, SA, PC, LC, Twist]
# PA=-200, SA=200: 温和拉动避免失温, PC=120/LC=80: 适度收缩
v5DoF_cvx = np.array([-200, 200, 120, 80, 0])/1000 
E_fun_cvx = ease_off_5DoF(v5DoF_cvx)

Z_cvx, R_cvx = H.compute_zr_grid('pinion', 'convex', num_profile, num_face)
U_cvx, V_cvx = np.meshgrid(np.linspace(-1, 1, num_face), np.linspace(-1, 1, num_profile))
E_cvx = E_fun_cvx(U_cvx, V_cvx)

base_points_cvx = H.identificationProblemEaseOff.pinion.convex['base_points'].squeeze()
base_normals_cvx = H.identificationProblemEaseOff.pinion.convex['base_normals'].squeeze()
target_points_cvx = base_points_cvx - (E_cvx.flatten(order='F')) * base_normals_cvx

# 可视化 convex ease-off
EO_cvx = np.sum((base_points_cvx - target_points_cvx) * base_normals_cvx, axis=0)
plot_ease_off(EO_cvx.reshape(Z_cvx.shape, order='F'), Z_cvx, R_cvx, 
              aspect_ratio=[1,1,0.010], labels=['z (mm)', 'R (mm)', 'E ($\\mu$m)'])

# 添加齿根约束点
root_points_cvx = H.identificationProblemEaseOff.pinion.convex['root_constraint']['points']
target_points_cvx = np.concatenate((target_points_cvx[0:3,:], root_points_cvx[0:3,:]), 1)

# 3. 执行优化求解
print("\n开始执行 Convex 共轭齿面识别优化中...")
new_settings_cvx, residuals_cvx = evaluate_identification_problem(solver_cvx, settings_cvx, target_points_cvx)
print(f"Convex Maximum Residual: {np.max(np.abs(residuals_cvx)):.4f} mm")

# 4. 更新机床参数 (Convex)
print("更新小齿轮机床参数 (Convex)...")
H.designData.update_settings('pinion', 'convex', x_index_cvx, new_settings_cvx)
# 重新计算齿面点
print("重新采样 pinion convex 齿面...")
H.sample_surface('pinion', 'convex')

# %%
# ============================================================
# 保存最终优化结果 (Concave + Convex 都已优化)
# ============================================================
output_filename = 'optimized_surfaces.npz'
save_all_surfaces(H, filename=output_filename)
print(f"优化后的齿面已保存至: {output_filename} (Concave + Convex 均已优化)")

# identified_data = H.identificationProblemEaseOff.designData.update_settings('pinion', 'concave', x_index, new_settings, return_copy=True)

# print(identified_data.pinion_cutter_data.concave)
# print('\n\n')
# print(H.designData.pinion_cutter_data.concave)

# %%
n_face = 10
n_flank = 20
z,R = H.compute_zr_grid('pinion', 'concave', n_face, n_flank, active_flank=True)
zR = np.vstack((z.flatten(order='F'), R.flatten(order = 'F'))).T
# %matplotlib qt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(zR[:,0], zR[:,1], c = 'blue')
ax.axis('equal')
plt.show()




# %%
H.surfTriplets.gear.concave[2,:]

# %%
print(zR.shape)
print(H.designData.system_data.hypoid_offset)
p, n, triplets_conj, zRconj, psi_P, angular_ease_off, v_pg_p, omega, psi_G, offset_psi =  H.compute_conjugate_points_to_gear('concave', zR, [0,0,0,0], 0)

# %%
X = p[0,:].reshape(n_face, n_flank, order='F').T
Y = p[1,:].reshape(n_face, n_flank, order='F').T
Z = p[2,:].reshape(n_face, n_flank, order='F').T

import easy_plot as ep
F = ep.Figure()
S = ep.surface(F,X, Y, Z)
F.show()

# %%

H.plot_zr_bounds('pinion', 'concave')

# %%

z, R = H.compute_zr_grid('pinion', 'concave', 20, 30)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.pcolormesh(z, R, np.zeros_like(z), edgecolors='k', linewidth=0.5)
plt.axis('equal')
H.plot_zr_bounds('pinion', 'concave')

H.zRbounds.pinion.concave[0,:]



# %%
points, _, _ = H.samplezR([], [], 'gear', 'concave')
points_cvx, _, _ = H.samplezR([], [], 'gear', 'convex')
print(points.shape)

points = np.squeeze(points)
print(points.shape)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[0,:], points[1,:], points[2,:])
ax.scatter(points_cvx[0,:], points_cvx[1,:], points_cvx[2,:])
# set axis equal
ax.axis('equal')

# set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()



# %%
from hypoid.main.utils import machine_settings_index
print(H.get_settings_index(machine_settings_names=['RADIALSETTING', 'SPHERICALRADIUS']))

dd = H.get_machine_settings_names()

dictprint(machine_settings_index(completing=True))


# %%
from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt
import sys

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication([])

dark_stylesheet = """
QWidget {
    background-color: #2b2b2b;
    color: #f0f0f0;
}

QPushButton {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    padding: 5px;
}

QPushButton:hover {
    background-color: #444444;
}
"""

app.setStyleSheet(dark_stylesheet)

class SettingsSelector(QtWidgets.QDialog):
    def __init__(self, settings_list):
        super().__init__()
        self.setWindowTitle("Select Optimization Settings")

        self.settings_list = settings_list
        self.selected_settings = []
        self.bounds = {}

        self.layout = QtWidgets.QVBoxLayout()
        self.table = QtWidgets.QTableWidget(len(settings_list), 4)
        self.table.setHorizontalHeaderLabels(["Select", "Current value", "Min Bound", "Max Bound"])

        # Set background color for table cells
        self.table.setStyleSheet("""
        QTableWidget {
            background-color: #2b2b2b;
            gridline-color: #555555;
            color: #f0f0f0;
        }

        QHeaderView::section {
            background-color: #3c3c3c;
            color: #f0f0f0;
        }
        """)
        for i, s in enumerate(settings_list):
            # Checkbox
            chk_item = QtWidgets.QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk_item.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(i, 0, chk_item)
            
            # Current value
            current_item = QtWidgets.QTableWidgetItem("0")
            self.table.setItem(i, 1, current_item)

            # Min Bound
            min_item = QtWidgets.QTableWidgetItem("0")
            self.table.setItem(i, 2, min_item)
            
            # Max Bound
            max_item = QtWidgets.QTableWidgetItem("10")
            self.table.setItem(i, 3, max_item)

        self.layout.addWidget(self.table)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(self.ok_clicked)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        self.layout.addLayout(btn_layout)

        self.setLayout(self.layout)

    def ok_clicked(self):
        for i, s in enumerate(self.settings_list):
            item = self.table.item(i, 0)
            if item.checkState() == Qt.CheckState.Checked:
                self.selected_settings.append(s)
                min_val = float(self.table.item(i, 1).text())
                max_val = float(self.table.item(i, 2).text())
                self.bounds[s] = (min_val, max_val)
        self.accept()

def select_settings(settings_list):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    dialog = SettingsSelector(settings_list)
    if dialog.exec() == QtWidgets.QDialog.Accepted:
        return dialog.selected_settings, dialog.bounds
    else:
        return [], {}

# Example usage
# if __name__ == "__main__":
# settings = ["feed_rate", "spindle_speed", "tool_diameter", "cut_depth"]
# sel, bnds = select_settings(settings)
# print(sel)
# print(bnds)

# Keep the script running to prevent windows from closing
input("Press Enter to exit...")



