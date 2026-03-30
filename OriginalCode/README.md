# OriginalCode 文件夹文件说明

此文件夹包含项目早期的核心代码库、基础数学库以及实验性功能。虽然当前项目已迁移至 `hypoid` 包结构，但此文件夹中的许多模块仍作为基础依赖或参考实现存在。

以下是各文件的详细功能说明：

## 1. 核心数学与物理引擎

*   **`screwCalculus.py`** (核心)
    *   **用途**: 螺旋理论微积分库。
    *   **功能**: 提供了旋转矩阵 (`rotX`, `rotY`...)、齐次变换矩阵 (`TrotX`, `DH_transformation`)、李代数算子 (`hat`, `adjoint`) 以及旋量指数映射 (`expTw`) 的实现。
    *   **重要性**: 整个项目运动学计算的基石，用于描述机床运动链和齿轮副的相对运动。

*   **`BEM.py`**
    *   **用途**: 边界元法 (Boundary Element Method) 求解器。
    *   **功能**: 实现了基于 BEM 的接触力学求解器 (`simulate_bem`)，包含赫兹接触算例。用于计算齿面接触时的压力分布和变形（承载接触分析 LTCA）。

*   **`computational_geometry.py`**
    *   **用途**: 计算几何算法库。
    *   **功能**: 提供几何体求交（如 `intersCPU_shapely`）、曲线重采样 (`interp_arc`) 等算法。用于处理复杂的几何拓扑运算。

*   **`solvers.py`**
    *   **用途**: 数值求解器库。
    *   **功能**: 包含自定义实现的牛顿法求解器 (`simple_newton_solver`, `robust_newton_solver`) 和模式搜索法 (`pattern_search`)。

## 2. 几何建模与拟合

*   **`nurbs.py` / `nurbs_2.py`**
    *   **用途**: NURBS 曲面建模。
    *   **功能**: 定义了 `Nurbs` 类，支持将散点数据拟合为 B 样条曲面 (`fit`), 并计算曲率（高斯曲率、主曲率等）。支持导出为 STEP 格式以便于 CAD 交互。

*   **`tool_geometries.py`**
    *   **用途**: 刀具几何定义。
    *   **功能**: 定义各种刀具的参数化几何，如齿条轮廓 (`rack_profile2D`)、各种类型的刀盘几何 (`hiirt_tool2D`)。支持 CasADi 符号计算。

## 3. 可视化工具

*   **`easy_plot.py`**
    *   **用途**: 3D 绘图包装器 (推荐)。
    *   **功能**: 基于 `pyvista` 库封装了一套类似 MATLAB 语法的绘图工具 (`Figure`, `surface`, `line`, `quiver` 等)。既支持交互式窗口，也支持静态渲染。

*   **`easyPlot_myavi.py`**
    *   **用途**: 旧版 3D 绘图库。
    *   **功能**: 基于 `Mayavi` 库的绘图工具。目前主要使用 `easy_plot.py`。

*   **`graphical_primitives.py`**
    *   **用途**: 绘图基础图元。
    *   **功能**: 提供生成圆柱体、立方体、补丁等基础几何数据的函数，供绘图工具调用。

## 4. 具体齿轮实现与应用

*   **`palloid_bevel_gears.py`**
    *   **用途**: Palloid 等基圆锥齿轮实现。
    *   **功能**: 一个独立的模块，展示了如何利用基础库实现特定类型螺旋锥齿轮的完整流程（建模、运动学、NURBS 拟合、接触分析）。

*   **`hypoid/` (子目录)**
    *   **用途**: 准双曲面齿轮核心逻辑的早期/原始版本。
    *   **内容**:
        *   `main/core.py`: `Hypoid` 类定义，核心数据结构。
        *   `main/geometry.py`: 几何计算核心，包含共轭求解算法。
        *   `main/kinematics.py`: 机床运动学定义。
        *   `main/identification.py`: 反求工程与优化算法。
        *   `main/utils.py`: 辅助工具。

## 5. 通用工具

*   **`general_utils.py`**
    *   **用途**: 通用辅助函数。
    *   **功能**: 包含计时器、进度条 (`Waitbar`)、字典打印、文件清理等通用工具。

## 6. 实验性与高级功能

*   **`nn_micro_modification.py`**
    *   **用途**: 基于神经网络的微观修形优化。
    *   **功能**: 使用 PyTorch 实现的强化学习模型 (`HypoidSolver`)，尝试通过智能调整齿面参数来优化接触压力分布。属于 AI 辅助设计的探索性代码。

*   **`postprocessRTEC.py`**
    *   **用途**: 实验数据处理。
    *   **功能**: 用于处理 Rtec 摩擦磨损试验机的数据日志 (`.csv`) 和 3D 扫描文件 (`.bcrf`)。包含信号读取、滤波和 3D 形貌绘制功能。

*   **`FreeCAD_hiirt.py`**
    *   **用途**: FreeCAD 自动化脚本。
    *   **功能**: 用于在 FreeCAD 中自动生成 HIIrT (High Investigation of Interacting Real Tooth) 类型的齿轮几何体。

*   **`nurbs_2.py` / `nurbs_gen_2.py`**
    *   **用途**: 增强版 NURBS 库。
    *   **功能**: `nurbs_2.py` 是 `nurbs.py` 的复杂版本，增加了正则化拟合（平滑度/光顺度约束）、迭代拟合和自适应节点细分功能。

## 7. Jupyter Notebooks 与测试脚本

文件夹中包含大量的 `.ipynb` 文件，通常是用于特定算例的测试或演示：
*   **`Palloid_gear.ipynb`, `Palloid_pinion.ipynb`**: Palloid 齿轮的具体设计算例。
*   **`HIRT.ipynb`**: HIIrT 齿轮相关研究笔记。
*   **`Betti_BEM.ipynb`**: 贝蒂定理与边界元法的验证算例。
*   **`heat_equation.ipynb`**: 热传导方程求解测试（可能用于闪温计算）。
*   **`hypoid_test.ipynb`**: 准双曲面齿轮类的单元测试。

## 8. 无关/其他文件

*   **`recoil_macro_PUBG.py`**: 游戏鼠标宏脚本 (与项目无关)。
*   **`snake.py`**: 贪吃蛇小游戏 (与项目无关)。
*   **`Qt-app.py`**: PyQt 界面测试脚本。

