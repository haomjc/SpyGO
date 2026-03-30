# SpyGO - Hypoid Gear Geometry & Contact Analysis

SpyGO is a Python-based framework for the geometric modeling and static contact analysis (TCA) of spiral bevel and hypoid gears. It specifically focuses on Gleason-style CNC machine settings and 5-DoF ease-off optimization.

## Key Features

- **Ease-off Optimization (5-DoF)**: Implements Surface-to-Surface (S2S) optimization to match target ease-off topologies.
- **Machine Setting Identification**: Solves the inverse kinematics for machine settings (Radial, Tilt, Swivel, Center-back, etc.) using CasADi optimization.
- **Contact Analysis (TCA)**: Direct mesh-based signed distance calculation to determine contact centroids, contact paths, and interference patterns.
- **Coordinate Mapping**: Automatic translation of contact points into normalized tooth surface percentages (Root-to-Tip and Toe-to-Heel).

## Recent Major Fixes

- **Sign Convention**: Fixed the fundamental ease-off targeting equation. Corrected from `target = base + E*n` to `target = base - E*n`, ensuring that crowning (relief) removes material rather than causing interference.
- **Label Correction**: Rectified swapped labels in the contact output (R now correctly represents Facewidth/Toe-Heel, Z represents Profile/Root-Tip).
- **Kinematic Unlocking**: Unlocked `TILTANGLE` and `SWIVELANGLE` in the optimization variable pool (`x_index`), allowing the solver to physically achieve the requested surface twists.

## Quick Start

### 1. Perform Ease-off Optimization
Run the main testing script to calculate optimized machine settings and generate the tooth surfaces:
```bash
python hypoid_test.py
```
This will generate `optimized_surfaces.npz`.

### 2. Check Contact Pattern
Analyze the contact between the newly generated pinion and the gear:
```bash
python run_contact_check.py
```
This script outputs the contact centroid and relative position on the tooth face.

## Requirements
- `numpy`, `scipy`, `casadi`, `pyvista`, `vtk`

---
*Created by haomjc during gear optimization exploration.*
