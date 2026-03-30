# -*- coding: utf-8 -*-
"""
准双曲面齿轮参数生成与对比工具

用法:
    python generate_gear.py J4-2           # 生成配置 + 对比表
    python generate_gear.py J4-2 config    # 仅生成配置
    python generate_gear.py J4-2 compare   # 仅生成对比表
"""
import sys
import os
import json
import csv
import math

# 添加上级目录到路径
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from hypoid import Hypoid

# ==================== 配置加载 ====================

def load_config(gear_id):
    """加载指定齿轮的配置文件"""
    config_path = os.path.join(script_dir, 'configs', f'{gear_id}_input.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_generated_data(gear_id):
    """加载已生成的设计数据"""
    data_path = os.path.join(script_dir, f'basic_data_{gear_id}.json')
    if not os.path.exists(data_path):
        return None
    
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==================== 配置生成 ====================

def generate_gear_data(gear_id):
    """根据配置生成齿轮设计数据"""
    config = load_config(gear_id)
    
    print("=" * 60)
    print(f"{gear_id} 齿轮参数:")
    print(f"  小齿轮齿数 z1: {config['cone_data']['z1']}")
    print(f"  大齿轮齿数 z2: {config['cone_data']['z2']}")
    print(f"  传动比 u: {config['cone_data']['u']:.2f}")
    print(f"  偏置距 a: {config['cone_data']['a']} mm")
    print(f"  螺旋角 β: {config['cone_data']['betam1']}°")
    print(f"  刀具半径 rc0: {config['cone_data']['rc0']} mm")
    print("=" * 60)
    
    # 系统设置
    system_data = {
        "HAND": config['system_settings']['HAND'],
        "taper": config['system_settings']['taper'],
        "hypoidOffset": config['cone_data']['a'],
        "gearGenType": config['system_settings']['gearGenType']
    }
    
    # 创建 Hypoid 对象
    print("\n正在创建 Hypoid 对象...")
    H = Hypoid(nProf=11, nFace=16, nFillet=12)
    H = H.from_macro_geometry(system_data, config['tooth_data'], config['cone_data'])
    print("Hypoid 对象创建成功!")
    
    # 保存设计数据
    output_filename = os.path.join(script_dir, f'basic_data_{gear_id}.json')
    print(f"\n正在保存设计数据到 {output_filename}...")
    H.save_design_data_json(output_filename, 'basic')
    print(f"设计数据已保存到 {output_filename}")
    
    # 保存额外参数到单独文件
    extra_filename = os.path.join(script_dir, f'basic_data_{gear_id}_extra.json')
    extra_data = {
        'tooth_data': config['tooth_data'],
        'measured': config['measured']
    }
    with open(extra_filename, 'w', encoding='utf-8') as f:
        json.dump(extra_data, f, indent=4, ensure_ascii=False)
    print(f"额外参数已保存到 {extra_filename}")
    
    # 验证加载
    print(f"\n正在从 {output_filename} 加载验证...")
    H2 = Hypoid().from_file(output_filename)
    print("验证加载成功!")
    
    # 输出摘要
    print("\n" + "=" * 60)
    print("生成的配置文件摘要:")
    print(f"  传动比: {H.designData.system_data.ratio:.4f}")
    print(f"  小轮齿数: {H.designData.pinion_common_data.NTEETH}")
    print(f"  大轮齿数: {H.designData.gear_common_data.NTEETH}")
    print(f"  小轮螺旋角: {H.designData.pinion_common_data.SPIRALANGLE:.2f}°")
    print(f"  大轮螺旋角: {H.designData.gear_common_data.SPIRALANGLE:.2f}°")
    print(f"  偏置距: {H.designData.system_data.hypoid_offset} mm")
    print(f"  小轮齿宽: {H.designData.pinion_common_data.FACEWIDTH} mm")
    print(f"  大轮齿宽: {H.designData.gear_common_data.FACEWIDTH} mm")
    print("=" * 60)
    print("\n配置生成完成!")
    
    return H

# ==================== 对比表生成 ====================

def generate_comparison(gear_id):
    """生成对比表"""
    # 加载计算数据
    calc = load_generated_data(gear_id)
    if calc is None:
        print(f"未找到 {gear_id} 的计算数据，正在自动生成...")
        generate_gear_data(gear_id)
        calc = load_generated_data(gear_id)

    # 加载配置（测量值以 config 为准，避免 extra 文件过期导致对比表不更新）
    config = load_config(gear_id)

    # 加载额外参数（主要用于保存与计算数据一致的 tooth_data 快照）
    extra_path = os.path.join(script_dir, f'basic_data_{gear_id}_extra.json')   
    if os.path.exists(extra_path):
        with open(extra_path, 'r', encoding='utf-8') as f:
            extra = json.load(f)
    else:
        extra = {}

    sd = calc['system_data']
    pin = calc['pinion_common_data']
    gear = calc['gear_common_data']
    pc = calc.get('pinion_cutter_data', {}).get('concave', {})
    gc = calc.get('gear_cutter_data', {}).get('concave', {})

    tooth_data = extra.get('tooth_data', config.get('tooth_data', {}))
    measured = config.get('measured', extra.get('measured', {}))
    
    # 提取系数
    mmn = sd['NORMALMODULE']
    khap1 = tooth_data.get('khap1', 1.0)
    khfp1 = tooth_data.get('khfp1', 1.25)
    khap2 = tooth_data.get('khap2', 1.0)
    khfp2 = tooth_data.get('khfp2', 1.25)
    xhm1 = tooth_data.get('xhm1', 0.5)
    
    print(f"已加载动态参数: khap1={khap1}, khfp1={khfp1}, xhm1={xhm1}")
    
    # 计算齿高
    tooth_height = {
        'ham1': mmn * (khap1 + xhm1),
        'hfm1': mmn * (khfp1 - xhm1),
        'hm1': mmn * (khap1 + khfp1),
        'ham2': mmn * (khap2 - xhm1),
        'hfm2': mmn * (khfp2 + xhm1),
        'hm2': mmn * (khap2 + khfp2)
    }
    
    # 计算外径 (ISO标准: dae = de + 2*hae*cos(delta))
    de1 = 2 * pin['OUTERCONEDIST'] * math.sin(math.radians(pin['PITCHANGLE']))
    de2 = 2 * gear['OUTERCONEDIST'] * math.sin(math.radians(gear['PITCHANGLE']))
    hae1 = pin.get('MEANADDENDUM', tooth_height['ham1'])
    hae2 = gear.get('MEANADDENDUM', tooth_height['ham2'])
    dae1 = de1 + 2 * hae1 * math.cos(math.radians(pin['PITCHANGLE']))
    dae2 = de2 + 2 * hae2 * math.cos(math.radians(gear['PITCHANGLE']))
    
    # 中点分度圆直径
    dm1 = 2 * pin['MEANCONEDIST'] * math.sin(math.radians(pin['PITCHANGLE']))
    dm2 = 2 * gear['MEANCONEDIST'] * math.sin(math.radians(gear['PITCHANGLE']))
    
    # 对比表数据
    comparison = [
        # 基本参数
        ['基本参数', '齿数', pin['NTEETH'], gear['NTEETH'], measured.get('z1', '-'), measured.get('z2', '-'), '', ''],
        ['基本参数', '大端端面模数', '-', '-', measured.get('met', '-'), '-', 'mm', ''],
        ['基本参数', '中点法向模数', f"{mmn:.4f}", '-', '-', '-', 'mm', ''],
        ['基本参数', '齿宽', f"{pin['FACEWIDTH']:.3f}", f"{gear['FACEWIDTH']:.3f}", measured.get('b1', '-'), measured.get('b2', '-'), 'mm', ''],
        ['基本参数', '平均压力角', '20', '20', measured.get('alpha', '-'), '-', '°', ''],
        ['基本参数', '驱动面压力角', f"{sd['NOMINALDRIVEPRESSUREANGLE']:.2f}", '-', '-', '-', '°', ''],
        ['基本参数', '非驱动面压力角', f"{sd['NOMINALCOASTPRESSUREANGLE']:.2f}", '-', '-', '-', '°', ''],
        ['基本参数', '中点分度圆直径', f"{dm1:.2f}", f"{dm2:.2f}", '/', '-', 'mm', 'dm=2*Rm*sin(δ)'],
        ['基本参数', '外径(齿顶圆)', f"{dae1:.3f}", f"{dae2:.3f}", measured.get('pin_od', '-'), measured.get('gear_od', '-'), 'mm', 'dae=de+2*hae*cos(δ)'],
        ['基本参数', '中点螺旋角', f"{pin['SPIRALANGLE']:.2f}", f"{gear['SPIRALANGLE']:.2f}", measured.get('betam1', '-'), measured.get('betam2', '-'), '°', ''],
        ['基本参数', '螺旋方向', '左旋' if sd['hand']=='Left' else '右旋', '右旋' if sd['hand']=='Left' else '左旋', measured.get('hand_pin', '-'), measured.get('hand_gear', '-'), '', ''],
        ['基本参数', '偏置距', sd['hypoid_offset'], '-', measured.get('offset', '-'), '-', 'mm', ''],
        ['基本参数', '传动比', f"{sd['ratio']:.4f}", '-', '-', '-', '', ''],
        ['', '', '', '', '', '', '', ''],
        
        # 齿高参数
        ['齿高参数', '测量齿顶高(大端)', '-', '-', measured.get('pin_addendum', '-'), measured.get('gear_addendum', '-'), 'mm', ''],
        ['齿高参数', '测量齿根高(大端)', '-', '-', measured.get('pin_dedendum', '-'), measured.get('gear_dedendum', '-'), 'mm', ''],
        ['齿高参数', '测量全齿高(大端)', '-', '-', measured.get('pin_total', '-'), measured.get('gear_total', '-'), 'mm', ''],
        ['齿高参数', '设计齿顶高(中点)', f"{tooth_height['ham1']:.3f}", f"{tooth_height['ham2']:.3f}", '-', '-', 'mm', ''],
        ['齿高参数', '设计齿根高(中点)', f"{tooth_height['hfm1']:.3f}", f"{tooth_height['hfm2']:.3f}", '-', '-', 'mm', ''],
        ['齿高参数', '设计全齿高(中点)', f"{tooth_height['hm1']:.3f}", f"{tooth_height['hm2']:.3f}", '-', '-', 'mm', ''],
        ['齿高参数', '齿顶高系数', f"{khap1}(小轮), {khap2}(大轮)", '', '-', '-', '', ''],
        ['齿高参数', '顶隙系数', f"{khfp1-khap2:.2f}(小轮), {khfp2-khap1:.2f}(大轮)", '', '-', '-', '', ''],
        ['齿高参数', '齿根高系数', f"{khfp1}(小轮), {khfp2}(大轮)", '', '-', '-', '', ''],
        ['齿高参数', '变位系数', xhm1, -xhm1, '-', '-', '', ''],
        ['', '', '', '', '', '', '', ''],
        
        # 锥角参数
        ['锥角参数', '顶锥角', f"{pin['FACEANGLE']:.2f}", f"{gear['FACEANGLE']:.2f}", measured.get('face_angle_pin', '-'), measured.get('face_angle_gear', '-'), '°', ''],
        ['锥角参数', '节锥角', f"{pin['PITCHANGLE']:.2f}", f"{gear['PITCHANGLE']:.2f}", measured.get('pitch_angle_pin', '-'), measured.get('pitch_angle_gear', '-'), '°', ''],
        ['锥角参数', '根锥角', f"{pin['BACKANGLE']:.2f}", f"{gear['BACKANGLE']:.2f}", measured.get('root_angle_pin', '-'), measured.get('root_angle_gear', '-'), '°', ''],
        ['锥角参数', '齿顶角', f"{pin['FACEANGLE'] - pin['PITCHANGLE']:.2f}", f"{gear['FACEANGLE'] - gear['PITCHANGLE']:.2f}", '-', '-', '°', ''],
        ['锥角参数', '齿根角', f"{pin['PITCHANGLE'] - pin['BACKANGLE']:.2f}", f"{gear['PITCHANGLE'] - gear['BACKANGLE']:.2f}", '-', '-', '°', ''],
        ['', '', '', '', '', '', '', ''],
        
        # 锥距参数
        ['锥距参数', '外锥距', f"{pin['OUTERCONEDIST']:.3f}", f"{gear['OUTERCONEDIST']:.3f}", '-', '-', 'mm', ''],
        ['锥距参数', '中点锥距', f"{pin['MEANCONEDIST']:.3f}", f"{gear['MEANCONEDIST']:.3f}", '-', '-', 'mm', ''],
        ['锥距参数', '内锥距', f"{pin['INNERCONEDIST']:.3f}", f"{gear['INNERCONEDIST']:.3f}", '-', '-', 'mm', ''],
        ['', '', '', '', '', '', '', ''],
        
        # 齿厚参数
        ['齿厚参数', '中点法向弦齿厚', f"{pin.get('MEANNORMALCHORDALTHICKNESS', 0):.3f}", f"{gear.get('MEANNORMALCHORDALTHICKNESS', 0):.3f}", '-', '-', 'mm', ''],
        ['齿厚参数', '大端法向侧隙', '0.13', '-', measured.get('backlash', '-'), '-', 'mm', ''],
        ['', '', '', '', '', '', '', ''],
        
        # 刀具参数
        ['刀具参数', '刀具半径', f"{pin.get('MEANCUTTERRAIDUS', 0):.2f}", f"{gear.get('MEANCUTTERRAIDUS', 0):.2f}", measured.get('rc0', '-'), '-', 'mm', ''],
        ['刀具参数', '刀刃角', f"{pc.get('BLADEANGLE', 0):.1f}", f"{gc.get('BLADEANGLE', 0):.1f}", '-', '-', '°', ''],
    ]
    
    # 写入CSV
    output_path = os.path.join(script_dir, f'{gear_id}参数对比表.csv')
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['参数分类', '参数名称', '小齿轮(计算)', '大齿轮(计算)', '小齿轮(测量)', '大齿轮(测量)', '单位', '备注'])
        writer.writerow(['', '', '', '', '', '', '', ''])
        writer.writerows(comparison)
    
    print(f"\n{gear_id}参数对比表已生成: {output_path}")
    
    # 输出关键参数对比
    if 'pin_addendum' in measured:
        print("\n关键参数对比:")
        print(f"  小轮齿顶高: {tooth_height['ham1']:.3f} mm vs 测量 {measured['pin_addendum']} mm (偏差: {abs(tooth_height['ham1'] - measured['pin_addendum']):.3f} mm)")
        print(f"  小轮齿根高: {tooth_height['hfm1']:.3f} mm vs 测量 {measured['pin_dedendum']} mm (偏差: {abs(tooth_height['hfm1'] - measured['pin_dedendum']):.3f} mm)")
        print(f"  大轮齿顶高: {tooth_height['ham2']:.3f} mm vs 测量 {measured['gear_addendum']} mm (偏差: {abs(tooth_height['ham2'] - measured['gear_addendum']):.3f} mm)")
        print(f"  大轮齿根高: {tooth_height['hfm2']:.3f} mm vs 测量 {measured['gear_dedendum']} mm (偏差: {abs(tooth_height['hfm2'] - measured['gear_dedendum']):.3f} mm)")
    
    if 'face_angle_pin' in measured:
        print(f"\n锥角对比:")
        print(f"  顶锥角: 小轮 {pin['FACEANGLE']:.2f}° vs {measured['face_angle_pin']}°, 大轮 {gear['FACEANGLE']:.2f}° vs {measured['face_angle_gear']}°")
        print(f"  节锥角: 小轮 {pin['PITCHANGLE']:.2f}° vs {measured['pitch_angle_pin']}°, 大轮 {gear['PITCHANGLE']:.2f}° vs {measured['pitch_angle_gear']}°")
    
    return comparison

# ==================== 主程序 ====================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("准双曲面齿轮参数生成与对比工具")
        print("=" * 40)
        print("用法: python generate_gear.py <齿轮型号> [模式]")
        print("")
        print("模式:")
        print("  (无参数)  生成配置 + 对比表")
        print("  config    仅生成配置")
        print("  compare   仅生成对比表")
        print("")
        print("示例:")
        print("  python generate_gear.py J4-2")
        print("  python generate_gear.py J5-3 config")
        print("  python generate_gear.py J6-2 compare")
        print("")
        print("可用型号: J4-2, J5-3, J6-2")
        sys.exit(1)
    
    gear_id = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'all'
    
    if mode in ['all', 'config']:
        generate_gear_data(gear_id)
    
    if mode in ['all', 'compare']:
        generate_comparison(gear_id)
