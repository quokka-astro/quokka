#!/usr/bin/env python3
"""
分析 Quokka 模拟结果：
1. 检查测试运行状态
2. 可视化气体密度图和金属丰度图
3. 输出超新星(SN)信息
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def check_test_status():
    """检查测试是否已运行"""
    print("=" * 80)
    print("【第1步】测试运行状态检查")
    print("=" * 80)
    
    build_dir = Path('/Users/meow/quokka/build')
    
    if not build_dir.exists():
        print("❌ 构建目录不存在")
        return False
    
    # 检查测试
    import subprocess
    try:
        result = subprocess.run(['ctest', '--output-on-failure'], 
                               cwd=str(build_dir),
                               capture_output=True, 
                               text=True,
                               timeout=10)
        
        if "passed" in result.stdout or "Passed" in result.stdout:
            print("✅ 测试已运行")
            # 提取测试摘要
            lines = result.stdout.split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print("❓ 测试状态不确定")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  测试仍在运行中")
        return True
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def analyze_sn_info():
    """分析超新星信息"""
    print("\n" + "=" * 80)
    print("【第3步】超新星(SN)信息统计")
    print("=" * 80)
    
    csv_path = Path('/Users/meow/quokka/tests/chuhan_density_sn_plots/sn_feedback_summary.csv')
    
    if not csv_path.exists():
        print(f"❌ SN 反馈总结文件不存在: {csv_path}")
        return
    
    try:
        df = pd.read_csv(csv_path)
        
        print(f"\n📊 模拟时间范围:")
        print(f"  起始时间: {df['time_myr'].min():.4f} Myr")
        print(f"  结束时间: {df['time_myr'].max():.4f} Myr")
        print(f"  总时间步数: {len(df)}")
        
        print(f"\n💣 超新星统计:")
        total_sn = df['n_sn_markers'].sum()
        print(f"  总 SN 事件数: {int(total_sn)}")
        
        # 找到 SN 发生的位置
        sn_rows = df[df['n_sn_markers'] > 0]
        if len(sn_rows) > 0:
            print(f"  发生 SN 的时间步数: {len(sn_rows)}")
            print(f"\n  SN 事件详情:")
            for idx, row in sn_rows.iterrows():
                print(f"    时间: {row['time_myr']:.4f} Myr, SN数: {int(row['n_sn_markers'])}")
        else:
            print(f"  ⚠️  在模拟期间内未检测到 SN 事件")
        
        print(f"\n📍 粒子信息:")
        print(f"  总粒子检测数: {df['n_particles_seen'].sum()}")
        print(f"  有粒子数据的时间步: {df[df['has_particle_payload'] > 0].shape[0]}")
        
        # 化学元素丰度信息
        print(f"\n⚗️  化学元素丰度信息:")
        print(f"  Scalar 0 总和: {df['scalar_0_sum'].sum():.6e}")
        print(f"  Scalar 1 总和: {df['scalar_1_sum'].sum():.6e}")
        print(f"  Scalar 2 总和: {df['scalar_2_sum'].sum():.6e}")
        
    except Exception as e:
        print(f"❌ 读取 SN 反馈文件出错: {e}")


def visualize_density():
    """可视化气体密度图"""
    print("\n" + "=" * 80)
    print("【第2步】可视化气体密度图和金属丰度图")
    print("=" * 80)
    
    density_dir = Path('/Users/meow/quokka/tests/chuhan_density_sn_plots')
    metallicity_dir = Path('/Users/meow/quokka/tests/chuhan_metallicity_plots')
    
    # 气体密度图
    if density_dir.exists():
        density_files = sorted(glob.glob(str(density_dir / 'density_ts*.png')))
        if density_files:
            print(f"\n📈 气体密度图:")
            print(f"  已生成 {len(density_files)} 张密度图")
            print(f"  保存位置: {density_dir}")
            for i, f in enumerate(density_files[:3]):
                print(f"    {i+1}. {Path(f).name}")
            if len(density_files) > 3:
                print(f"    ... 以及 {len(density_files)-3} 张")
        else:
            print(f"❌ 未找到密度图文件")
    else:
        print(f"❌ 密度图目录不存在: {density_dir}")
    
    # 金属丰度（标量）图
    if metallicity_dir.exists():
        scalar_files = sorted(glob.glob(str(metallicity_dir / 'scalar_*.png')))
        if scalar_files:
            print(f"\n📊 金属丰度(Scalar)图:")
            print(f"  已生成 {len(scalar_files)} 张标量图")
            print(f"  保存位置: {metallicity_dir}")
            
            # 按 scalar 分类统计
            for scalar_id in range(3):
                files = [f for f in scalar_files if f'scalar_{scalar_id}' in f]
                if files:
                    print(f"    Scalar {scalar_id}: {len(files)} 张图")
        else:
            print(f"❌ 未找到金属丰度图文件")
    else:
        print(f"❌ 金属丰度图目录不存在: {metallicity_dir}")
    
    # 尝试打开图像查看器（如果可用）
    print(f"\n💡 提示: 可以在以下位置查看生成的图像:")
    if density_dir.exists():
        print(f"  - 密度图: {density_dir}")
    if metallicity_dir.exists():
        print(f"  - 金属丰度图: {metallicity_dir}")


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Quokka 模拟结果分析工具".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # 运行分析
    check_test_status()
    visualize_density()
    analyze_sn_info()
    
    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
