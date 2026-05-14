#!/usr/bin/env python3
"""
Quokka ChuhanProblem 模拟结果完整分析和可视化脚本
包括: SN 信息统计、密度分布可视化、化学丰度演化
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
from datetime import datetime

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 10)

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f"  {title}".center(80))
    print("=" * 80)

def analyze_sn_feedback():
    """详细分析 SN 反馈信息"""
    print_header("【第1部分】超新星(SN)反馈信息详细分析")
    
    csv_path = Path('/Users/meow/quokka/tests/chuhan_density_sn_plots/sn_feedback_summary.csv')
    
    if not csv_path.exists():
        print(f"❌ SN 反馈文件不存在: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # 基本信息
    print(f"\n📊 模拟基本信息:")
    print(f"  数据点数量: {len(df)}")
    print(f"  时间范围: {df['time_myr'].min():.6f} - {df['time_myr'].max():.6f} Myr")
    print(f"  总模拟时长: {df['time_myr'].max() - df['time_myr'].min():.6f} Myr")
    
    # SN 统计
    print(f"\n💣 超新星事件统计:")
    total_sn = int(df['n_sn_markers'].sum())
    print(f"  总 SN 事件数: {total_sn}")
    
    sn_events = df[df['n_sn_markers'] > 0]
    if len(sn_events) > 0:
        print(f"  ✅ 检测到 SN 事件！")
        print(f"  首次 SN 出现时间: {sn_events.iloc[0]['time_myr']:.6f} Myr")
        print(f"  最后一次 SN 出现时间: {sn_events.iloc[-1]['time_myr']:.6f} Myr")
        print(f"  SN 事件分布:")
        for idx, row in sn_events.iterrows():
            print(f"    时间步 {idx}: {row['time_myr']:8.6f} Myr, SN数: {int(row['n_sn_markers']):5d} events")
    else:
        print(f"  ⚠️  在模拟期间未检测到 SN 事件")
        print(f"  → 建议: 增加 max_timesteps 或降低环境密度以加速恒星演化")
    
    # 粒子信息
    print(f"\n🌟 粒子信息:")
    print(f"  总粒子检测数: {int(df['n_particles_seen'].sum())}")
    print(f"  有粒子数据的时间步: {len(df[df['has_particle_payload'] > 0])}")
    
    # 化学丰度
    print(f"\n⚗️  化学元素丰度统计:")
    print(f"  Scalar 0 总和: {df['scalar_0_sum'].sum():.6e}")
    print(f"  Scalar 1 总和: {df['scalar_1_sum'].sum():.6e}")  
    print(f"  Scalar 2 总和: {df['scalar_2_sum'].sum():.6e}")
    print(f"  Scalar 0 最大: {df['scalar_0_sum'].max():.6e}")
    print(f"  Scalar 1 最大: {df['scalar_1_sum'].max():.6e}")
    print(f"  Scalar 2 最大: {df['scalar_2_sum'].max():.6e}")
    
    return df

def create_sn_plots(df):
    """创建 SN 相关的图表"""
    print_header("【第2部分】超新星事件可视化")
    
    if df is None or len(df) == 0:
        print("❌ 数据不可用")
        return
    
    # 创建多子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ChuhanProblem: 超新星反馈和化学演化分析', fontsize=16, fontweight='bold')
    
    # 1. SN 事件数随时间变化
    ax1 = axes[0, 0]
    ax1.bar(df['time_myr'], df['n_sn_markers'], width=0.02, color='red', alpha=0.7)
    ax1.set_xlabel('Time (Myr)', fontsize=12)
    ax1.set_ylabel('Number of SN Events', fontsize=12)
    ax1.set_title('SN Events vs Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 累积 SN 数
    ax2 = axes[0, 1]
    cumsum_sn = df['n_sn_markers'].cumsum()
    ax2.plot(df['time_myr'], cumsum_sn, 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Time (Myr)', fontsize=12)
    ax2.set_ylabel('Cumulative SN Events', fontsize=12)
    ax2.set_title('Cumulative SN Events', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 化学标量演化
    ax3 = axes[1, 0]
    ax3.plot(df['time_myr'], df['scalar_0_sum'], 'b-o', label='Scalar 0', linewidth=2, markersize=4)
    ax3.plot(df['time_myr'], df['scalar_1_sum'], 'g-s', label='Scalar 1', linewidth=2, markersize=4)
    ax3.plot(df['time_myr'], df['scalar_2_sum'], 'm-^', label='Scalar 2', linewidth=2, markersize=4)
    ax3.set_xlabel('Time (Myr)', fontsize=12)
    ax3.set_ylabel('Scalar Sum', fontsize=12)
    ax3.set_title('Chemical Abundance Evolution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. 粒子信息
    ax4 = axes[1, 1]
    ax4.plot(df['time_myr'], df['n_particles_seen'], 'c-d', linewidth=2, markersize=4)
    ax4.set_xlabel('Time (Myr)', fontsize=12)
    ax4.set_ylabel('Number of Particles Detected', fontsize=12)
    ax4.set_title('Particle Detection', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = '/Users/meow/quokka/tests/sn_analysis_plots.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ SN 分析图表已保存: {output_file}")
    plt.close()

def list_visualization_files():
    """列出所有可用的可视化文件"""
    print_header("【第3部分】已生成的可视化文件")
    
    dirs_info = {
        '气体密度图': '/Users/meow/quokka/tests/chuhan_density_sn_plots',
        '化学元素丰度图': '/Users/meow/quokka/tests/chuhan_metallicity_plots'
    }
    
    for name, path in dirs_info.items():
        path_obj = Path(path)
        if path_obj.exists():
            png_files = list(path_obj.glob('*.png'))
            print(f"\n📊 {name}:")
            print(f"  路径: {path}")
            print(f"  文件数: {len(png_files)}")
            if png_files:
                print(f"  样本文件:")
                for f in sorted(png_files)[:5]:
                    print(f"    - {f.name}")
                if len(png_files) > 5:
                    print(f"    ... 和 {len(png_files) - 5} 个其他文件")
        else:
            print(f"\n❌ {name}目录不存在: {path}")

def generate_summary_report(df):
    """生成总结报告"""
    print_header("【第4部分】模拟总结报告")
    
    if df is None or len(df) == 0:
        print("❌ 数据不可用")
        return
    
    total_sn = int(df['n_sn_markers'].sum())
    sn_detected = total_sn > 0
    
    print(f"\n📋 关键指标汇总:")
    print(f"  模拟时长: {df['time_myr'].max():.4f} Myr")
    print(f"  超新星事件: {'✅ 已检测' if sn_detected else '❌ 未检测'} ({total_sn} 个事件)")
    print(f"  时间步数: {len(df)}")
    print(f"  平均每步时间: {(df['time_myr'].max() - df['time_myr'].min()) / (len(df) - 1):.4f} Myr")
    
    print(f"\n✨ 关键发现:")
    if sn_detected:
        first_sn_time = df[df['n_sn_markers'] > 0].iloc[0]['time_myr']
        print(f"  首次超新星爆发时间: {first_sn_time:.4f} Myr")
        print(f"  这表明恒星已经达到演化末期，触发了超新星反馈")
    else:
        print(f"  尚未检测到超新星事件")
        print(f"  恒星演化可能尚未达到超新星阶段")
        print(f"  或需要调整环境参数以加速演化")
    
    # 化学反馈状态
    has_chemical_feedback = (df['scalar_0_sum'].sum() + df['scalar_1_sum'].sum() + df['scalar_2_sum'].sum()) > 0
    print(f"  化学反馈状态: {'✅ 已激活' if has_chemical_feedback else '⚠️  未激活'}")
    
    print(f"\n💡 建议:")
    if not sn_detected and df['time_myr'].max() < 50:
        print(f"  • 模拟时长较短，建议进一步增加 max_timesteps")
        print(f"  • 考虑增加初始恒星质量以加速演化")
    if sn_detected:
        print(f"  • 已观察到超新星反馈，结果有效！")
        print(f"  • 可继续进行长期模拟观察恒星种族合成的影响")

def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "Quokka ChuhanProblem 模拟结果完整分析".center(78) + "║")
    print("║" + f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # 分析 SN 反馈
    df = analyze_sn_feedback()
    
    # 创建图表
    if df is not None:
        create_sn_plots(df)
    
    # 列出可视化文件
    list_visualization_files()
    
    # 生成报告
    if df is not None:
        generate_summary_report(df)
    
    print("\n" + "=" * 80)
    print("✅ 分析完成！所有结果已生成".center(80))
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
