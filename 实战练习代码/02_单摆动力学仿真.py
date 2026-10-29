"""对应《05综合实战与仿真》第二项：无电机驱动的带阻尼单摆。

运行：python 02_单摆动力学仿真.py
仅保存图片、不打开窗口：python 02_单摆动力学仿真.py --no-show
角度从竖直向下起算；模型为无质量刚杆加杆端质点。
"""

import argparse  # 读取是否显示图形窗口的命令行选项。
from pathlib import Path  # 将结果保存到脚本所在目录。

import matplotlib.pyplot as plt  # 绘制角度曲线和相图。
import numpy as np  # 提供三角函数和采样时间数组。
from scipy.integrate import odeint  # 按动力学方程积分，计算运动过程。


# ==================== 模块1：描述当前状态如何变化 ====================
def pendulum_dynamics(y, t, g, L, m, b):
    """输入状态 [角度, 角速度]，返回变化率 [角速度, 角加速度]。"""
    theta, omega = y  # y 是状态数组，不是空间中的 Y 坐标。
    # 重力力矩为 -mgL sin(theta)，阻尼力矩为 -b*omega，惯量为 mL²。
    acceleration = -(g / L) * np.sin(theta) - b * omega / (m * L**2)
    return [omega, acceleration]  # 求解器需要导数，不是下一时刻的状态。


# ==================== 模块2：直接运行脚本时设置参数 ====================
if __name__ == "__main__":  # 被其他脚本导入时不自动弹出图形。
    parser = argparse.ArgumentParser(description="带阻尼单摆动力学仿真")  # 定义运行选项。
    parser.add_argument("--no-show", action="store_true", help="仅保存结果图片，不弹窗")
    args = parser.parse_args()  # 读取用户传入的选项。

    g = 9.81  # 重力加速度，单位 m/s²。
    L = 1.0  # 杆长，单位 m。
    m = 1.0  # 杆端小球质量，单位 kg。
    b = 0.5  # 转动阻尼系数，单位 N·m·s/rad；可改成 0 观察无阻尼运动。
    theta_0 = np.radians(179)  # 接近顶部，但偏离不稳定平衡点，计算时使用弧度。
    omega_0 = 0.0  # 松手时的初始角速度，单位 rad/s。
    y0 = [theta_0, omega_0]  # 初始状态，不包含外加电机力矩或 PID。
    t = np.linspace(0, 10, 200)  # 请求 0～10 秒的 200 个输出点，不是内部积分步数。

    # ==================== 模块3：积分得到每个时刻的状态 ====================
    solution = odeint(pendulum_dynamics, y0, t, args=(g, L, m, b))  # 反复计算导数并积分。
    theta_traj = solution[:, 0]  # 第一列为所有采样时刻的角度，单位 rad。
    omega_traj = solution[:, 1]  # 第二列为所有采样时刻的角速度，单位 rad/s。
    # 势能以最低点为零；有正阻尼时机械能应随时间降低。
    energy = 0.5 * m * L**2 * omega_traj**2 + m * g * L * (1 - np.cos(theta_traj))
    print(f"仿真时长：{t[-1]:.1f} s，输出点数：{len(t)}")  # 报告本轮设置。
    print(f"初始角度：{np.degrees(theta_0):.1f}°，初始角速度：{omega_0:.4f} rad/s")
    print(f"最终角度：{np.degrees(theta_traj[-1]):.4f}°，最终角速度：{omega_traj[-1]:.4f} rad/s")
    print(f"机械能：{energy[0]:.6f} J → {energy[-1]:.6f} J")  # 10 秒后不一定已经完全停止。

    # ==================== 模块4：绘图并保存到脚本旁 ====================
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))  # 上方时域图，下方相图。
    axes[0].plot(t, theta_traj, color="tab:blue")  # 角度随时间变化，不是空间轨迹。
    axes[0].set(title="Damped pendulum: angle over time", xlabel="Time (s)", ylabel="Angle (rad)")
    axes[0].grid(True)  # 网格便于读取时间和角度。
    axes[1].plot(theta_traj, omega_traj, color="tab:red")  # 相图横轴角度、纵轴角速度。
    axes[1].scatter(theta_traj[0], omega_traj[0], color="green", label="Start", zorder=3)
    axes[1].scatter(theta_traj[-1], omega_traj[-1], color="black", label="End at 10 s", zorder=3)
    axes[1].set(title="Phase portrait (not the spatial path)", xlabel="Angle (rad)", ylabel="Angular velocity (rad/s)")
    axes[1].grid(True)  # 接近原点表示角度、角速度都接近零。
    axes[1].legend()  # 标出本轮仿真的起点和终点。
    fig.tight_layout()  # 调整子图间距，避免标签重叠。
    output_path = Path(__file__).resolve().with_name("02_单摆动力学仿真结果.png")  # 与脚本同目录。
    fig.savefig(output_path, dpi=150)  # 窗口关闭后仍可查看保存的结果。
    print(f"结果图片：{output_path}")  # 输出保存位置。
    if not args.no_show:  # 默认显示窗口，使用 --no-show 可只计算和保存。
        plt.show()  # 保持图形窗口，直到用户关闭。
    plt.close(fig)  # 释放本轮图形资源。
