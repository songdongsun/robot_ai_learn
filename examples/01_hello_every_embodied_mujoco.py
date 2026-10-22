"""
Every-Embodied 入门示例：六轴机械臂、两指夹爪和随机方块抓取。

阅读主线：XML 创建物理世界 → Python 计算 IK 和轨迹 → data.ctrl 发送关节目标
→ mujoco.mj_step 推进物理仿真。这里没有使用 VLA 或 GraspNet，动作由规则生成。
如果安装了 Ruckig，就使用加加速度受限的轨迹；否则退回平滑插值。
"""

# 让类型注解延迟求值，避免运行时立即解析尚未定义的类型。
from __future__ import annotations

# ------------------------------ 模块一：导入依赖 ------------------------------

# 解析 --headless、--fast 等命令行参数。
import argparse
# 在实时演示时暂停一个仿真时间步，使画面速度接近真实时间。
import time
# 自动生成只保存数据的类所需的 __init__ 等方法。
from dataclasses import dataclass
# Optional 表示参数可以为 None；Tuple 表示固定结构的元组返回值。
from typing import Optional, Tuple

# NumPy 负责向量、矩阵、关节数组和线性代数计算。
import numpy as np

# MuJoCo 是必需依赖，负责加载物理模型并推进仿真。
try:
    import mujoco
except ImportError as exc:
    # 提前给出明确安装提示，不让用户看到难理解的后续导入错误。
    raise SystemExit("未检测到 mujoco，请先安装：pip install mujoco") from exc

# Ruckig 是可选依赖，负责生成满足速度、加速度和加加速度约束的轨迹。
try:
    from ruckig import InputParameter, OutputParameter, Result, Ruckig  # type: ignore

    # 导入成功时，后面的轨迹规划优先使用 Ruckig。
    HAS_RUCKIG = True
except Exception:
    # 导入失败不终止程序，后面自动退回平滑插值轨迹。
    HAS_RUCKIG = False


# ------------------------------ 模块二：定义 MuJoCo 物理世界 ------------------------------

# MJCF 物理模型：定义机器人、方块、地面、执行器和物理参数。
MODEL_XML = r"""
<!-- 根节点：整个模型名称为 every_embodied_ur_demo。 -->
<mujoco model="every_embodied_ur_demo">
  <!-- XML 中角度使用度；子 body 的坐标相对父 body 定义。 -->
  <compiler angle="degree" coordinate="local"/>
  <!-- 每步 0.01 秒，设置重力方向和接触求解迭代次数。 -->
  <option timestep="0.01" gravity="0 0 -9.81" iterations="80" ls_iterations="20"/>

  <!-- 纹理和材质只影响显示效果，不决定机器人如何运动。 -->
  <asset>
    <!-- 创建棋盘格纹理，供地面材质使用。 -->
    <texture name="grid" type="2d" builtin="checker" rgb1="0.94 0.96 1.0" rgb2="0.90 0.93 0.97" width="512" height="512"/>
    <!-- 定义地面、机械臂、夹爪和方块的显示材质。 -->
    <material name="mat_floor" texture="grid" texrepeat="5 5"/>
    <material name="mat_ur" rgba="0.93 0.93 0.95 1"/>
    <material name="mat_gripper" rgba="0.14 0.14 0.18 1"/>
    <material name="mat_cube" rgba="0.97 0.43 0.18 1"/>
  </asset>

  <!-- worldbody 是物理世界，其中 body 可以逐层嵌套形成机器人运动链。 -->
  <worldbody>
    <!-- 从上方向场景照射的灯光。 -->
    <light name="key" pos="0.0 0.0 2.0" dir="0 0 -1"/>
    <!-- 无限平面地面，同时参与接触和碰撞计算。 -->
    <geom name="floor" type="plane" size="2 2 0.1" material="mat_floor"/>

    <!-- 机器人根节点位于世界坐标 z=0.05 米处。 -->
    <body name="robot_base" pos="0 0 0.05">
      <!-- 圆柱形底座，只定义几何外形，没有可动关节。 -->
      <geom name="robot_base_geom" type="cylinder" size="0.09 0.05" rgba="0.16 0.16 0.18 1"/>

      <!-- 以下 body 逐层嵌套，组成从肩部到腕部的串联运动链。 -->
      <body name="shoulder_link" pos="0 0 0.05">
        <!-- 第一关节绕 Z 轴旋转，允许旋转正负 180 度。 -->
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-180 180" damping="2"/>
        <geom name="robot_shoulder_geom" type="capsule" fromto="0 0 0 0 0 0.16" size="0.03" material="mat_ur"/>

        <body name="upper_arm_link" pos="0 0 0.16">
          <!-- 第二关节绕 Y 轴抬升大臂。 -->
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-170 170" damping="2"/>
          <geom name="robot_upper_arm_geom" type="capsule" fromto="0 0 0 0.25 0 0" size="0.026" material="mat_ur"/>

          <body name="forearm_link" pos="0.25 0 0">
            <!-- 第三关节是肘关节，控制前臂弯曲。 -->
            <joint name="elbow" type="hinge" axis="0 1 0" range="-170 170" damping="1.8"/>
            <geom name="robot_forearm_geom" type="capsule" fromto="0 0 0 0.24 0 0" size="0.022" material="mat_ur"/>

            <body name="wrist1_link" pos="0.24 0 0">
              <!-- 第四关节控制第一段腕部。 -->
              <joint name="wrist_1" type="hinge" axis="0 1 0" range="-180 180" damping="1.2"/>
              <geom name="robot_wrist1_geom" type="capsule" fromto="0 0 0 0.12 0 0" size="0.018" material="mat_ur"/>

              <body name="wrist2_link" pos="0.12 0 0">
                <!-- 第五关节绕 Z 轴旋转腕部。 -->
                <joint name="wrist_2" type="hinge" axis="0 0 1" range="-180 180" damping="1.0"/>
                <geom name="robot_wrist2_geom" type="capsule" fromto="0 0 0 0.08 0 0" size="0.016" material="mat_ur"/>

                <body name="wrist3_link" pos="0.08 0 0">
                  <!-- 第六关节完成末端腕部旋转。 -->
                  <joint name="wrist_3" type="hinge" axis="0 1 0" range="-180 180" damping="0.8"/>
                  <geom name="robot_wrist3_geom" type="box" pos="0.04 0 0" size="0.035 0.025 0.02" material="mat_gripper"/>

                  <body name="tool" pos="0.08 0 0">
                    <!-- site 是无质量参考点，Python 用它读取末端位置和姿态。 -->
                    <site name="ee_site" pos="0 0 0" size="0.008" rgba="0 1 0 1"/>
                    <!-- 左右手指分别使用滑动关节，构成两指夹爪。 -->
                    <body name="left_finger" pos="0 0.018 0">
                      <joint name="left_finger_joint" type="slide" axis="0 1 0" range="0 0.03" damping="0.4"/>
                      <geom name="robot_left_finger_geom" type="box" pos="0.02 0.01 0" size="0.02 0.004 0.012" material="mat_gripper"/>
                    </body>
                    <body name="right_finger" pos="0 -0.018 0">
                      <joint name="right_finger_joint" type="slide" axis="0 -1 0" range="0 0.03" damping="0.4"/>
                      <geom name="robot_right_finger_geom" type="box" pos="0.02 -0.01 0" size="0.02 0.004 0.012" material="mat_gripper"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- 待抓方块使用 freejoint，因此具有三维平移和三维旋转自由度。 -->
    <body name="object" pos="0.42 0.0 0.03">
      <freejoint name="object_free"/>
      <geom name="cube" type="box" size="0.02 0.02 0.02" material="mat_cube" friction="1.5 0.08 0.03" solref="0.005 1"/>
    </body>

    <!-- 绿色薄盒子只表示目标放置区域。 -->
    <body name="drop_zone" pos="0.30 -0.22 0.002">
      <geom name="drop_zone_geom" type="box" size="0.06 0.06 0.002" rgba="0.2 0.82 0.32 0.45"/>
    </body>

    <!-- 透明 site 用来在查看器中显示项目标签。 -->
    <body name="banner_anchor" pos="0.12 -0.34 0.22">
      <site name="hello_every_embodied" type="sphere" size="0.001" rgba="0 0 0 0"/>
    </body>
  </worldbody>

  <!-- position 执行器接收目标关节位置，kp 决定跟踪目标的力度。 -->
  <actuator>
    <!-- 执行器顺序很重要：Python 按前 6 个机械臂、后 2 个夹爪写入 data.ctrl。 -->
    <position joint="shoulder_pan" kp="120"/>
    <position joint="shoulder_lift" kp="120"/>
    <position joint="elbow" kp="110"/>
    <position joint="wrist_1" kp="90"/>
    <position joint="wrist_2" kp="80"/>
    <position joint="wrist_3" kp="70"/>
    <position joint="left_finger_joint" kp="150"/>
    <position joint="right_finger_joint" kp="150"/>
  </actuator>
</mujoco>
"""


# ------------------------------ 模块三：名称、常量和索引结构 ------------------------------

# 机械臂六个关节的名称必须与 XML 中的 joint name 完全一致。
ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]
# 两个夹爪滑动关节的 XML 名称。
GRIPPER_JOINTS = ["left_finger_joint", "right_finger_joint"]

# 两个滑动关节的位置目标：数值较大表示张开，较小表示闭合。
GRIPPER_OPEN = np.array([0.028, 0.028], dtype=np.float64)
GRIPPER_CLOSE = np.array([0.003, 0.003], dtype=np.float64)


@dataclass
class KinematicsIndex:
    """集中保存常用对象在 MuJoCo 数组中的索引，避免控制循环反复按名称查找。"""

    # 六个机械臂关节在 data.qpos 中的下标。
    arm_qpos_ids: np.ndarray
    # 六个机械臂自由度在 data.qvel 和雅可比矩阵中的下标。
    arm_dof_ids: np.ndarray
    # 六个机械臂关节在 model 关节表中的编号。
    arm_jnt_ids: np.ndarray
    # 两个夹爪关节在 data.qpos 中的下标。
    gripper_qpos_ids: np.ndarray
    # 方块自由关节在 data.qpos 中的起始位置。
    object_qpos_adr: int
    # 方块自由关节在 data.qvel 中的起始位置。
    object_dof_adr: int
    # 末端 site 在 model.site_* 和 data.site_* 数组中的编号。
    ee_site_id: int


# ------------------------------ 模块四：控制器初始化与索引映射 ------------------------------

class URTrajectoryDemo:
    """组织状态读取、IK、轨迹执行和抓取动作的控制器。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        # model 是基本不变的设计图；data 保存 qpos、qvel、ctrl 等实时状态。
        # 保存 MuJoCo 静态模型引用。
        self.model = model
        # 保存 MuJoCo 动态数据引用。
        self.data = data
        # 提前建立常用对象的数组索引。
        self.idx = self._build_kinematics_index()
        # 绿色目标区域中心上方的放置位置，单位是米。
        self.drop_pos = np.array([0.30, -0.22, 0.03], dtype=np.float64)
        # 创建随机数生成器，用于随机方块位置和朝向。
        self.rng = np.random.default_rng()
        # 记录当前方块是否被教学逻辑绑定到机械臂末端。
        self.carrying = False
        # 方块跟随末端时，相对末端向下偏移 2 厘米。
        self.attach_offset = np.array([0.0, 0.0, -0.02], dtype=np.float64)
        # 方块跟随时使用的四元数，顺序是 w、x、y、z。
        self.attach_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # 统计成功抓取次数。
        self.pick_success = 0
        # 统计总抓取尝试次数。
        self.pick_total = 0

        # 六个关节的预设姿态，所有角度在 Python 中使用弧度。
        # home 是默认收拢姿态。
        self.home = np.array([0.0, -0.7, 1.2, -1.1, 1.57, 0.0], dtype=np.float64)
        # wave_a 和 wave_b 只改变部分腕部角度，来回切换形成挥手。
        self.wave_a = np.array([0.3, -0.6, 1.1, -1.3, 1.2, 0.8], dtype=np.float64)
        self.wave_b = np.array([0.3, -0.6, 1.1, -1.3, 1.2, -0.8], dtype=np.float64)
        # dance_a 和 dance_b 的差异更大，来回切换形成舞蹈动作。
        self.dance_a = np.array([-0.9, -0.8, 1.5, -1.2, 1.2, 0.5], dtype=np.float64)
        self.dance_b = np.array([1.0, -0.9, 1.4, -1.0, 1.6, -0.4], dtype=np.float64)
        # 机器人底座在水平面的坐标，用于计算俯视抓取方向。
        self.base_xy = np.array([0.0, 0.0], dtype=np.float64)

    def _build_kinematics_index(self) -> KinematicsIndex:
        """把 XML 中的对象名称转换为 qpos、qvel 和 site 数组索引。"""

        # 分别收集 model 关节编号、qpos 下标和自由度下标。
        arm_jnt_ids = []
        arm_qpos_ids = []
        arm_dof_ids = []
        # 逐个处理六个机械臂关节名称。
        for name in ARM_JOINTS:
            # 根据 XML 名称查找关节在 MuJoCo model 中的整数编号。
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            # 保存关节编号，后面用于读取关节范围。
            arm_jnt_ids.append(jid)
            # jnt_qposadr 给出该关节的位置状态在 data.qpos 中的起点。
            arm_qpos_ids.append(self.model.jnt_qposadr[jid])
            # jnt_dofadr 给出该关节自由度在 data.qvel 和雅可比矩阵中的起点。
            arm_dof_ids.append(self.model.jnt_dofadr[jid])

        # 收集两个夹爪关节在 data.qpos 中的下标。
        grip_qpos_ids = []
        for name in GRIPPER_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            grip_qpos_ids.append(self.model.jnt_qposadr[jid])

        # 获取方块自由关节的编号，以便直接设置它的位置、姿态和速度。
        object_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        # 获取机械臂末端参考点 ee_site 的编号。
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        # 将列表转成整数数组并封装成 KinematicsIndex 返回。
        return KinematicsIndex(
            arm_qpos_ids=np.array(arm_qpos_ids, dtype=np.int32),
            arm_dof_ids=np.array(arm_dof_ids, dtype=np.int32),
            arm_jnt_ids=np.array(arm_jnt_ids, dtype=np.int32),
            gripper_qpos_ids=np.array(grip_qpos_ids, dtype=np.int32),
            object_qpos_adr=int(self.model.jnt_qposadr[object_jid]),
            object_dof_adr=int(self.model.jnt_dofadr[object_jid]),
            ee_site_id=ee_site_id,
        )

    def print_banner(self) -> None:
        """输出演示名称、当前轨迹后端和键盘命令。"""

        # 打印上下分隔线和标题。
        print("\n" + "=" * 72)
        print("Hello Every-Embodied! | UR + Gripper + Random Block Grasp Demo")
        print("=" * 72)
        # 根据可选依赖是否成功导入，显示实际使用的轨迹规划方式。
        print(f"轨迹规划后端: {'Ruckig (jerk-limited)' if HAS_RUCKIG else 'Quintic fallback'}")
        print("输入说明：")
        print("  1 -> 机械臂打招呼")
        print("  2 -> 机械臂趣味动作")
        print("  3 -> 随机方块抓取与放置（轨迹规划 + IK）")
        print("  4 -> 自动连播模式（适合录屏宣传）")
        print("  q -> 退出")
        print("-" * 72)

    def set_random_block(self, pos: Optional[np.ndarray] = None) -> np.ndarray:
        """重置自由方块的位置、朝向和速度，并重新计算场景状态。"""

        # 没有指定位置时，在机械臂前方的矩形区域随机采样。
        if pos is None:
            # x 是机器人正前方距离。
            x = self.rng.uniform(0.33, 0.50)
            # y 是机器人左右方向位置。
            y = self.rng.uniform(-0.16, 0.16)
            # z 让方块中心位于桌面上方。
            z = 0.03
            # 合并成三维位置向量。
            pos = np.array([x, y, z], dtype=np.float64)
        # 随机生成绕世界 Z 轴旋转的偏航角。
        yaw = float(self.rng.uniform(-0.35, 0.35))
        # 计算半角的余弦和正弦，用于构造四元数。
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        # MuJoCo 四元数顺序为 w、x、y、z；这里只包含绕 Z 轴旋转。
        quat = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
        # qadr 是方块七维 qpos 的起始下标。
        qadr = self.idx.object_qpos_adr
        # 自由关节 qpos 的前三项是 x、y、z。
        self.data.qpos[qadr : qadr + 3] = pos
        # 后四项是表示朝向的四元数。
        self.data.qpos[qadr + 3 : qadr + 7] = quat
        # 自由关节有六维速度：三维线速度和三维角速度，重置时全部清零。
        self.data.qvel[self.idx.object_dof_adr : self.idx.object_dof_adr + 6] = 0.0
        # 新方块尚未被夹爪抓住。
        self.carrying = False
        # qpos 被直接修改后，用 mj_forward 重新计算所有派生位置和接触信息。
        mujoco.mj_forward(self.model, self.data)
        # 返回最终使用的位置，供后续计算抓取点。
        return pos

    def get_ee_pos(self) -> np.ndarray:
        """读取末端 site 的世界坐标，并复制出来避免意外修改 MuJoCo 内存。"""

        return self.data.site_xpos[self.idx.ee_site_id].copy()

    def get_ee_rotmat(self) -> np.ndarray:
        """读取末端 site 的 3×3 世界旋转矩阵。"""

        return self.data.site_xmat[self.idx.ee_site_id].reshape(3, 3).copy()

    # -------------------------- 模块五：末端位姿和旋转数学 --------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """将向量缩放到长度为 1，同时避免除以接近零的长度。"""

        # 计算向量的欧氏长度。
        n = np.linalg.norm(v)
        # 零向量没有确定方向，直接返回副本。
        if n < 1e-8:
            return v.copy()
        # 非零向量除以自身长度得到单位向量。
        return v / n

    def build_grasp_orientation(self, target_pos: np.ndarray) -> np.ndarray:
        """根据目标位置构造从正上方接近方块的末端旋转矩阵。"""

        # 让工具 Z 轴朝下进行俯视抓取，X 轴大致从机器人底座指向方块。
        z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        # 只使用目标在水平面的方向作为工具 X 轴候选方向。
        x_hint = np.array([target_pos[0] - self.base_xy[0], target_pos[1] - self.base_xy[1], 0.0], dtype=np.float64)
        # 目标正好位于底座中心时无法确定水平朝向，使用世界 X 轴兜底。
        if np.linalg.norm(x_hint) < 1e-6:
            x_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        # 将 X 轴候选方向单位化。
        x_axis = self._normalize(x_hint)
        # Z×X 得到与二者垂直的 Y 轴。
        y_axis = self._normalize(np.cross(z_axis, x_axis))
        # 再由 Y×Z 重算 X，保证三根轴彼此正交。
        x_axis = self._normalize(np.cross(y_axis, z_axis))
        # 三根单位轴按列拼成 3×3 旋转矩阵。
        return np.column_stack([x_axis, y_axis, z_axis])

    @staticmethod
    def orientation_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
        # 用三组对应轴的叉乘近似世界坐标系中的小角度姿态误差。
        # 每一列是一根坐标轴；当前轴与目标轴的叉乘表示应该旋转的方向。
        e = 0.5 * (
            np.cross(R_cur[:, 0], R_des[:, 0])
            + np.cross(R_cur[:, 1], R_des[:, 1])
            + np.cross(R_cur[:, 2], R_des[:, 2])
        )
        # 返回三维小角度误差向量，供 IK 与角速度雅可比配合使用。
        return e

    @staticmethod
    def rotmat_to_quat(rotmat: np.ndarray) -> np.ndarray:
        """调用 MuJoCo 工具函数，把 3×3 旋转矩阵转换为 wxyz 四元数。"""

        # 先分配长度为 4 的输出数组。
        quat = np.zeros(4, dtype=np.float64)
        # mju_mat2Quat 接收展平后的 9 个旋转矩阵元素并写入 quat。
        mujoco.mju_mat2Quat(quat, rotmat.reshape(-1))
        return quat

    # -------------------------- 模块六：关节轨迹规划 --------------------------

    @staticmethod
    def _smoothstep(t: np.ndarray) -> np.ndarray:
        """计算首尾速度为零的三次平滑插值比例。"""

        return t * t * (3.0 - 2.0 * t)

    def _plan_quintic(self, q0: np.ndarray, q1: np.ndarray, steps: int = 120) -> np.ndarray:
        """在起点与终点关节角之间生成平滑插值，作为 Ruckig 的备用方案。"""

        # 生成从 0 到 1 的时间比例，共 steps 个采样点。
        t = np.linspace(0.0, 1.0, steps, dtype=np.float64)
        # 变成列向量，使同一个插值比例可广播到六个关节。
        s = self._smoothstep(t)[:, None]
        # 每一行都是六个关节在某个时间点的目标角度。
        return q0[None, :] + (q1 - q0)[None, :] * s

    def _plan_ruckig(self, q0: np.ndarray, q1: np.ndarray) -> Optional[np.ndarray]:
        """在速度、加速度和加加速度限制下逐步生成关节轨迹。"""

        # 没有安装 Ruckig 时返回 None，让上层选择备用规划器。
        if not HAS_RUCKIG:
            return None
        # 自由度数量等于要规划的关节数量，此示例为 6。
        dof = len(q0)
        # 创建在线轨迹生成器，控制周期与 MuJoCo timestep 相同。
        otg = Ruckig(dof, self.model.opt.timestep)
        # inp 保存当前状态、目标状态和运动约束。
        inp = InputParameter(dof)
        # out 保存 Ruckig 每一步计算出的新状态。
        out = OutputParameter(dof)
        # 当前关节位置来自轨迹起点。
        inp.current_position = q0.tolist()
        # 假设规划开始时关节速度均为零。
        inp.current_velocity = [0.0] * dof
        # 假设规划开始时关节加速度均为零。
        inp.current_acceleration = [0.0] * dof
        # 目标关节位置来自轨迹终点。
        inp.target_position = q1.tolist()
        # 要求到达终点时速度为零。
        inp.target_velocity = [0.0] * dof
        # 要求到达终点时加速度为零。
        inp.target_acceleration = [0.0] * dof
        # 每个关节的最大速度限制，单位为弧度每秒。
        inp.max_velocity = [1.2] * dof
        # 每个关节的最大加速度限制。
        inp.max_acceleration = [2.0] * dof
        # 每个关节的最大加加速度限制，用于减少运动突变。
        inp.max_jerk = [8.0] * dof

        # 用列表收集每个控制周期的关节位置。
        traj = []
        # 初始状态表示规划仍在进行。
        result = Result.Working
        # 设置最大循环次数，防止异常情况下无限循环。
        safe_max_steps = 3000
        for _ in range(safe_max_steps):
            # 根据当前输入计算下一个控制周期的状态。
            result = otg.update(inp, out)
            # 保存这一周期的六维目标关节位置。
            traj.append(np.array(out.new_position, dtype=np.float64))
            # 把本次输出作为下一次迭代的当前状态。
            out.pass_to_input(inp)
            # Finished 表示已经到达目标状态。
            if result == Result.Finished:
                break
            # 除 Working 和 Finished 外的状态都视为规划失败。
            if result != Result.Working:
                return None
        # 没生成任何轨迹点时返回失败。
        if not traj:
            return None
        # 将多个六维向量沿行方向堆叠成“时间步数×6”的矩阵。
        return np.vstack(traj)

    def plan_trajectory(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        """优先使用 Ruckig；不可用或规划失败时使用平滑插值。"""

        # 首先尝试带运动学约束的 Ruckig 轨迹。
        traj = self._plan_ruckig(q0, q1)
        # Ruckig 成功时直接使用其结果。
        if traj is not None:
            return traj
        # Ruckig 不可用或失败时，生成固定 140 点的平滑插值。
        return self._plan_quintic(q0, q1, steps=140)

    # -------------------------- 模块七：逆运动学和碰撞筛选 --------------------------

    def solve_ik_pose(
        self,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        max_iters: int = 260,
        w_pos: float = 1.0,
        w_ori: float = 0.35,
    ) -> Optional[np.ndarray]:
        """用阻尼最小二乘迭代求解满足目标位置和姿态的六个关节角。"""

        # IK 会临时修改 data 来试算姿态，结束前必须恢复真实仿真状态。
        # 备份完整位置状态，包括机械臂、夹爪和方块。
        q_backup = self.data.qpos.copy()
        # 备份完整速度状态，避免 IK 试算污染正在运行的仿真。
        qvel_backup = self.data.qvel.copy()
        # 从当前 qpos 取出六个机械臂关节角，作为迭代初值。
        arm_q = self.data.qpos[self.idx.arm_qpos_ids].copy()

        # 最多迭代 max_iters 次；提前达到误差阈值时会直接返回。
        for _ in range(max_iters):
            # 将本轮候选关节角临时写入 MuJoCo 状态。
            self.data.qpos[self.idx.arm_qpos_ids] = arm_q
            # 由候选关节角重新计算末端位置、旋转矩阵等派生量。
            mujoco.mj_forward(self.model, self.data)
            # 位置误差等于目标位置减去当前末端位置。
            err_pos = target_pos - self.get_ee_pos()
            # 默认姿态误差为零，表示只求位置 IK。
            err_ori = np.zeros(3, dtype=np.float64)
            # 调用者给出目标旋转矩阵时，同时计算姿态误差。
            if target_rot is not None:
                err_ori = self.orientation_error(self.get_ee_rotmat(), target_rot)
            # 位置小于 3 毫米，且姿态误差小于 0.05 时认为求解成功。
            if np.linalg.norm(err_pos) < 0.003 and (target_rot is None or np.linalg.norm(err_ori) < 0.05):
                # 恢复进入 IK 前的真实位置状态。
                self.data.qpos[:] = q_backup
                # 恢复进入 IK 前的真实速度状态。
                self.data.qvel[:] = qvel_backup
                # 恢复后重新计算派生量，保持 data 内部一致。
                mujoco.mj_forward(self.model, self.data)
                # 返回求得的六个关节角副本。
                return arm_q.copy()

            # 雅可比矩阵描述“关节小变化”如何引起“末端位置和姿态小变化”。
            # jacp 用于末端位置，行数 3；列数 nv 是模型总自由度数。
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            # jacr 用于末端旋转，尺寸与 jacp 相同。
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            # 计算 ee_site 相对于所有模型自由度的位置和旋转雅可比。
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.idx.ee_site_id)
            # 只选六个机械臂自由度对应的列，忽略夹爪和方块。
            Jp = jacp[:, self.idx.arm_dof_ids]
            # 同样截取六个机械臂自由度对应的旋转雅可比。
            Jr = jacr[:, self.idx.arm_dof_ids]

            # 没有姿态目标时，只使用 3×6 的位置雅可比和三维位置误差。
            if target_rot is None:
                J = Jp
                err = err_pos
            else:
                # 有姿态目标时，把位置和姿态雅可比上下拼成 6×6 矩阵。
                J = np.vstack([w_pos * Jp, w_ori * Jr])
                # 位置和姿态误差也按相同权重拼成六维向量。
                err = np.hstack([w_pos * err_pos, w_ori * err_ori])

            # 阻尼项可降低接近奇异位形时矩阵求解不稳定的问题。
            lam = 1e-3
            # 阻尼最小二乘公式：dq = Jᵀ(JJᵀ + λI)⁻¹e。
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), err)
            # 限制单次迭代最多改变 0.08 弧度，避免跳得过远。
            dq = np.clip(dq, -0.08, 0.08)
            # 用本轮关节增量更新下一轮候选角度。
            arm_q = arm_q + dq

            # 将每次迭代结果限制在 XML 定义的关节活动范围内。
            # enumerate 同时提供六个关节的局部序号和 MuJoCo 关节编号。
            for i, jid in enumerate(self.idx.arm_jnt_ids):
                # 从 model.jnt_range 读取该关节最小角和最大角。
                jmin, jmax = self.model.jnt_range[jid]
                # 超出范围的候选角被截断到边界。
                arm_q[i] = np.clip(arm_q[i], jmin, jmax)

        # 达到最大迭代次数仍未收敛时，同样恢复真实仿真状态。
        self.data.qpos[:] = q_backup
        self.data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.model, self.data)
        # None 告诉上层这个末端目标没有求得可用关节角。
        return None

    def is_robot_collision_low_risk(self, arm_q: np.ndarray) -> bool:
        """试放一组关节角，排除机器人撞地面或深度穿入方块的候选姿态。"""

        # 与 IK 相同，试算前保存真实位置和速度状态。
        q_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()

        # 临时应用待检查的机械臂关节角。
        self.data.qpos[self.idx.arm_qpos_ids] = arm_q
        # 检查候选姿态时保持夹爪张开。
        self.data.qpos[self.idx.gripper_qpos_ids] = GRIPPER_OPEN
        # 重新计算当前姿态下的接触列表 data.contact。
        mujoco.mj_forward(self.model, self.data)

        # 默认认为候选姿态安全，发现危险接触后改为 False。
        ok = True
        # data.ncon 是当前接触数量，逐个检查接触对。
        for i in range(self.data.ncon):
            # 取第 i 个接触记录。
            c = self.data.contact[i]
            # contact 中保存发生接触的两个几何体编号。
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            # 将几何体编号转换回 XML 中的名称，找不到时使用空字符串。
            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
            # 使用集合后无需区分两个几何体在接触记录中的先后顺序。
            pair = {n1, n2}
            # 如果 floor 与任何 robot_ 开头的部件接触，就判定风险较高。
            if "floor" in pair and any(name.startswith("robot_") for name in pair):
                ok = False
                break
            # 机器人可以轻触方块，但不允许穿入深度超过 2 毫米。
            if "cube" in pair and any(name.startswith("robot_") for name in pair) and c.dist < -0.002:
                ok = False
                break

        # 检查完毕后恢复真实状态，避免候选姿态出现在实际仿真中。
        self.data.qpos[:] = q_backup
        self.data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.model, self.data)
        # 返回这个候选关节姿态是否通过筛选。
        return ok

    # -------------------------- 模块八：控制执行和完整动作 --------------------------

    def _set_ctrl(self, q_arm: np.ndarray, q_gripper: np.ndarray) -> None:
        # XML 中前六个执行器控制机械臂，最后两个执行器控制两根夹爪手指。
        # 将六个机械臂目标关节角写入位置执行器。
        self.data.ctrl[:6] = q_arm
        # 将两个夹爪滑动关节目标位置写入位置执行器。
        self.data.ctrl[6:8] = q_gripper

    def _update_attached_object(self) -> None:
        """教学演示中抓取成功后，让方块稳定跟随末端，避免简化夹爪滑落。"""

        # 没有抓住方块时不需要修改方块状态。
        if not self.carrying:
            return
        # 读取当前机械臂末端世界坐标。
        ee = self.get_ee_pos()
        # 取得方块在 qpos 中的起始下标。
        qadr = self.idx.object_qpos_adr
        # 方块位置跟随末端，并应用预设的向下偏移。
        self.data.qpos[qadr : qadr + 3] = ee + self.attach_offset
        # 方块姿态保持为抓取发生时保存的姿态。
        self.data.qpos[qadr + 3 : qadr + 7] = self.attach_quat
        # 清零方块线速度和角速度，防止物理积分产生抖动。
        self.data.qvel[self.idx.object_dof_adr : self.idx.object_dof_adr + 6] = 0.0

    def pin_object_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """直接设定方块位姿并清零速度，用于稳定放置结果。"""

        # 找到方块自由关节在 qpos 中的起始位置。
        qadr = self.idx.object_qpos_adr
        # 写入三维位置。
        self.data.qpos[qadr : qadr + 3] = pos
        # 写入四元数姿态。
        self.data.qpos[qadr + 3 : qadr + 7] = quat
        # 清零三维线速度和三维角速度。
        self.data.qvel[self.idx.object_dof_adr : self.idx.object_dof_adr + 6] = 0.0
        # 直接改状态后立即更新派生数据。
        mujoco.mj_forward(self.model, self.data)

    def settle_object(self, steps: int = 40, viewer=None, realtime: bool = True) -> None:
        # 短暂推进仿真并衰减速度，避免释放瞬间的接触冲量把方块弹飞。
        # 保存当前机械臂姿态，稳定阶段一直保持不动。
        q_hold = self.data.qpos[self.idx.arm_qpos_ids].copy()
        # 取得方块位置状态起点，后面用于检查高度。
        qadr = self.idx.object_qpos_adr
        # 连续运行指定数量的短时间步。
        for _ in range(steps):
            # 保持机械臂姿态并让夹爪维持张开。
            self._set_ctrl(q_hold, GRIPPER_OPEN)
            # 推进一个 0.01 秒物理时间步。
            mujoco.mj_step(self.model, self.data)
            # 在刚释放的几个时间步内压低方块速度。
            self.data.qvel[self.idx.object_dof_adr : self.idx.object_dof_adr + 6] *= 0.2
            # 若数值误差使方块轻微陷入桌面，就把它拉回桌面高度。
            if self.data.qpos[qadr + 2] < 0.021:
                # 0.021 米约等于方块半高加少量离桌间隙。
                self.data.qpos[qadr + 2] = 0.021
                # 清零方块三维线速度，阻止继续向下穿透。
                self.data.qvel[self.idx.object_dof_adr : self.idx.object_dof_adr + 3] = 0.0
            # 有图形窗口时，将最新状态同步到画面。
            if viewer is not None:
                viewer.sync()
            # 实时模式按仿真步长休眠；--fast 模式跳过休眠。
            if realtime:
                time.sleep(self.model.opt.timestep)

    def execute_trajectory(self, q_traj: np.ndarray, gripper: np.ndarray, viewer=None, realtime: bool = True) -> None:
        """逐点写入控制目标；每写一次就让 MuJoCo 向前计算一个时间步。"""

        # q_traj 每一行是某个时间点的六个目标关节角。
        for q in q_traj:
            # 写入机械臂目标和本段轨迹固定的夹爪目标。
            self._set_ctrl(q, gripper)
            # 根据控制目标推进一次物理仿真。
            mujoco.mj_step(self.model, self.data)
            # 如果已抓住方块，更新方块位姿使其跟随末端。
            self._update_attached_object()
            # 同步图形窗口。
            if viewer is not None:
                viewer.sync()
            # 实时运行时等待一个仿真时间步。
            if realtime:
                time.sleep(self.model.opt.timestep)

    def move_arm_to(self, target_q: np.ndarray, gripper: np.ndarray, viewer=None, realtime: bool = True) -> None:
        """从当前关节角规划并执行到目标关节角的轨迹。"""

        # 读取机械臂当前六个关节角作为轨迹起点。
        q0 = self.data.qpos[self.idx.arm_qpos_ids].copy()
        # 从当前角度到目标角度生成完整轨迹。
        traj = self.plan_trajectory(q0, target_q)
        # 执行轨迹，同时保持指定的夹爪开合状态。
        self.execute_trajectory(traj, gripper, viewer=viewer, realtime=realtime)

    def move_to_xyz(
        self,
        target_xyz: np.ndarray,
        gripper: np.ndarray,
        viewer=None,
        realtime: bool = True,
        target_rot: Optional[np.ndarray] = None,
    ) -> bool:
        """先通过 IK 将末端目标转换成关节角，再规划并执行关节轨迹。"""

        # 求解能让末端达到目标位置和可选目标姿态的关节角。
        q_goal = self.solve_ik_pose(target_xyz, target_rot=target_rot)
        # IK 无解时返回 False，由调用者决定回退动作。
        if q_goal is None:
            return False
        # IK 成功后，从当前姿态平滑运动到求得的关节姿态。
        self.move_arm_to(q_goal, gripper, viewer=viewer, realtime=realtime)
        # True 表示目标已成功转换并执行。
        return True

    def sample_reachable_block(self, trials: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        """随机采样方块位置，只接受 IK 可解且通过简单碰撞筛选的位置。"""

        # 最多尝试 trials 个随机候选位置。
        for _ in range(trials):
            # 在机械臂前方采样水平坐标，并将方块放在桌面上。
            x = self.rng.uniform(0.34, 0.48)
            y = self.rng.uniform(-0.14, 0.14)
            z = 0.03
            # 合并为目标三维位置。
            pos = np.array([x, y, z], dtype=np.float64)
            # 根据目标方向构造俯视抓取姿态。
            target_rot = self.build_grasp_orientation(pos)
            # pre 是方块上方 13 厘米的预抓取点。
            pre = pos + np.array([0.0, 0.0, 0.13], dtype=np.float64)
            # grasp 是方块上方 5 厘米的实际抓取点。
            grasp = pos + np.array([0.0, 0.0, 0.05], dtype=np.float64)
            # 分别求解预抓取点和抓取点的 IK。
            q_pre = self.solve_ik_pose(pre, target_rot=target_rot)
            q_grasp = self.solve_ik_pose(grasp, target_rot=target_rot)
            # 任意一个位置没有 IK 解，就放弃这个候选。
            if q_pre is None or q_grasp is None:
                continue
            # 预抓取姿态存在明显碰撞时放弃候选。
            if not self.is_robot_collision_low_risk(q_pre):
                continue
            # 实际抓取姿态存在明显碰撞时也放弃候选。
            if not self.is_robot_collision_low_risk(q_grasp):
                continue
            # 候选通过所有检查后，真正把方块放到该位置。
            self.set_random_block(pos)
            # 返回位置和对应抓取姿态。
            return pos, target_rot

        # 多次随机采样都失败时，使用中央的保底位置。
        fallback = np.array([0.40, 0.0, 0.03], dtype=np.float64)
        # 将方块放到保底位置。
        self.set_random_block(fallback)
        # 返回保底位置及其俯视抓取姿态。
        return fallback, self.build_grasp_orientation(fallback)

    def routine_wave(self, viewer=None, realtime: bool = True) -> None:
        """在两个预设腕部姿态之间往返，形成挥手动作。"""

        # 先移动到第一个挥手姿态。
        self.move_arm_to(self.wave_a, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
        # 重复两次 A→B→A 的摆动。
        for _ in range(2):
            self.move_arm_to(self.wave_b, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
            self.move_arm_to(self.wave_a, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
        # 动作结束后回到默认姿态。
        self.move_arm_to(self.home, GRIPPER_OPEN, viewer=viewer, realtime=realtime)

    def routine_dance(self, viewer=None, realtime: bool = True) -> None:
        """在两个差异较大的预设关节姿态之间往返，形成舞蹈动作。"""

        # 重复两轮舞蹈姿态切换。
        for _ in range(2):
            self.move_arm_to(self.dance_a, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
            self.move_arm_to(self.dance_b, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
        # 舞蹈结束后回到默认姿态。
        self.move_arm_to(self.home, GRIPPER_OPEN, viewer=viewer, realtime=realtime)

    def routine_autoshow(self, viewer=None, realtime: bool = True, rounds: int = 2) -> None:
        """按挥手、舞蹈、随机抓取的顺序自动执行指定轮数。"""

        # i 从 0 开始，因此显示轮数时加 1。
        for i in range(rounds):
            print(f"[AutoShow] Round {i + 1}/{rounds}")
            # 依次调用三个独立动作流程。
            self.routine_wave(viewer=viewer, realtime=realtime)
            self.routine_dance(viewer=viewer, realtime=realtime)
            self.routine_random_pick(viewer=viewer, realtime=realtime)

    def routine_random_pick(self, viewer=None, realtime: bool = True) -> None:
        """依次执行预抓取、下降、闭合、抬升、搬运、释放和撤离。"""

        # 每调用一次就增加总尝试次数。
        self.pick_total += 1
        # 采样一个 IK 可达且碰撞风险较低的方块位置和抓取姿态。
        block, grasp_rot = self.sample_reachable_block()
        # 预抓取点位于方块正上方 13 厘米。
        pre = block + np.array([0.0, 0.0, 0.13], dtype=np.float64)
        # 抓取点位于方块中心上方 5.5 厘米。
        grasp = block + np.array([0.0, 0.0, 0.055], dtype=np.float64)
        # 抓住后先垂直抬升到方块上方 18 厘米。
        lift = block + np.array([0.0, 0.0, 0.18], dtype=np.float64)
        # 放置前先移动到目标区域上方 14 厘米。
        place_pre = self.drop_pos + np.array([0.0, 0.0, 0.14], dtype=np.float64)
        # 最终放置点位于目标中心上方 5.5 厘米。
        place = self.drop_pos + np.array([0.0, 0.0, 0.055], dtype=np.float64)
        # 单独计算面向放置区域的俯视末端姿态。
        place_rot = self.build_grasp_orientation(self.drop_pos)

        # 张开夹爪移动到预抓取点。
        ok = self.move_to_xyz(pre, GRIPPER_OPEN, viewer=viewer, realtime=realtime, target_rot=grasp_rot)
        # 只有上一步成功，才继续向下移动到实际抓取点。
        ok = ok and self.move_to_xyz(grasp, GRIPPER_OPEN, viewer=viewer, realtime=realtime, target_rot=grasp_rot)
        # 任一步 IK 失败都终止本次抓取并返回 home。
        if not ok:
            print("[抓取] IK 失败：目标位置不可达，已返回 home。")
            self.move_arm_to(self.home, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
            return

        # 保持机械臂不动并逐步闭合夹爪；足够接近时才把方块标记为已抓住。
        self.execute_trajectory(
            # 起点和终点关节角相同，所以这段轨迹只用于等待夹爪闭合。
            self._plan_quintic(
                self.data.qpos[self.idx.arm_qpos_ids].copy(),
                self.data.qpos[self.idx.arm_qpos_ids].copy(),
                steps=30,
            ),
            # 整段等待轨迹都发送闭合夹爪目标。
            GRIPPER_CLOSE,
            viewer=viewer,
            realtime=realtime,
        )
        # 计算当前末端位置与方块中心的欧氏距离。
        dist = np.linalg.norm(self.get_ee_pos() - block)
        # 距离小于 7 厘米时，认为这个教学示例抓取成功。
        self.carrying = dist < 0.07
        # 保存抓取姿态对应的四元数，搬运时让方块保持这个姿态。
        self.attach_quat = self.rotmat_to_quat(grasp_rot)
        # 末端距离过大时不绑定方块，直接终止本次搬运。
        if not self.carrying:
            print("[抓取] 夹爪与目标偏差较大，本次跳过搬运。")
            self.move_arm_to(self.home, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
            return

        # 保持夹爪闭合，依次执行抬升、移动到目标上方和下降。
        self.move_to_xyz(lift, GRIPPER_CLOSE, viewer=viewer, realtime=realtime, target_rot=grasp_rot)
        self.move_to_xyz(place_pre, GRIPPER_CLOSE, viewer=viewer, realtime=realtime, target_rot=place_rot)
        self.move_to_xyz(place, GRIPPER_CLOSE, viewer=viewer, realtime=realtime, target_rot=place_rot)

        # 把目标放置旋转矩阵转换成 MuJoCo 使用的四元数。
        place_quat = self.rotmat_to_quat(place_rot)
        # 将方块稳定放到桌面，避免教学用简化夹爪在最后阶段将其弹飞。
        # 复制目标位置，避免直接修改 self.drop_pos。
        placed_pos = self.drop_pos.copy()
        # 设为方块半高加少量间隙，使方块落在桌面上而不穿透。
        placed_pos[2] = 0.021  # cube half-height + tiny clearance
        # 直接固定方块的最终位置和姿态。
        self.pin_object_pose(placed_pos, place_quat)
        # 解除“方块跟随末端”的教学绑定。
        self.carrying = False
        # 推进一小段仿真，让方块稳定接触桌面。
        self.settle_object(steps=35, viewer=viewer, realtime=realtime)

        # 先张开夹爪再向上撤离，减少侧向接触冲量。
        self.execute_trajectory(
            # 机械臂保持不动，只给夹爪 24 个时间步完成张开。
            self._plan_quintic(
                self.data.qpos[self.idx.arm_qpos_ids].copy(),
                self.data.qpos[self.idx.arm_qpos_ids].copy(),
                steps=24,
            ),
            GRIPPER_OPEN,
            viewer=viewer,
            realtime=realtime,
        )
        # 夹爪张开后，垂直撤回放置区域上方。
        self.move_to_xyz(place_pre, GRIPPER_OPEN, viewer=viewer, realtime=realtime, target_rot=place_rot)

        # 完整流程执行到这里，成功次数加 1。
        self.pick_success += 1
        # 计算累计成功率，max 防止理论上的除零。
        rate = 100.0 * self.pick_success / max(1, self.pick_total)
        # 输出成功次数、总次数和百分比。
        print(f"[抓取] 成功放置到目标区域，累计成功率：{self.pick_success}/{self.pick_total} ({rate:.1f}%)")
        # 最后让机械臂张开夹爪并回到默认姿态。
        self.move_arm_to(self.home, GRIPPER_OPEN, viewer=viewer, realtime=realtime)


# ------------------------------ 模块九：命令行参数和交互循环 ------------------------------

def parse_args() -> argparse.Namespace:
    """解析图形界面、运行速度和自动演示相关命令行参数。"""

    # 创建参数解析器，并设置命令行帮助中的程序描述。
    parser = argparse.ArgumentParser(description="UR-style MuJoCo random block grasp demo")
    # --headless 出现时为 True，表示不创建图形窗口。
    parser.add_argument("--headless", action="store_true", help="无图形模式运行")
    # --fast 出现时为 True，表示不按真实时间 sleep。
    parser.add_argument("--fast", action="store_true", help="关闭实时 sleep，加速执行")
    # --autoplay 出现时，进入交互循环前先执行自动连播。
    parser.add_argument("--autoplay", action="store_true", help="启动后先自动连播一次")
    # 指定自动连播轮数；没有指定时默认为两轮。
    parser.add_argument("--autoplay-rounds", type=int, default=2, help="自动连播轮数，默认 2")
    # 解析当前进程收到的命令行参数并返回 Namespace。
    return parser.parse_args()


def run_loop(
    demo: URTrajectoryDemo,
    viewer=None,
    realtime: bool = True,
    autoplay: bool = False,
    autoplay_rounds: int = 2,
) -> None:
    """初始化姿态并根据键盘命令调用不同的动作流程。"""

    # 显示程序名称、轨迹后端和按键说明。
    demo.print_banner()
    # 程序启动后先把机械臂移动到统一的 home 姿态。
    demo.move_arm_to(demo.home, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
    # 指定 --autoplay 时，先执行至少一轮自动演示。
    if autoplay:
        demo.routine_autoshow(viewer=viewer, realtime=realtime, rounds=max(1, autoplay_rounds))
    # 不断读取用户输入，直到收到 q。
    while True:
        # 去除输入首尾空格并转成小写，方便统一判断。
        cmd = input("请输入 [1/2/3/4/q]: ").strip().lower()
        # 输入 1 时执行挥手。
        if cmd == "1":
            demo.routine_wave(viewer=viewer, realtime=realtime)
        # 输入 2 时执行舞蹈。
        elif cmd == "2":
            demo.routine_dance(viewer=viewer, realtime=realtime)
        # 输入 3 时执行一次随机抓取和放置。
        elif cmd == "3":
            demo.routine_random_pick(viewer=viewer, realtime=realtime)
        # 输入 4 时连续执行两轮完整演示。
        elif cmd == "4":
            demo.routine_autoshow(viewer=viewer, realtime=realtime, rounds=2)
        # 输入 q 时跳出无限循环并退出程序。
        elif cmd == "q":
            print("已退出示例，欢迎继续体验 Every-Embodied。")
            break
        # 其他输入不执行动作，只提示合法命令。
        else:
            print("无效输入，请输入 1/2/3/4/q。")


def configure_viewer_branding(viewer) -> None:
    # 设置窗口标题，并显示 XML 中 site 的名称标签。
    # glfw 只用于访问 MuJoCo 查看器底层窗口。
    try:
        import glfw  # type: ignore

        # 把项目名称写到图形窗口标题栏。
        glfw.set_window_title(viewer.window, "hello_every_embodied | Every-Embodied MuJoCo Demo")
    except Exception:
        # 某些平台拿不到底层窗口时跳过标题设置，不影响仿真。
        pass

    # site 标签设置同样不是核心功能，因此单独容错。
    try:
        # 显示 XML 中 ee_site 和 hello_every_embodied 等 site 名称。
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
    except Exception:
        pass


# ------------------------------ 模块十：创建模型并启动程序 ------------------------------

def main() -> None:
    """创建 MuJoCo 模型和控制器，并选择图形或无界面运行方式。"""

    # 读取用户传入的命令行选项。
    args = parse_args()
    # --fast 为 True 时关闭实时等待，否则按物理时间实时播放。
    realtime = not args.fast

    # 1. 编译 XML 得到静态模型；2. 创建保存实时状态的 data。
    # from_xml_string 会检查 XML 并把它编译成高效的 MjModel。
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    # MjData 根据 model 分配 qpos、qvel、ctrl、contact 等运行时数组。
    data = mujoco.MjData(model)
    # 把 model 和 data 交给上面的控制器。
    demo = URTrajectoryDemo(model, data)

    # 无界面模式仍会执行完整物理仿真，只是不创建可视化窗口。
    if args.headless:
        # viewer=None 告诉控制循环不要同步画面。
        run_loop(
            demo,
            viewer=None,
            realtime=realtime,
            autoplay=args.autoplay,
            autoplay_rounds=args.autoplay_rounds,
        )
        # 无界面循环退出后直接结束 main，不再尝试创建窗口。
        return

    # launch_passive 由当前 Python 主循环负责推进仿真和刷新画面。
    try:
        # 延迟导入 viewer，使纯无界面模式不依赖图形窗口。
        from mujoco import viewer as mj_viewer

        # 上下文管理器负责创建窗口，并在退出时自动释放资源。
        with mj_viewer.launch_passive(model, data) as viewer:
            # 设置窗口标题和 site 标签。
            configure_viewer_branding(viewer)
            # 进入带图形同步的键盘交互循环。
            run_loop(
                demo,
                viewer=viewer,
                realtime=realtime,
                autoplay=args.autoplay,
                autoplay_rounds=args.autoplay_rounds,
            )
    except Exception as exc:
        # 图形环境不可用时给出原因，并继续使用无界面模式运行。
        print(f"[提示] viewer 启动失败，切换到 headless 模式：{exc}")
        run_loop(
            demo,
            viewer=None,
            realtime=realtime,
            autoplay=args.autoplay,
            autoplay_rounds=args.autoplay_rounds,
        )


# 只有直接执行本文件时才调用 main；被其他模块 import 时不会自动启动。
if __name__ == "__main__":
    main()
