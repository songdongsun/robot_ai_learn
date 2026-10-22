# MuJoCo、XML 与 Python 控制入门

MuJoCo 是一个机器人和多关节物理系统仿真器。它在电脑中模拟机械臂如何运动、碰撞和抓取物体，也会计算重力、摩擦力、接触力以及关节受到控制后的变化。

在 `examples/01_hello_every_embodied_mujoco.py` 示例中，XML 描述机器人和环境，Python 决定机器人接下来做什么，MuJoCo 根据物理规律计算执行结果。

```text
XML：机器人和环境是什么
Python：机器人接下来做什么
MuJoCo：动作执行后会发生什么
```

# MuJoCo 的主要用途

MuJoCo 常用于机械臂抓取、机器人运动控制、强化学习、模仿学习，以及 ACT、Diffusion Policy 和 VLA 等策略的训练或评估。没有真实机器人时，可以先在 MuJoCo 中检查算法能否运行、动作是否合理以及机器人会不会发生碰撞。

当前示例中，MuJoCo 负责模拟：

- 六轴机械臂和两指夹爪的运动
- 方块受到的重力、接触和摩擦
- 机械臂与物体之间的碰撞
- 关节控制指令产生的运动结果
- 抓取、搬运和放置的完整过程

# XML 物理模型

MuJoCo 使用 MJCF 格式的 XML 描述仿真模型。这里的“模型”是物理模型，不是 YOLO、ACT 或 VLA 那样的神经网络模型。

XML 相当于机器人及其工作环境的设计图，主要描述：

- 机器人有哪些连杆和关节
- 关节的类型、旋转轴和运动范围
- 连杆的形状、尺寸、质量和惯量
- 哪些形状参与碰撞
- 电机如何驱动关节
- 环境中有哪些地面、桌子和物体
- 重力、摩擦和接触等物理参数

例如：

```xml
<body name="upper_arm">
    <joint
        name="shoulder"
        type="hinge"
        axis="0 1 0"
        range="-170 170"
    />
    <geom
        type="capsule"
        size="0.03"
        mass="1.5"
    />
</body>
```

这段 XML 创建一段名为 `upper_arm` 的连杆。它通过 `shoulder` 旋转关节连接，绕 Y 轴转动，活动范围为 `-170°` 到 `170°`。连杆使用胶囊形碰撞体，质量为 `1.5 kg`。

当前示例没有使用单独的 XML 文件，而是把完整 XML 保存为 Python 中的 `MODEL_XML` 字符串。里面包含机械臂、夹爪、方块、地面、绿色放置区域和灯光。

# Python 控制程序

Python 是仿真的控制程序，相当于机器人的大脑和操作员。它读取当前状态，计算目标动作，然后把关节控制指令发送给 MuJoCo。

Python 首先把 XML 交给 MuJoCo：

```python
model = mujoco.MjModel.from_xml_string(MODEL_XML)
data = mujoco.MjData(model)
```

其中：

- `model` 保存由 XML 编译出的静态模型，例如机器人结构、质量、关节范围、碰撞形状和执行器参数。
- `data` 保存不断变化的仿真状态，例如关节位置、关节速度、控制指令和仿真时间。

```python
data.qpos  # 当前关节位置
data.qvel  # 当前关节速度
data.ctrl  # 发给执行器的控制目标
data.time  # 当前仿真时间
```

可以简单记为：

```text
model = 机器人和环境的设计图
data  = 机器人和环境此刻的状态
```

# Python 与 XML 的连接

XML 给关节和末端位置命名，例如：

```xml
<joint name="shoulder_pan"/>
<joint name="elbow"/>
<site name="ee_site"/>
```

Python 根据这些名称查找 MuJoCo 内部的对象编号：

```python
jid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "shoulder_pan",
)
```

找到编号后，Python 才知道哪里保存着肩关节、肘关节、夹爪和末端执行器的数据。XML 中的名称因此构成了物理模型与控制程序之间的接口。

# 一次控制循环

Python 和 MuJoCo 按以下顺序不断合作：

```text
Python 读取 XML 并创建 MuJoCo 模型
                    ↓
Python 读取当前关节和末端状态
                    ↓
Python 计算逆运动学和运动轨迹
                    ↓
Python 将关节目标写入 data.ctrl
                    ↓
MuJoCo 计算动力学、重力、碰撞和摩擦
                    ↓
MuJoCo 将新状态写回 data
                    ↓
Python 读取新状态并计算下一步动作
```

每调用一次下面的函数，物理仿真就向前运行一个时间步：

```python
mujoco.mj_step(model, data)
```

一个简化的控制循环类似这样：

```python
for joint_target in trajectory:
    data.ctrl[:] = joint_target
    mujoco.mj_step(model, data)
```

Python 提供目标，MuJoCo 不会直接把机器人瞬移过去，而是根据电机、关节和物理参数计算机器人实际如何运动。

# 逆运动学与轨迹规划

Python 通常不能直接向 MuJoCo 发送“把夹爪移动到这个空间坐标”的指令，因为真正受到执行器控制的是各个关节。

逆运动学 IK 负责把末端目标位置转换成关节角：

```text
目标夹爪位置 x y z
          ↓ 逆运动学 IK
肩关节角、肘关节角、腕关节角
```

得到目标关节角后，还需要轨迹规划生成一系列平滑的中间状态：

```text
当前角度 0° → 3° → 8° → 16° → ... → 目标角度 90°
```

当前示例优先使用 Ruckig 规划受速度、加速度和加加速度约束的平滑轨迹。如果没有安装 Ruckig，程序会退回五次多项式插值。

# 随机抓取示例的执行过程

运行程序后输入 `3`，Python 大致执行以下流程：

1. 随机生成方块位置。
2. 检查目标是否位于机械臂可达范围。
3. 计算夹爪的预抓取位置、抓取位置和姿态。
4. 使用 IK 计算各关节的目标角度。
5. 使用 Ruckig 生成平滑关节轨迹。
6. 将轨迹中的关节目标逐步写入 `data.ctrl`。
7. MuJoCo 模拟机械臂运动、接触和碰撞。
8. Python 关闭夹爪并抬起方块。
9. 机械臂将方块移动到绿色目标区域。
10. Python 打开夹爪并让机械臂返回初始姿态。

示例中的主要 Python 方法包括：

| 方法 | 作用 |
| --- | --- |
| `_build_kinematics_index()` | 找到 XML 中的关节、夹爪和末端编号 |
| `move_arm_to()` | 将机械臂移动到目标关节角 |
| `move_to_xyz()` | 使用 IK 将夹爪移动到目标空间位置 |
| `execute_trajectory()` | 逐步发送轨迹控制指令 |
| `routine_wave()` | 组织挥手动作 |
| `routine_dance()` | 组织舞蹈动作 |
| `routine_random_pick()` | 完成随机抓取和放置 |
| `mujoco.mj_step()` | 让物理仿真向前计算一步 |

# 物理模型与学习模型的区别

具身智能中经常同时出现两种“模型”：

| 模型类型 | 示例 | 作用 |
| --- | --- | --- |
| 物理模型 | MuJoCo XML、URDF | 描述机器人、环境和物理属性 |
| 学习模型 | YOLO、ACT、Diffusion Policy、VLA | 根据图像、语言和机器人状态预测结果或动作 |

它们可以组成完整系统：

```text
相机图像、语言指令、机器人状态
                 ↓
ACT、Diffusion Policy 或 VLA
                 ↓
生成机器人动作或关节目标
                 ↓
MuJoCo XML 创建的机器人执行动作
                 ↓
MuJoCo 返回新的图像和机器人状态
```

当前示例通过人工规则、IK 和轨迹规划产生动作。以后学习 ACT、Diffusion Policy 或 VLA 时，主要变化的是 Python 产生动作的方式；XML 和 MuJoCo 仍负责机器人、环境和物理反馈。

# 与 ROS 2 和其他仿真器的关系

MuJoCo 是物理仿真器，ROS 2 是机器人软件通信框架，两者解决的问题不同。ROS 2 可以连接感知、规划和控制模块，MuJoCo 可以作为接收控制命令并返回仿真状态的虚拟机器人。

| 工具 | 主要用途 |
| --- | --- |
| MuJoCo | 轻量、快速的机器人动力学和接触仿真 |
| ROS 2 | 机器人感知、规划和控制模块之间的通信 |
| Gazebo | 与 ROS 生态结合较紧密的机器人系统仿真 |
| Isaac Sim 和 Isaac Lab | 高质量视觉仿真和大规模并行训练 |
| ManiSkill | 面向机器人操作和强化学习的任务环境 |

# 当前示例的运行方法

```powershell
conda activate embodied
cd E:\cpp_project\every-embodied
python examples\01_hello_every_embodied_mujoco.py
```

交互命令：

- `1`：机械臂挥手
- `2`：机械臂舞蹈
- `3`：随机抓取并放置方块
- `4`：自动连续演示
- `q`：退出

无界面快速验证：

```powershell
python examples\01_hello_every_embodied_mujoco.py --headless --fast --autoplay --autoplay-rounds 1
```

# 一句话总结

XML 创建虚拟机器人和环境，Python 读取状态并生成控制命令，MuJoCo 根据物理规律执行命令并返回新状态。三者循环协作，组成一个可以观察、控制和训练的机器人仿真系统。
