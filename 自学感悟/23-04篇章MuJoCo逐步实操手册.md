# 04 篇章 MuJoCo 逐步实操手册

建立日期：2026-11-19。

学习范围：[04mujoco复现ACT、Pi0、SmolVLA](../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA/README.md)。路线讨论见 [22-实操路线学习记录](22-MuJoCo实操路线与ACT、SmolVLA、Pi0学习记录.md)。

**使用方式：一次只完成一个阶段。每一步都包括操作、观察结果和理解目标；遇到报错先记录完整错误，不跳过去继续跑。**

本手册的命令和修改点已对照本地源码检查，但环境安装、GUI、数据采集和训练还没有执行验证。后续实际完成一阶段，再更新完成记录。第一轮不要点击所有 Notebook 的“运行全部”。

## 0. 全程地图与当前进度

```text
独立环境＋场景资源
         ↓
纯键盘遥操作：先让机械臂动起来
         ↓
采集一条自己的示范
         ↓
检查图片、状态、动作 → 回放
         ↓
ACT：读一批数据 → 前向 → loss → 反向 → 参数更新
         ↓
保存模型 → 模型接管机械臂 → 记录成功和失败
         ↓
增加红杯／蓝杯语言任务，重新采集匹配数据
         ↓
SmolVLA：加载预训练策略 → 微调 → 仿真执行
         ↓
补充 π0 原理 → 检查预训练加载与配置 → 实操
```

| 阶段 | 当前状态 | 要留下的证据 |
|---|---|---|
| 本地环境与路径 | 待执行 | Python 路径、包版本、依赖检查输出 |
| 纯遥操作 | 待执行 | 窗口正常、键盘能控制机械臂 |
| 示范采集 | 待执行 | 自己的数据目录、回合数、帧数 |
| 样本查看与回放 | 待执行 | 一帧图片和数值说明、完整回放 |
| ACT 训练 | 待执行 | 实际配置、loss、检查点 |
| ACT 仿真部署 | 待执行 | 多次任务结果、失败原因 |
| SmolVLA 实操 | 待执行 | 语言配对数据、加载记录、训练与执行结果 |
| π0 实操 | 待执行 | 原理记录、配置核对、训练或推理结果 |

不跳过 ACT；它用于建立基础闭环。SmolVLA 用来对比加入语言和更换动作生成模型后的变化。π0 先补原理再操作。

## 1. 第一次只做环境和纯遥操作

### 1.1 已检查到的本地环境

当前 Conda 位于 `D:\itsoft\conda`。原来的 `embodied` 环境是 Python 3.11.16、MuJoCo 3.16.0、PyTorch 2.8.0+cpu；本教程 README 指定 Python 3.10、MuJoCo 3.1.6，并依赖一个较早的 LeRobot 提交。

因此新建 `mujoco04` 环境，保留原来能用的学习环境。

在 **PowerShell** 中依次运行以下命令，每条完成后再执行下一条：

```powershell
& 'D:\itsoft\conda\Scripts\conda.exe' create -n mujoco04 python=3.10 -y
```

```powershell
& 'D:\itsoft\conda\envs\mujoco04\python.exe' -m pip install 'numpy==1.26.4' 'mujoco==3.1.6' pyautogui matplotlib scipy pillow 'opencv-python-headless==4.11.0.86' jupyterlab ipykernel ipywidgets
```

```powershell
& 'D:\itsoft\conda\envs\mujoco04\python.exe' -m pip check
```

```powershell
& 'D:\itsoft\conda\envs\mujoco04\python.exe' -c "import sys, mujoco, glfw, cv2, pyautogui; print(sys.executable); print('MuJoCo:', mujoco.__version__)"
```

这些是第一阶段所需依赖；暂时没有安装 LeRobot、下载模型或启动训练。包安装若因网络或版本问题失败，保存错误信息，不要转而把所有包安装到 base。

**通过标准：** Python 路径包含 `envs\mujoco04`，MuJoCo 是 `3.1.6`，导入成功，`pip check` 没有冲突。

### 1.2 建立独立练习副本

原教程留作阅读。练习代码、资源、数据和输出放在自己的实战目录，避免覆盖教程已有的 `demo_data`。

PowerShell：

```powershell
$lessonSource = 'E:\cpp_project\robot_ai_learn\06-策略抓取或抓取VLA\大模型控制、VLA、VLM\04mujoco复现ACT、Pi0、SmolVLA'
$lessonWork = 'E:\cpp_project\robot_ai_learn\实战练习代码\06_MuJoCo_ACT_VLA'
if (Test-Path -LiteralPath $lessonWork) {
    throw '练习目录已存在，请先检查内容，不要重复覆盖。'
}
New-Item -ItemType Directory -Path $lessonWork | Out-Null
Copy-Item -LiteralPath (Join-Path $lessonSource 'asset') -Destination $lessonWork -Recurse
Copy-Item -LiteralPath (Join-Path $lessonSource 'mujoco_env') -Destination $lessonWork -Recurse
Get-ChildItem -LiteralPath $lessonSource -File |
    Where-Object { $_.Extension -in '.ipynb', '.py', '.yaml', '.txt' } |
    Copy-Item -Destination $lessonWork
Copy-Item -LiteralPath (Join-Path $lessonSource 'media') -Destination $lessonWork -Recurse
```

这会复制资源，可能需要一些时间和磁盘空间。没有复制已有示范，后面创建自己的数据。

为了避免 MuJoCo 资源加载遇到中文路径，再创建指向练习目录的英文目录联接：

```powershell
$lessonLink = 'E:\cpp_project\mujoco04_lab'
if (Test-Path -LiteralPath $lessonLink) {
    throw '英文入口已存在，请先检查它指向哪里。'
}
New-Item -ItemType Junction -Path $lessonLink -Target $lessonWork | Out-Null
Set-Location -LiteralPath $lessonLink
```

`mujoco04_lab` 是同一份练习文件的英文入口，不是另一份副本。此处只创建联接，没有删除或移动文件。

解压盘子资源：

```powershell
if (-not (Test-Path -LiteralPath '.\asset\objaverse\plate_11\model_new.xml')) {
    Expand-Archive -LiteralPath '.\asset\objaverse\plate_11.zip' -DestinationPath '.\asset\objaverse'
}
Test-Path -LiteralPath '.\asset\objaverse\plate_11\model_new.xml'
```

**通过标准：** 最后一条输出 `True`，并且 `asset/example_scene_y.xml` 存在。

### 1.3 启动 Notebook

```powershell
Set-Location -LiteralPath 'E:\cpp_project\mujoco04_lab'
& 'D:\itsoft\conda\envs\mujoco04\python.exe' -m ipykernel install --user --name mujoco04 --display-name 'MuJoCo04 Python3.10'
& 'D:\itsoft\conda\envs\mujoco04\python.exe' -m jupyterlab .
```

打开浏览器中的 JupyterLab，新建 Notebook，选择 `MuJoCo04 Python3.10` 内核，保存为 `00_先玩键盘遥操作.ipynb`。

先运行这个检查单元格：

```python
import os
import sys
import mujoco

os.chdir(r"E:\cpp_project\mujoco04_lab")
print("Python:", sys.executable)
print("工作目录:", os.getcwd())
print("MuJoCo:", mujoco.__version__)
```

第一轮不导入 `env_config`；我们已经明确设置英文工作路径，避免它另外创建家目录链接并切换目录。以后运行练习副本中的 Notebook，也先执行这个路径单元格，并跳过其中的 `import env_config`。

### 1.4 只打开环境，不采集、不训练

下一个单元格：

```python
from mujoco_env.y_env import SimpleEnv

PnPEnv = SimpleEnv(
    './asset/example_scene_y.xml',
    seed=0,
    action_type='eef_pose',
    state_type='joint_angle',
)
```

再运行：

```python
try:
    while PnPEnv.env.is_viewer_alive():
        PnPEnv.step_env()
        if PnPEnv.env.loop_every(HZ=20):
            command, reset = PnPEnv.teleop_robot()
            if reset:
                PnPEnv.reset(seed=0)
                continue
            PnPEnv.step(command)
            PnPEnv.render(teleop=True)
finally:
    PnPEnv.env.close_viewer()
```

运行循环后，把焦点放在仿真窗口上：

| 按键 | 观察什么 |
|---|---|
| W/A/S/D | 末端在水平面移动 |
| R/F | 末端升高或降低 |
| Q/E、方向键 | 末端朝向变化 |
| 空格 | 夹爪开合 |
| Z | 恢复固定初始场景 |

第一次只练习移动和开合，不要求立刻抓取成功。关闭仿真窗口退出循环；若单元格不返回，再中断内核执行。

**理解目标：** 键盘给末端运动要求，环境通过逆运动学等处理形成关节控制命令；此时没有神经网络参与。

**通过标准：** 能用键盘控制机械臂，并说明哪个按键影响位置、朝向、夹爪。

## 2. 采集前安装数据处理依赖

只有阶段 1 通过后，再安装 PyTorch 和旧版 LeRobot。以下仍在本地 PowerShell 执行：

```powershell
& 'D:\itsoft\conda\envs\mujoco04\python.exe' -m pip install 'torch==2.6.0' 'torchvision==0.21.0' 'torchaudio==2.6.0' --index-url https://download.pytorch.org/whl/cpu
```

```powershell
& 'D:\itsoft\conda\envs\mujoco04\python.exe' -m pip install 'lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git@10b7b3532543b4adfb65760f02a49b4c537afde7' 'torch==2.6.0' 'torchvision==0.21.0' 'numpy==1.26.4' 'datasets==3.4.1' 'transformers==4.50.3' 'safetensors==0.5.3' 'huggingface-hub<1.0' 'opencv-python-headless==4.11.0.86'
```

```powershell
& 'D:\itsoft\conda\envs\mujoco04\python.exe' -m pip check
& 'D:\itsoft\conda\envs\mujoco04\python.exe' -c "import torch; from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; from lerobot.common.policies.act.modeling_act import ACTPolicy; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('LeRobot dataset/ACT imports OK')"
```

本地 CPU 环境输出 `CUDA: False` 正常。此安装方案依据项目固定版本和该提交的依赖声明制定，仍需在这台电脑实际完成解析和导入验证；没有把未测试的安装写成已成功。

安装后重启 Notebook 内核，重新执行路径单元格。训练模型可能进一步下载视觉编码器权重，不能把导入成功当作所有模型资源均已就绪。

官方依赖依据：[固定提交的 pyproject.toml](https://github.com/huggingface/lerobot/blob/10b7b3532543b4adfb65760f02a49b4c537afde7/pyproject.toml)。

## 3. 采集自己的第一条示范

打开练习副本的 `1.collect_data.ipynb`。按单元格顺序执行，但先完成下面的调整。

### 3.1 统一数据目录和数据集标识

替换参数单元格：

```python
SEED = 0
REPO_NAME = 'sdsun_omy_pnp'
NUM_DEMO = 1
ROOT = './demo_data_sdsun_omy_v1'
TASK_NAME = 'Put mug cup on the plate'
```

这里的 `repo_id` 是本地数据标识，不代表自动发布到 Hugging Face。训练与回放时保持相同标识和路径。

建数据集单元格中，将原来的“询问是否删除已有目录”整段替换为：

```python
from pathlib import Path

if Path(ROOT).exists():
    raise FileExistsError(f'{ROOT} 已存在；请改用 v2 等新目录，不要删除已有示范。')
create_new = True
```

保留后面的 `LeRobotDataset.create(...)`，初次在 Windows 上将 `image_writer_processes=5` 改成 `0`，`image_writer_threads` 改成 `4`，减少 Notebook 多进程问题。

### 3.2 修正动作标签来源

本地环境实现中：

```text
step(command)
    ↓
更新 compute_q 等目标命令
    ↓
返回 get_joint_state()
    ↓
读取当前仿真关节状态
```

所以不能直接把返回值称为新的目标关节动作。为了学习“根据当前观察预测下一步目标”，基础 OMY 采集循环里的：

```python
joint_q = PnPEnv.step(action)
```

替换为：

```python
PnPEnv.step(action)
target_action = np.concatenate([
    np.asarray(PnPEnv.compute_q, dtype=np.float32),
    np.asarray([action[-1]], dtype=np.float32),
])
assert target_action.shape == (7,)
```

随后 `dataset.add_frame` 中使用：

```text
"observation.state": ee_pose,
"action": target_action,
```

**图片和 `ee_pose` 仍在这次 `step(action)` 之前获取。** 数据配对成为：当前观察 → 这次命令对应的目标动作。

这个修改针对六关节 OMY 和当前 `eef_pose` 控制路径。不要直接复制到 xarm7 等其他分支。

代码依据：[step](E:/cpp_project/robot_ai_learn/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA/mujoco_env/y_env.py:505)、[get_joint_state](E:/cpp_project/robot_ai_learn/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA/mujoco_env/y_env.py:700)。

### 3.3 修正开始记录和回合结束的逻辑

将：

```text
if not record_flag and sum(action) != 0:
```

改成：

```text
if not record_flag and np.any(np.abs(action) > 1e-6):
```

因为各维动作求和可能互相抵消，不能用总和判断“是否有动作”。

成功保存的分支改为：

```python
if done and record_flag:
    dataset.save_episode()
    episode_id += 1
    record_flag = False
    if episode_id >= NUM_DEMO:
        break
    PnPEnv.reset(seed=SEED)
    continue
```

键盘重置分支在清空缓存、设置 `record_flag=False` 后加 `continue`，避免旧动作落入新回合。

暂时跳过 Notebook 最后的 `shutil.rmtree(dataset.root / 'images')` 清理单元格。先确认保存和重新加载成功，再决定是否需要清理临时图片。

### 3.4 完成任务并检查保存结果

按以下顺序操作：靠近杯子 → 调整夹爪 → 夹住 → 抬高 → 移到盘子上方 → 降低 → 松开 → 抬起末端。

成功判断以当前 `check_success()` 代码为准，包含杯盘距离、末端高度和夹爪开放状态，不只检查视觉上“杯子似乎在盘子附近”。不通过时检查姿态和距离，不先放宽成功阈值。

保存成功后，在新单元格运行：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

saved = LeRobotDataset('sdsun_omy_pnp', root='./demo_data_sdsun_omy_v1')
print('回合数:', saved.num_episodes)
print('帧数:', saved.num_frames)
print('采样频率:', saved.fps)
```

**通过标准：** 能重新加载，至少保存一条完整回合，帧数大于零。

## 4. 亲眼看一个样本，再回放

### 4.1 打印数据，而不是只看视频

```python
import matplotlib.pyplot as plt

frame_index = saved.num_frames // 2
sample = saved[frame_index]
print('样本索引:', frame_index)
print('指令:', sample['task'])
print('状态:', sample['observation.state'])
print('动作:', sample['action'])
print('主相机形状:', sample['observation.image'].shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, key in zip(axes, ['observation.image', 'observation.wrist_image']):
    axis.imshow(sample[key].permute(1, 2, 0).cpu().numpy())
    axis.set_title(key)
    axis.axis('off')
plt.show()
```

这里未设置动作时间窗口，因此单样本动作预期是 `(7,)`；图像加载为张量后通常是 `(3, 256, 256)`。训练设置动作块后，形状会变化。

需要自己能说清楚：当前夹爪在哪、任务是什么、state 是末端位姿还是关节角、action 的前六维和最后一维分别是什么。

### 4.2 回放

打开练习副本 `2.visualize_data-omy.ipynb`，把数据加载改成：

```python
dataset = LeRobotDataset('sdsun_omy_pnp', root='./demo_data_sdsun_omy_v1')
```

第一回合使用 `episode_index=0`，Windows 数据加载器先用 `num_workers=0`。环境保持 `action_type='joint_angle'`，按保存的 `obj_init` 重建物体初始位置。

运行到动作回放即可，先不运行后面的统计量重写等辅助单元格。

**通过标准：** 回放轨迹与示范对应，杯子没有因为错误的初始位置或动作单位而明显偏离。

**理解目标：回放没有模型做决策，它只是执行保存的动作。回放成功与训练成功是两件事。**

## 5. ACT：先看懂一批数据，再正式训练

### 5.1 修改训练入口

打开练习副本 `3.train.ipynb`：

- 所有数据集标识换成 `sdsun_omy_pnp`，数据路径换成 `./demo_data_sdsun_omy_v1`。
- 本地阅读和小规模检查使用 `device = torch.device('cpu')`。
- 首次检查 `batch_size=2`、`num_workers=0`、`drop_last=False`。
- `chunk_size=10`、`n_action_steps=10`，保留移除腕部相机的操作，便于先按教程单相机路径学习。
- 跳过注释掉的旧训练循环、数据改写与后面的经验性长篇说明，先跟踪实际运行代码。

**原 Notebook 的 MAE 评估不能直接当作可信验证结果。** 它可能在打乱的 batch 之间复用 `select_action()` 的动作队列，导致预测与目标错配；默认又使用训练数据评估。第一轮先用下面的基础训练循环，独立的评估和闭环成功率另做。

### 5.2 理解张量形状

查看一个 batch：

```python
batch = next(iter(dataloader))
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(key, tuple(value.shape), value.dtype)
```

使用 batch size 2、动作块长度 10 时，重点看：

```text
主相机图片： [2, 3, 256, 256]
机器人状态： [2, 6]
目标动作块： [2, 10, 7]
               │   │  └── 每步的动作维度
               │   └───── 每个样本连续的动作步数
               └───────── 一批中的样本数
```

回合末尾不够 10 步时需要填充并配合掩码，不能把填充当作额外真实示范。

### 5.3 第一轮只检查少量参数更新

在完成数据加载器、policy 和 optimizer 的初始化后，以此替换原训练循环单元格。不要同时执行原循环：

```python
training_steps = 5  # 本地先检查流程，不用于宣称已学会任务
assert len(dataloader) > 0, '数据加载器为空，请检查样本数和 batch_size'
step = 0
loss_history = []
policy.train()

while step < training_steps:
    for batch in dataloader:
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = policy.forward(inputs)
        if not torch.isfinite(loss):
            raise RuntimeError(f'损失异常: {loss.item()}')
        loss.backward()
        optimizer.step()
        step += 1
        loss_history.append(loss.item())
        print(f'step={step}, loss={loss.item():.6f}')
        if step >= training_steps:
            break
```

CPU 单步可能较慢，先观察首步耗时，再决定是否把剩余检查转到服务器。

要对应前面的理论：

```text
batch：图片＋状态＋示范动作块
                 ↓
policy.forward：预测，并计算训练损失
                 ↓
loss.backward：求梯度
                 ↓
optimizer.step：真正改变参数
```

后续讲 ACT 时，进一步拆开动作重建损失和 CVAE 的 KL 项。不能把 ACT 说成 SmolVLA 的流匹配，也不能把它的训练直接等同于 OpenVLA 的动作 token 交叉熵。

### 5.4 保存检查点

```python
from pathlib import Path

checkpoint_dir = Path('./ckpt/act_sdsun_omy_smoke')
if checkpoint_dir.exists():
    raise FileExistsError('检查点目录已存在，请换一个实验名')
policy.save_pretrained(checkpoint_dir)
print(checkpoint_dir.resolve())
```

这是少量更新的检查点，用于验证保存与加载，不等于训练完成。正式训练使用独立目录，例如 `act_sdsun_omy_run01`。

**通过标准：** batch 形状正确，loss 有限，优化器更新成功，可以保存模型。

## 6. 正式训练转到服务器

本阶段在数据检查通过后单独带着执行。不要现在就启动教程的长训练。

### 6.1 先检查资源，再确定训练参数

```powershell
ssh sdsun@192.168.0.131
```

服务器终端中执行只读检查：

```bash
pwd
df -h /mnt/sdb-36T/sdsun
nvidia-smi
```

拟使用工作目录为 `/mnt/sdb-36T/sdsun/mujoco04_workspace`。环境、数据、检查点、模型缓存和日志都放在用户目录下，不在系统 Python 或别人的目录里安装。

GPU 占用低不等于已获准占用。到这一步结合实际共享情况选择可用卡，不终止其他人的进程，也不默认使用 GPU 0。

服务器环境需单独核对驱动与 CUDA 版 PyTorch，不能照搬本地 CPU 安装命令，也不复用此前 CooHOI 的旧 PyTorch 环境。

### 6.2 正式运行前核对清单

| 项目 | 核对内容 |
|---|---|
| 数据 | 复制完整数据和元数据，服务器能重新加载 |
| 软件 | 与采集版本匹配，ACT 导入与一个 batch 前向通过 |
| GPU | 已选可用卡，仅让本次进程看到选定设备 |
| 输入 | 主图像、六维末端位姿、七维关节目标动作 |
| 训练步数 | 先做短检查，再考虑教程的 5000 步，不承诺固定耗时 |
| batch size | 根据实际显存决定，不能盲抄 64 |
| 输出目录 | 新实验独立目录，保存配置与损失记录 |
| 验证 | 有多条示范时按完整回合分训练与验证，不随机拆同轨迹相邻帧充当独立验证 |

只有一条示范时，第一轮可用于固定场景的流程验证与拟合观察，不能宣称模型学会了泛化。需要更多示范时，先回到采集阶段扩充变化。

服务器安装命令、实际 GPU 编号和启动命令在本阶段开始时根据实时检查补入，不在手册中虚构已验证的服务器状态。

## 7. ACT 部署：第一次让模型接管机械臂

将正式训练的模型目录完整带回本地，保持配套数据统计和配置一致。

在练习副本 `4.deploy.ipynb` 中：

1. 数据路径和标识指向自己的 OMY 数据。
2. 本地设备改成 `device='cpu'`，实测每步耗时。
3. `ACTPolicy.from_pretrained(...)` 指向自己的检查点。
4. 保持基础 ACT 的输入状态为 `get_ee_pose()`，不要换成六个关节角。
5. 保持 `action_type='joint_angle'`。
6. 每次新回合调用 `policy.reset()`，并复位环境。

推理调用使用：

```python
with torch.inference_mode():
    action = policy.select_action(data)
action = action[0].cpu().numpy()
PnPEnv.step(action)
```

原部署代码启用 `chunk_size=10`、`n_action_steps=1` 和时间集成。理解这三个参数后再改，不能直接把“块长度”理解成“必须一次执行的步数”。

给每回合设置有限的动作步数，例如先观察 600 次决策内的表现；到上限记录失败并退出，避免一直重复动作。600 次在 20 Hz 的理想控制节奏下约对应 30 秒，但 CPU 推理时实际墙钟时间可能更长。

首次用固定 `seed=0` 检查链路，再用新的初始布局评估。至少记录：

| 回合 | 初始条件 | 结果 | 现象 |
|---|---|---|---|
| 1 | 固定布局 | 待填 | 是否靠近正确位置、是否夹住、是否松开 |
| 2 | 固定布局重复 | 待填 | 行为是否稳定 |
| 3 | 新布局 | 待填 | 是否偏离、是否仍完成任务 |

固定场景成功只说明在该条件下成功。失败时按图片与状态 → 模型动作 → 动作接口 → 环境执行的顺序排查，不立刻把原因归结为模型不够大。

## 8. 语言任务：为 SmolVLA 准备正确数据

打开练习副本 `5.language_env.ipynb`，使用独立数据标识和目录，例如：

```python
REPO_NAME = 'sdsun_omy_language'
ROOT = './demo_data_sdsun_language_v1'
```

先录红杯、蓝杯各一条验证流程；再扩充示范，避免所有数据都来自一个任务。教程的 20 条是示例数量，不是保证成功的标准。

**这里状态含义与基础 ACT 阶段不同：语言示例及其部署使用六个关节角作为状态，不是末端位姿。**

调整采集顺序：在 `PnPEnv.step(action)` 前获取主图、腕图和 `PnPEnv.get_joint_state()[:6]`；调用 `step` 后使用目标关节角与夹爪命令构造动作标签。

```python
# 与图像同一时刻的当前关节状态
current_state = PnPEnv.get_joint_state()[:6].copy()

# 原有图像采集、resize 仍在这里完成，然后处理本次遥操作命令
PnPEnv.step(action)
target_action = np.concatenate([
    np.asarray(PnPEnv.compute_q, dtype=np.float32),
    np.asarray([action[-1]], dtype=np.float32),
])
```

保存时使用 `observation.state=current_state`、`action=target_action`、`task=PnPEnv.instruction`。回合保存、重置和开始记录的修正沿用阶段 3。

在 `6.visualize_data.ipynb` 中切换为自己的标识和路径，检查每条指令和实际抓取对象一致。不能只看动作回放成功，却忽略标注写“红杯”、示范抓“蓝杯”。

## 9. SmolVLA：与 ACT 对照完成训练和执行

### 9.1 先明确本次加载的是什么

区分三条路径：

| 路径 | 学习目的 |
|---|---|
| 加载作者已经在本任务训练好的模型 | 检查环境、输入与动作接口 |
| 加载预训练 SmolVLA，再用自己的示范微调 | 完成自己的训练闭环 |
| 只创建 SmolVLA 配置后构造模型 | 不等于加载了完整预训练策略，须继续核对内部初始化 |

当前 `smolvla_omy.yaml` 没有显式写出完整策略的 `pretrained_path`。不能只看到 `type: smolvla` 就称它为完整策略微调。

依据：[固定版本策略工厂](https://github.com/huggingface/lerobot/blob/10b7b3532543b4adfb65760f02a49b4c537afde7/lerobot/common/policies/factory.py)。后续正式操作时查看实际加载日志，确认模型、配置、数据形状一致。

### 9.2 需要改哪些地方？

在独立练习副本中，复制 `smolvla_omy.yaml` 为自己的实验配置，修改：

- 数据集标识与根目录。
- 预训练策略路径及其与旧版 LeRobot 的兼容性。
- 输出目录为本次独立实验目录。
- 根据可用 GPU 设置 batch size、训练步数和设备。
- 首次关闭 W&B：`wandb.enable: false`，去掉示例用户占位值。
- 重复的 `save_checkpoint` 键仅保留一份。

目录中真实配置名是 `smolvla_omy.yaml`，README 和 Notebook 引用的 `smolvla_datawhale_eai.yaml` 不能原样使用。

在 `8.smolvla.ipynb` 中，把包含 `SmolVLAPolicy.from_pretrained(...)` 的 Markdown 单元格改成代码单元格。它才会真正创建 `policy`。

先检查：

```text
图片＋当前关节状态＋指令
              ↓
预处理后的模型输入
              ↓
SmolVLA 预测动作块
              ↓
从动作块取出动作
              ↓
七维关节目标与夹爪命令
              ↓
环境执行
```

加载作者模型只用于验证其匹配的原始任务接口，不能保证与我们修正后的新数据和动作统计完全一致。自己的模型部署使用自己训练得到的配置与统计。

**通过标准：** 能解释预训练从哪里来、自己的示范更新了什么、保存了什么，以及当前执行的是哪一个检查点。

## 10. π0：先补原理，再运行 Notebook

当前没有系统学过 π0。开始这阶段先回答：

1. π0 的图像、语言和状态怎样进入模型？
2. 它怎样生成动作块，与 SmolVLA 相同和不同的地方是什么？
3. 完整预训练策略、视觉语言骨干和随机初始化动作模块有何区别？
4. 本项目使用 LeRobot 的 π0 实现，与之前下载的 OpenPI 仓库是什么关系？

随后核对 `pi0_omy.yaml` 和 `7.pi0.ipynb`。真实配置名与 README 中命令不同；预训练路径、权重访问、版本兼容、动作维度和计算资源都需要逐项验证。

不要把 ACT 的显存用量或训练耗时直接套给 π0，也不要把只成功下载仓库当成已能推理。

先跑匹配模型的推理检查，再决定微调规模；沿用“输入检查 → 短训练 → 保存加载 → 闭环评估”的顺序。

## 11. 常见现象与排查入口

| 现象 | 优先检查 |
|---|---|
| 找不到 `lerobot.common` | 当前 Python 是否是本篇环境、LeRobot 是否为指定版本 |
| 找不到 XML 或 mesh | 工作目录、英文入口、盘子解压目录 |
| Notebook 能导入但窗口不动 | 是否运行控制循环、窗口焦点、是否有多个未关闭 Viewer |
| 数据目录已存在 | 换实验目录，不删除已有示范 |
| 没有保存回合 | 是否真的触发当前成功条件、是否已经开始记录 |
| 保存了但加载失败 | 图片写入、元数据和数据文件是否齐全，完整保留错误信息 |
| CPU 上报 CUDA 错误 | 代码中的设备字段是否仍写死 `cuda` |
| 数据加载器为空 | batch size、drop_last、样本数量 |
| loss 下降但抓取失败 | 动作标签时序、初始布局、输入含义、动作尺度和闭环偏差 |
| state 都是六维但执行不对 | 六个关节角与六维末端位姿不能混用 |
| 换回合后行为奇怪 | 是否重置 policy 的动作队列和环境 |
| SmolVLA 提示 `policy` 未定义 | 模型加载单元格是否仍是 Markdown |

## 12. 逐步完成记录

每完成一个阶段，再填这一表，不凭预计结果提前打勾。

| 日期 | 阶段 | 实际执行内容 | 输出／错误 | 下一步 |
|---|---|---|---|---|
| 2026-11-19 | 准备 | 已检查原环境与本地源码，建立手册 | 安装、窗口、采集与训练尚未验证 | 先执行 1.1，检查独立环境 |

最终要能自己解释：**数据由谁产生、输入是什么、目标动作是什么、训练更新了什么、检查点在哪里、部署执行了什么，以及失败发生在哪一层。**
