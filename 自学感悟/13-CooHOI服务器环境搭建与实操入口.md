# CooHOI 服务器环境搭建与实操入口

日期：2026-11-09。

目标：把 05 篇章的 CooHOI 环境搭到服务器，能够读取作者策略、执行小规模训练，并将仿真画面带回本地学习。

## 1. 服务器与目录约定

- SSH：`ssh sdsun@192.168.0.131`。
- 用户要求的工作根目录：`/mnt/sdb-36T/sdsun`。
- 本次项目目录：`/mnt/sdb-36T/sdsun/coohoi_workspace`。
- Conda 环境：`/mnt/sdb-36T/sdsun/miniconda3/envs/coohoi`。
- 系统：Ubuntu 20.04.6 LTS。
- GPU：8 张 NVIDIA RTX A6000，每张约 48 GB 显存。
- 驱动：550.120，`nvidia-smi` 显示支持 CUDA 12.4。

驱动支持的 CUDA 版本不等于项目 PyTorch 携带的 CUDA 运行时版本；本次按项目安装的是 PyTorch CUDA 11.1 构建。

GPU 上存在其他任务，不能把某张卡长期视为空闲。本次短测试使用物理 GPU 1，启动脚本将 GPU 编号作为参数，之后每次运行前需检查负载和使用安排。

没有升级系统驱动，没有停止其他用户任务。项目、日志、录像、下载包均在指定根目录内。后续启动脚本显式设置 XDG、PyTorch 扩展、CUDA、pip、W&B 和临时文件目录。本次自动生成在个人 home 中的 gymtorch 编译缓存已移入工作目录；没有批量搬动或删除已有的个人缓存。

## 2. 安装的核心版本

| 组件 | 版本 |
|---|---|
| Python | 3.8 |
| Isaac Gym | Preview 4 |
| PyTorch | 1.8.1+cu111 |
| torchvision | 0.9.1+cu111 |
| NumPy | 1.23.5 |
| rl-games | 1.1.4 |
| Gym | 0.23.1 |
| Ray | 1.13.0 |
| TensorBoard | 1.15.0 |
| protobuf | 3.20.1 |
| W&B | 0.16.6，运行时使用离线模式 |

完整依赖和环境导出保存在服务器：

```text
coohoi_workspace/requirements-installed.txt
coohoi_workspace/environment-installed.yml
```

这些文件是本机安装状态记录。重新搭建时还需要 Isaac Gym 安装包、驱动与相应系统库，并不能只凭 pip 列表就保证任意机器可运行。

## 3. 源码和资源准备

官方项目：https://github.com/Winston-Gu/CooHOI

使用的源码提交：`3183d992b3725cdd9502419cc1eeee1576930e1c`。

由于服务器直接下载 GitHub 较慢，最终从本地官方仓库传入了运行快照。服务器的 `CooHOI/` 不含 Git 历史，不应在该目录直接按完整 Git 克隆使用。

已包含：

- 项目代码、环境配置和训练配置。
- 人形机器人资产。
- 配置引用的全部 26 个参考动作文件。
- `SingleAgent.pth` 和 `TwoAgent.pth` 两份作者权重。

省略了额外的 `coohoi/data/objects_asset` 家具网格数据。已核对当前单人、双人箱子任务使用 `gym.create_box` 生成物体，不需要该大型目录。以后研究其他物体任务时，再按需要补齐。

源码入口 `coohoi/run.py` 的规范化换行 SHA-256 已与本地核对一致：

```text
ff0394f3e8c0a2e447b0b4a79935a95efffc198795bab166d73f620aab323c31
```

打包过程中曾因工作目录选择错误传入一个无关副本，发现后未用于运行，并已清理本次误传副本和取消的下载文件。

## 4. 为服务器实操做的适配

### 4.1 离屏相机开关

原项目 `BaseTask` 在 headless 模式下默认关闭图形设备。新增了环境配置 `enableCameraSensors` 的读取，同时允许用 `COOHOI_GRAPHICS_DEVICE` 指定图形 GPU。

该修改在 `headless-camera.patch` 中保留，可对照原始代码。没有修改奖励、动作映射或作者策略参数。

### 4.2 多 GPU 编号

本机测试发现，录像时使用 `CUDA_VISIBLE_DEVICES` 屏蔽设备，会导致 CUDA 与 Vulkan 编号不一致并在图形初始化时崩溃。

录像入口保留物理 GPU 编号，同时指定 `--sim_device`、`--rl_device`、当前 PyTorch 设备和图形设备。加载策略权重时先在 CPU 读取，避免权重记录的 `cuda:0` 造成意外分配。

无画面训练与推理检查仍可使用 GPU 屏蔽，将选中的物理卡映射为逻辑卡 0。

### 4.3 NVRTC 库链接

旧版 PyTorch 包含 `libnvrtc-builtins.so`，运行时却查找 `libnvrtc-builtins.so.11.1`。已在本项目 Conda 环境的 `lib/` 下建立指向包内库的符号链接。

这项修改仅影响用户自己的 `coohoi` 环境，不涉及系统 CUDA 或其他环境。

### 4.4 录像入口

`record_policy.py` 复用 CooHOI 的配置、环境、网络构建和动作执行代码，增加相机与视频输出。视频是策略真实推理后的仿真结果，不是预先制作的动画。

双人实现把相邻两个环境槽组成一组，因此需要 `--num_envs 2`。相机显示所属环境的对象，录像用左右双视角显示两个角色，每个视角标记 Actor 编号。

视频分辨率为 640×480，30 帧/秒。默认 300 步对应约 10 秒。这个短视频用于学习观察，不是完整成功率评估。

## 5. 已完成的验证与尚存问题

### 已通过的基础与训练验证

- pip 依赖检查：没有损坏的依赖关系。
- PyTorch 能识别 RTX A6000 并执行 CUDA 张量计算。
- Isaac Gym 与 gymtorch 导入及扩展编译通过。
- 无图形的 GPU PhysX 仿真创建通过。
- 单人、双人作者权重能够加载并执行仿真动作。
- 官方无画面单人推理检查完成一个 119 步短回合，退出码为 0；日志为 `logs/infer-check.log`。该短回合只验证运行链路，不代表任务成功率评估。
- 16 环境的小规模训练完成采样、参数更新和检查点保存，训练命令退出码为 0。

短训练上限参数设为 2，但原版循环实际记录了 epoch 1～3。它只用于验证训练链路，不代表已经训练出可用策略。

生成的检查点示例：

```text
/mnt/sdb-36T/sdsun/coohoi_workspace/results/training-check/
  Humanoid_09-15-03-11/nn/Humanoid.pth
```

训练日志：`logs/train-check.log`。W&B 使用离线模式，日志没有上传到外部服务。

### 尚存的录像退出问题

录像能够完成帧输出并产生可播放的 MP4，但进程退出时，NVIDIA Vulkan 后台线程 `[vkps] Update` 仍会触发段错误，命令可能返回非零状态。

这是未解决的问题。不能因为视频文件存在就把录像命令当作完全正常；应检查日志是否记录完整帧数，并验证文件可解码。

本次没有通过强制退出或忽略错误码掩盖问题，也没有为了排查而升级共享服务器的驱动。排查用的临时虚拟显示进程已停止；尝试替换的本项目 Vulkan 加载库已恢复原版。

它不影响已经验证通过的无图形短训练，但录像入口仍有这个明确的可靠性限制。

已将单人和双人 300 帧、30 FPS 的视频复制到本地 `实战练习代码/.cache/coohoi/`，并检查可解码及实际画面。结束搭建时未留下继续运行的训练、录像或临时虚拟显示进程。

## 6. 以后怎样启动

本地连接服务器：

```powershell
ssh sdsun@192.168.0.131
```

服务器中：

```bash
cd /mnt/sdb-36T/sdsun/coohoi_workspace
nvidia-smi
```

以下 GPU 1 只是示例，每次应选择可用且允许使用的卡。

```bash
# 无画面、作者模型的短回合推理检查
bash tools/run.sh 1 infer-check

# 作者单人模型推理录像，存在上面记录的退出问题
bash tools/run.sh 1 single

# 作者双人模型推理录像，左右双视角
bash tools/run.sh 1 double

# 小规模训练链路检查，不是正式训练
bash tools/run.sh 1 train-check
```

详细使用说明见：[实战入口说明](../实战练习代码/04_CooHOI服务器实战/README.md)。

## 7. 后面的学习顺序

1. 在本地观看作者模型录像，先看清楚任务和行为。
2. 从 `run.sh` 对应到文档命令，区分训练与推理。
3. 阅读策略输入的观察向量，确认它使用哪些状态。
4. 阅读动作映射与 PD 控制，接上关节角、力矩和动力学。
5. 阅读任务奖励、AMP 奖励和结束条件。
6. 亲自跑小规模训练检查，找到日志和检查点。
7. 固定测试条件，比较训练不同阶段的策略表现。
8. 再深入双机器人协作和后续正式训练。

当前目标是理解并实际观察学习与控制链路，不是先把论文规模训练跑满。
