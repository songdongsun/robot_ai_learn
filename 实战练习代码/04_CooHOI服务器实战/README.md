# CooHOI 服务器实战入口

服务器：`sdsun@192.168.0.131`。工作目录：`/mnt/sdb-36T/sdsun/coohoi_workspace`。

使用独立 Conda 环境 `coohoi`，不与服务器上的其他训练环境混用。密码不写入脚本或记录。

Conda 环境位于 `/mnt/sdb-36T/sdsun/miniconda3/envs/coohoi`。启动脚本将项目缓存、临时文件、编译缓存和离线日志定向到 `coohoi_workspace` 下。不会修改其他用户目录、训练任务或系统驱动。

**当前限制：录像可完成并输出可播放的 MP4，但这台服务器的 NVIDIA Vulkan 后台线程会在退出时触发段错误，录像命令可能返回非零状态。这个问题尚未解决，不要把该返回码简单忽略，也不要把文件存在当作每次录像完整的证明。无图形的短训练已正常退出。**详见自学感悟中的环境搭建记录。

## 目录

```text
coohoi_workspace/
├── CooHOI/       官方项目运行快照、动作数据、作者模型
├── isaacgym/     Isaac Gym Preview 4
├── tools/        本目录中的启动与录像脚本，以及检查用训练配置
├── downloads/    安装包和传输文件
├── logs/         安装及运行日志
└── results/      仿真录像、摘要与训练检查结果
```

这里的脚本本地保留用于阅读与备份，实际运行需要放在服务器的 `tools/` 目录。原项目需要的小修改与验证结果另见环境搭建记录。

源码来自官方提交 `3183d992b3725cdd9502419cc1eeee1576930e1c`。服务器使用运行快照，不含 Git 历史；保留了全部代码、机器人资产、配置引用的 26 个动作和两份作者权重。省略的 `coohoi/data/objects_asset` 是额外家具网格数据，当前两个箱子任务使用 `gym.create_box`，不读取该目录。后续学习其他物体任务时再补齐。

## 第一次学习：看作者模型推理

在 Windows 终端连接：

```powershell
ssh sdsun@192.168.0.131
```

进入服务器后先看负载：

```bash
nvidia-smi
cd /mnt/sdb-36T/sdsun/coohoi_workspace
```

下面的 `1` 是 GPU 编号示例，不代表长期分配给你的空闲卡。服务器有其他任务，运行前应选择自己可以使用且资源足够的卡。

```bash
bash tools/run.sh 1 single
```

默认执行作者单机器人策略 300 步并录制视频。每次运行默认覆盖 `results/single.mp4` 和对应 JSON；需要保留旧结果时可指定不同名字：

```bash
bash tools/run.sh 1 single --steps 600 --video /mnt/sdb-36T/sdsun/coohoi_workspace/results/single_600.mp4
```

这不是训练。它加载 `SingleAgent.pth`，执行“观察 → 策略 → 动作 → 仿真”的循环。相机用于给人看结果，图像没有输入策略。

## 在本地观看

视频生成后，在本地另开 Windows 终端执行：

```powershell
New-Item -ItemType Directory -Force 'E:\cpp_project\robot_ai_learn\实战练习代码\.cache\coohoi'
scp sdsun@192.168.0.131:/mnt/sdb-36T/sdsun/coohoi_workspace/results/single.mp4 'E:\cpp_project\robot_ai_learn\实战练习代码\.cache\coohoi\single.mp4'
```

然后用本地播放器打开视频，可以暂停、慢放和反复对照代码。JSON 中的奖励均值只是运行摘要，不等同于成功率，也不能凭一段短视频就判定策略足够可靠。

## 第二步：看双机器人协作

```bash
bash tools/run.sh 1 double
```

加载作者的 `TwoAgent.pth`，输出 `results/double.mp4`。下载方式相同，将文件名换成 `double.mp4`。

这个项目的双人实现将两个相邻环境槽配成一组，所以脚本使用 `--num_envs 2`。不能把它改成 1 后仍称为双机器人演示。相机只呈现所属环境的对象，因此录像以左右双视角分别显示 Actor 1 和 Actor 2，不是两个人同时出现在同一个相机画面。

观察单人与双人的动作差别，再回到任务代码分析观察与奖励的变化。

## 第三步：理解训练流程

先检查不带图形的作者模型推理：

```bash
bash tools/run.sh 1 infer-check
```

这个入口使用官方测试流程执行一个短回合，不调用录像相机。

```bash
bash tools/run.sh 1 train-check
```

这个入口只使用 16 个环境进行短训练，用于检查采样、网络更新和模型保存是否可用。上限参数设置为 2，原版循环退出条件下实际记录了 epoch 1～3。它不会训练出可靠的搬运策略，也不是论文规模复现。

W&B 设为离线模式，日志保存在服务器本地，不需要登录外部账户或上传实验数据。检查点在 `results/training-check/` 下。

正式训练前，先读懂观察、动作映射、PD 控制、奖励和结束条件；再根据显存和任务负载调整并行环境数、采样量和批量大小。

## 一步一步的学习安排

1. 看单机器人视频，指出接近、抬起、搬运和失败等实际行为。
2. 沿动作路径读代码：Actor 输出如何变成关节目标，再经 PD 驱动运动。
3. 看策略的观察向量，明确它读取哪些状态，是否使用图像。
4. 对照任务奖励与 AMP 奖励，理解每部分鼓励什么行为。
5. 运行小规模训练检查，认识日志、检查点与推理的区别。
6. 在可用算力下进行正式实验，固定测试条件比较不同检查点的表现。
7. 再阅读双机器人配置和协作反馈机制。
