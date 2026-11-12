# OpenVLA 复现与动作 Token 学习记录

记录日期：2026-11-12。

对应教程：[02openvla复现.md](../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/02OpenVLA复现/02openvla复现.md)。

前置记录：

- [15-06篇章VLA入门与视觉编码器学习记录.md](15-06篇章VLA入门与视觉编码器学习记录.md)
- [17-SmolVLA仿真推理代码逐步教学记录.md](17-SmolVLA仿真推理代码逐步教学记录.md)
- [18-SmolVLA训练数据配对与学习目标记录.md](18-SmolVLA训练数据配对与学习目标记录.md)

本篇记录进入“02 OpenVLA 复现”后的学习内容。本次讨论以**原版 OpenVLA 7B**为对象，不混入 OpenVLA-OFT 的改进方案。当前完成的是文档阅读和官方源码核对，没有在本次学习中安装 OpenVLA 环境、下载其模型权重或运行评估。

本机行号链接适用于支持文件跳转的客户端；其他阅读器可打开上面的相对路径，再按章节名称定位。后续修改文档时，行号可能变化。

## 1. 这一篇主要做什么？

主线是：

```text
下载已有模型
      ↓
在 LIBERO 中做推理评估
      ↓
理解 OpenVLA 怎样把图像和语言变成动作
```

原教程前面的实操主要是**使用已经微调好的模型**，没有给出完整的训练流程。

因此，读到下载检查点、运行 `run_libero_eval.py` 时，不应把这些步骤理解为已经训练了一个新模型。

先沿着文档读懂每一步的作用，再深入模型内部。

## 2. 哪些 SmolVLA 知识可以直接接着用？

前面已经学过的执行流程仍然适用：

```text
准备模型和环境
       ↓
获取当前观察与任务指令
       ↓
模型生成动作
       ↓
环境控制器执行
       ↓
获得新的观察
       ↓
继续执行，最后统计是否成功
```

换成 OpenVLA 后，环境执行、控制器、动力学这些知识仍然适用。主要变化发生在模型内部，尤其是动作怎样生成。

| 对比点 | 前面学的 SmolVLA | 本篇原版 OpenVLA |
|---|---|---|
| 已讨论的模型输入 | 图像、语言、机器人状态 | 官方标准推理链路使用图像和语言 |
| 动作生成方式 | 流匹配生成连续动作块 | 自回归生成动作 token，再解码 |
| 主要训练目标 | 预测流匹配变化率 | 预测示范动作对应的 token |
| 执行方式 | 动作交给控制器和环境 | 同样交给控制器和环境 |

原版官方推理函数中，处理器接收的是 `prompt` 和 `image`，没有把机器人状态向量作为额外的模型输入。[官方推理代码](https://github.com/openvla/openvla/blob/main/experiments/robot/openvla_utils.py#L115)

**模型没有显式输入机器人状态，不代表底层控制器不需要机器人状态。**

```text
策略模型：根据图像和语言决定动作。
底层控制器：结合机器人当前实际状态，让机器人执行动作要求。
```

也不能据此推断所有 OpenVLA 衍生版本都不接收状态。这里限定为本篇原版的标准推理路径。

## 3. 文档第 2 节：为什么下载两个“仓库”？

[打开“环境搭建与模型准备”](E:/cpp_project/robot_ai_learn/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/02OpenVLA复现/02openvla复现.md:7)。

原文出现两条命令：

```bash
git clone https://huggingface.co/openvla/openvla-7b
```

```bash
git clone https://github.com/openvla/openvla.git
```

虽然都使用 `git clone`，下载内容的职责不同：

| 下载内容 | 主要作用 |
|---|---|
| Hugging Face 模型仓库 | 提供训练好的权重、配置和处理器等模型文件 |
| GitHub 代码仓库 | 提供加载、训练、推理和评估程序 |

联系前面下载的 LeRobot：

```text
代码：
    定义模型怎样构建、怎样计算、怎样运行。

权重：
    保存模型训练后学到的参数。
```

**有代码，不等于模型已经学会动作；有权重，也需要相应代码才能加载使用。**

这些命令是原教程内容的解释，不代表本次已经执行。真正下载模型时还需要确认大文件实际下载完整，而不是只得到权重文件的 Git LFS 指针。

### 3.1 环境安装命令的作用

文档里的 CUDA、Flash Attention 和 Python 环境安装，是为了使模型计算与评估程序能够运行，不属于模型训练本身。

真正实操时，应结合服务器已有驱动和依赖配置环境，不直接照抄原文的系统级安装命令。

服务器个人文件仍放在 `/mnt/sdb-36T/sdsun` 下，运行计算任务前确认资源可用，避免影响其他人的训练。

## 4. 为什么又下载 `finetuned-libero-spatial`？

[打开“模型权重批量下载”](E:/cpp_project/robot_ai_learn/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/02OpenVLA复现/02openvla复现.md:52)。

原文列出：

```text
openvla-7b
openvla-7b-finetuned-libero-spatial
openvla-7b-finetuned-libero-object
openvla-7b-finetuned-libero-goal
openvla-7b-finetuned-libero-10
```

拆开名字理解：

```text
openvla-7b
    → OpenVLA 的基础预训练模型。

finetuned
    → 已经进一步微调。

libero-spatial
    → 这份微调模型对应的 LIBERO 任务套件。
```

这里接上前面学过的预训练与微调：

```text
基础 OpenVLA
      ↓
使用对应任务的示范进一步训练
      ↓
得到面向该任务套件的检查点
      ↓
加载这个检查点进行评估
```

**本教程下载的是别人已经微调好的检查点，所以不用先亲自训练一遍，就可以尝试评估。**

基础预训练模型与面向某任务套件的微调模型不能仅凭名称相似就混用。读评估命令时，要看清加载哪个模型、评估哪组任务。

## 5. 第 3 节的命令实际在做什么？

[打开标准评估命令](E:/cpp_project/robot_ai_learn/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/02OpenVLA复现/02openvla复现.md:78)。

```bash
python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint /data1/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True
```

逐项阅读：

| 部分 | 含义 |
|---|---|
| `run_libero_eval.py` | 启动 LIBERO 评估程序 |
| `--model_family openvla` | 选择 OpenVLA 模型处理路径 |
| `--pretrained_checkpoint ...` | 指定加载哪一份已有检查点 |
| `--task_suite_name libero_spatial` | 指定在哪组任务上评估 |
| `--center_crop True` | 对输入图片做中心裁剪，匹配相应预处理约定 |

**这是一条评估命令，没有在这里让模型学习新参数。**

中心裁剪是根据训练预处理约定对输入进行处理，不是 YOLO 检测框，也不是自动识别物体后只截取物体区域。[官方图像处理与推理函数](https://github.com/openvla/openvla/blob/main/experiments/robot/openvla_utils.py#L115)

命令的运行主线是：

```text
加载 LIBERO-Spatial 微调模型
               ↓
创建 Spatial 套件中的一个任务
               ↓
设置场景初始状态
               ↓
读取当前相机图片＋任务指令
               ↓
OpenVLA 预测动作
               ↓
LIBERO 控制器执行
               ↓
得到新图片，继续预测
               ↓
任务完成或达到步数上限
               ↓
保存视频、统计成功率
```

`/data1/...` 是原作者的路径。以后在自己的服务器运行时，应使用 `/mnt/sdb-36T/sdsun` 下实际准备好的路径，而不是照抄作者目录。

## 6. 最需要新学的内容：OpenVLA 怎样输出动作 token？

[打开文档架构部分](E:/cpp_project/robot_ai_learn/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/02OpenVLA复现/02openvla复现.md:197)。

原版 OpenVLA 的主要结构：

```text
当前图像
   │
   ├──→ DINOv2 ──┐
   │              ├──→ 特征拼接 → Projector ──┐
   └──→ SigLIP ───┘                          │
                                             ▼
任务指令 → Tokenizer → 文本表示 ──────→ Llama 2
                                             │
                                             ▼
                                    逐个生成动作 token
                                             │
                                             ▼
                                    解码、反归一化
                                             │
                                             ▼
                                       数值动作
```

原版官方架构使用 DINOv2＋SigLIP、Projector 和 Llama 2 7B。[官方架构说明](https://openvla.github.io/#the-openvla-model)

这里的双编码器是两条视觉特征提取分支，不是要求有两台相机，也不是图像先后串行通过 DINOv2 和 SigLIP。

前面已经学习过视觉编码器和 Projector：视觉分支提取特征，Projector 将特征映射到语言模型需要的表示维度。接下来的重点转向：

```text
动作 token → 数值动作
```

### 6.1 动作怎样与 token 建立对应关系？

先只建立概念：

```text
示范动作中的一个连续数值
            ↓
归入一个数值区间
            ↓
映射成对应 token
            ↓
训练模型预测该 token
```

推理时反过来：

```text
模型生成动作 token
            ↓
解码为代表该区间的数值
            ↓
根据统计信息进行相应的反归一化
            ↓
得到交给执行接口的动作
```

[官方动作编码实现](https://github.com/openvla/openvla/blob/main/prismatic/vla/action_tokenizer.py)

虽然最终交给环境的是数值动作，原版模型内部并不是按 SmolVLA 的方式，从随机噪声逐步生成连续动作块。

动作 token 表示动作数值的编码，不要求它对应一个有自然语言含义的词，也不是先生成“向左移动”这样的文字，再由另一个程序理解文字去执行。

## 7. 原文中需要结合官方实现区分的地方

### 7.1 不要把通用组件选项当成具体模型的实际配置

原文架构部分列出了 CLIP、Vicuna、Mistral 等选项。阅读时需要区分：

```text
相关框架可能支持哪些组件
            ≠
本次具体 OpenVLA 模型使用哪些组件
```

本篇原版 OpenVLA 按官方说明理解为：

```text
DINOv2＋SigLIP → Projector → Llama 2 7B → 动作 token
```

不能将原文举例中的 DINOv2＋CLIP，直接当成此次官方检查点的固定结构。[官方架构说明](https://openvla.github.io/#the-openvla-model)

### 7.2 原文“连续动作输出”省略了内部 token 生成与解码

最终接口可以返回连续数值形式的动作，但内部先生成动作 token，再将其解码为数值。因此，“最终返回数值动作”不等于“内部使用直接连续回归或流匹配”。

[官方 `predict_action()`](https://github.com/openvla/openvla/blob/main/prismatic/extern/hf/modeling_prismatic.py#L449) 中可以看到生成 token、解码、反归一化的顺序。

### 7.3 原文的 Z-Score 归一化说明不能直接套用

[原文第 6.2 节](E:/cpp_project/robot_ai_learn/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/02OpenVLA复现/02openvla复现.md:291) 写了按均值和标准差进行归一化及反归一化。

但本次核对的原版官方 `predict_action()` 使用 `q01`、`q99` 分位数统计进行对应的反归一化，并根据 mask 选择需要转换的分量，不是统一“乘标准差、加均值”。

[官方反归一化实现](https://github.com/openvla/openvla/blob/main/prismatic/extern/hf/modeling_prismatic.py#L467)

这一节先记住差异。之后讲动作解码时，再按源码逐步展开，避免把 SmolVLA 或其他模型的归一化方式直接搬过来。

本次仅在学习记录中注明以上区别，没有修改原教程。

## 8. 当前进度与下一步

今天已经讲清楚：

- 本篇前半段是在准备、加载并测试已有策略，不是完整训练流程。
- GitHub 代码仓库与 Hugging Face 模型仓库的职责不同。
- 基础模型与 LIBERO 微调检查点的区别。
- 评估命令中模型路径、任务套件和图像预处理参数的作用。
- 原版 OpenVLA 与前面 SmolVLA 在输入和动作生成方式上的区别。
- 原文通用架构举例及归一化说明需要以具体官方实现为准。

下一步重点学习：**连续动作为什么能变成 token，模型怎样预测这些 token，以及 token 怎样再变回可以执行的动作。**

后续仍按“理解输入与输出 → 对照模型代码 → 理解训练目标 → 实际运行评估”的路径推进；本次尚未执行环境安装、权重下载或模型评估。

## 9. 连续动作为什么要变成 token？

本节继续讨论原版 OpenVLA。核心问题是：**机器人动作明明是数字，为什么要先把它变成 token？**

先看完整关系：

```text
训练：
示范动作数值 → 编码成动作 token → 让模型学习预测这些 token

推理：
图像＋指令 → 模型生成动作 token → 解码成数值动作 → 环境执行
```

原版 OpenVLA 将动作编码成词表中的 token，使已有视觉语言模型的序列预测机制可以用来预测机器人动作。它不是先生成一段自然语言操作说明，再让另一个程序理解这段文字。

### 9.1 先从一个动作分量开始

前面学过，一步动作可以包含：

```text
[平移 X，平移 Y，平移 Z，旋转三个分量，夹爪]
```

假设其中一个分量经过归一化后为：

```text
0.23
```

SmolVLA 可以通过流匹配生成这样的连续数值。原版 OpenVLA 则先判断 `0.23` 落在哪个数值区间，再用一个 token 代表这个区间。这叫**离散化**。

[查看本地动作编码函数](E:/cpp_project/openvla/prismatic/vla/action_tokenizer.py:38)；[官方动作编码源码](https://github.com/openvla/openvla/blob/main/prismatic/vla/action_tokenizer.py)。

### 9.2 用四个区间理解编码与解码

为了方便计算，临时将 `[-1, 1]` 分成四个区间：

| 数值区间 | 教学用 token 名称 | 解码时使用的区间中心 |
|---|---|---:|
| `[-1, -0.5)` | `T_A` | -0.75 |
| `[-0.5, 0)` | `T_B` | -0.25 |
| `[0, 0.5)` | `T_C` | 0.25 |
| `[0.5, 1]` | `T_D` | 0.75 |

**这里只是教学例子，`T_A` 等名字不是官方真实 token，官方使用更细的划分。**

编码过程：

```text
动作数值：0.23
      ↓
落在 [0, 0.5) 区间
      ↓
编码为 T_C
```

推理时，如果模型输出 `T_C`：

```text
T_C
 ↓
找到对应数值区间
 ↓
取区间中心
 ↓
0.25
```

注意：

```text
原始值：0.23
还原值：0.25
```

还原值不一定与原始值完全相同，这就是离散化带来的**量化误差**。区间越细，单个区间内的这种近似误差通常越小。

官方代码也是将动作分桶，在解码时使用对应区间中心。[本地解码函数](E:/cpp_project/openvla/prismatic/vla/action_tokenizer.py:49)

### 9.3 token 不是“向左”“向右”这些词

动作 token 是一种编码，不要求具有自然语言含义。

```text
T_C 的含义：
    这个动作分量落在某个数值区间。

不是：
    “抓杯子”这个完整行为。
```

可以理解为查表：

```text
token ID → 区间编号 → 代表数值
```

这个数值控制 X 方向还是夹爪，需要结合它在动作序列中的位置解释。

假设同一个 token 解码为 `0.25`：

```text
出现在第一个动作位置 → X 平移分量的数值为 0.25。
出现在第二个动作位置 → Y 平移分量的数值为 0.25。
```

**token 表示数值，序列位置帮助确定这个数值属于哪个动作分量。** 数值仍需经过相应的输出处理，不能在这里直接当成米或关节角。

### 9.4 一步七维动作，可以编码成七个动作 token

假设某一步的归一化动作为：

```text
[0.23, -0.12, 0.61, -0.72, 0.18, -0.31, 0.90]
```

沿用四个区间的教学映射：

```text
数值： [0.23, -0.12, 0.61, -0.72, 0.18, -0.31, 0.90]
          ↓      ↓     ↓      ↓     ↓      ↓     ↓
token：[T_C,    T_B,   T_D,   T_A,   T_C,   T_B,   T_D]
```

**七个 token 对应同一个控制时刻的七个动作分量，不是未来七步动作。**

```text
原版 OpenVLA 的这个例子：
    七个 token → 一步动作的七个分量。

SmolVLA 动作块的例子：
    四行、每行七个数 → 未来四步，每步七个分量。
```

原版 `predict_action()` 按动作维度设置生成长度，再把生成的 token 解码成动作。[本地生成位置，第 518 行](E:/cpp_project/openvla/prismatic/extern/hf/modeling_prismatic.py:518)

## 10. OpenVLA 怎样学习预测动作 token？

### 10.1 仍然先按时间配对，再转换监督目标

前面学过的配对关系仍然适用：

```text
某时刻的图片＋任务指令
           ↕ 配对
该时刻随后执行的示范动作
```

改变的是监督目标的表达：

```text
示范动作数值
      ↓
动作编码器
      ↓
示范动作 token 序列
      ↓
作为模型要学习预测的目标
```

例如：

```text
图片：夹爪位于杯子上方
指令：拿起杯子

对应示范动作编码：
[T_C, T_B, T_D, T_A, T_C, T_B, T_D]
```

模型学习在这样的图片和指令条件下预测这串动作 token。

官方数据转换代码将任务描述构造成提示词，将编码后的示范动作作为回答，并屏蔽提示词部分的监督损失；动作部分参与学习，结束 token 是否参与还受配置控制。

[本地训练样本转换类](E:/cpp_project/openvla/prismatic/vla/datasets/datasets.py:31)；[本地损失标签屏蔽位置](E:/cpp_project/openvla/prismatic/vla/datasets/datasets.py:63)。

### 10.2 与 SmolVLA 的损失有什么区别？

```text
SmolVLA：
预测变化率
    ↓
与目标变化率比较
    ↓
计算平方误差

原版 OpenVLA：
预测下一个 token 的概率
    ↓
查看示范中的正确 token 是哪个
    ↓
用交叉熵衡量预测
```

用一个简化分类例子说明。假设当前目标为 `T_C`，模型预测：

| token | 预测概率 |
|---|---:|
| `T_A` | 0.10 |
| `T_B` | 0.20 |
| `T_C` | 0.60 |
| `T_D` | 0.10 |

正确 token 的概率为 `0.60`，对应交叉熵：

$$
L=-\ln(0.60)\approx0.511
$$

如果训练后正确 token 的概率提高到 `0.90`：

$$
L=-\ln(0.90)\approx0.105
$$

损失变小了。**它在鼓励模型提高正确动作 token 的概率。**

这里用四类演示，真实模型的输出词表更大。token 预测得更准，也不自动等于任务一定完成，最终仍要看闭环执行效果。

## 11. 自回归生成：逐个预测，但不是逐个 token 驱动机器人

推理时可以按以下逻辑理解：

```text
图片＋指令
    ↓
生成第一个动作 token
    ↓
图片＋指令＋已经生成的第一个 token
    ↓
生成第二个动作 token
    ↓
继续生成剩余动作 token
    ↓
组成完整的一步动作
```

**后一个 token 的预测，可以利用前面已经生成的 token。**

但机器人不会生成一个 token 就立即动一下：

```text
生成这一步需要的动作 token
              ↓
解码、反归一化，得到完整动作向量
              ↓
交给 env.step(action)
```

训练时通常利用示范中的前序 token 预测后续 token，并通过因果掩码避免看到所预测位置及其后面的答案。

虽然训练序列中包含示范动作 token，但预测位置与监督目标存在移位关系，不是把当前答案直接交给当前预测位置去照抄。官方数据转换代码注明，使用模型的 `labels` 接口时，这个移位在模型内部完成。

训练也不要求像推理一样逐个 token 调用模型，可以在一次前向计算中处理多个预测位置。

## 12. 解码出数值以后，还要怎样才能执行？

解码成 `0.25`，不能直接认定为“移动 0.25 米”。完整链路是：

```text
动作 token
    ↓
解码为归一化空间中的代表数值
    ↓
使用对应数据集的统计信息反归一化
    ↓
按环境接口处理动作，例如夹爪编码
    ↓
控制器解释并执行
```

这解释了为什么模型检查点需要配套的动作统计信息。官方 `predict_action()` 根据 `unnorm_key` 选择统计数据，并对相应分量使用 `q01`、`q99` 进行反归一化。

[本地动作反归一化：第 527 行](E:/cpp_project/openvla/prismatic/extern/hf/modeling_prismatic.py:527)；[本地统计键检查：第 539 行](E:/cpp_project/openvla/prismatic/extern/hf/modeling_prismatic.py:539)。

这一阶段的关键区别：

```text
SmolVLA：
学习变化率 → 从噪声求解出连续动作块。

原版 OpenVLA：
学习预测动作 token → 将 token 解码为数值动作。
```

两者都学习“看到这种情况应该怎样行动”，只是动作表达方式和学习目标不同。

## 13. OpenVLA 官方代码已拉取到本地

| 项目 | 拉取结果 |
|---|---|
| 官方仓库 | [openvla/openvla](https://github.com/openvla/openvla) |
| 本地路径 | `E:\cpp_project\openvla` |
| 当前提交 | `c8f03f4` |
| 分支 | `main`，跟踪 `origin/main` |
| 拉取完成时的状态 | 工作区干净 |

本次下载的是源码，尚未安装依赖或下载模型权重、数据集，也未运行模型训练或评估。

### 13.1 直接打开本地代码

| 阅读内容 | 本地入口 |
|---|---|
| 动作数值编码成 token | [action_tokenizer.py，第 38 行](E:/cpp_project/openvla/prismatic/vla/action_tokenizer.py:38) |
| 动作 token 解码成数值 | [action_tokenizer.py，第 49 行](E:/cpp_project/openvla/prismatic/vla/action_tokenizer.py:49) |
| 训练样本构造 | [datasets.py，第 31 行](E:/cpp_project/openvla/prismatic/vla/datasets/datasets.py:31) |
| 模型预测动作与反归一化 | [modeling_prismatic.py，第 506 行](E:/cpp_project/openvla/prismatic/extern/hf/modeling_prismatic.py:506) |
| 图片与指令送入模型 | [openvla_utils.py，第 127 行](E:/cpp_project/openvla/experiments/robot/openvla_utils.py:127) |
| LIBERO 仿真评估入口 | [run_libero_eval.py，第 91 行](E:/cpp_project/openvla/experiments/robot/libero/run_libero_eval.py:91) |
| 模型微调入口 | [finetune.py，第 114 行](E:/cpp_project/openvla/vla-scripts/finetune.py:114) |

以上行号按本地文件核对。前面官方网页源码链接的行号可能与本地不同，后续阅读优先使用本地入口和函数名定位。

### 13.2 下一步学习位置

本次已经展开动作离散化、量化误差、动作 token 的含义、七个分量与未来七步的区别、训练交叉熵及自回归推理。

下一步用具体数字解释 `unnorm_key` 和反归一化，补全“动作 token → 可执行动作”的最后一段。此时仍处于代码与原理学习阶段，不把源码拉取等同于实操环境已搭建。

## 14. 从动作 token 到可执行动作：反归一化与环境接口

上一节走到：

```text
模型生成动作 token
        ↓
解码成归一化数值，例如 0.25
```

这一节回答：**`0.25` 到底代表多大的动作？为什么模型还需要 `unnorm_key`？**

对应本地代码：[模型动作预测函数，第 506 行](E:/cpp_project/openvla/prismatic/extern/hf/modeling_prismatic.py:506)。

### 14.1 同一个归一化数值，可以对应不同动作幅度

假设两份示范数据中，某个动作分量用于缩放的范围分别是：

```text
数据集 A：[-0.2, 0.6]
数据集 B：[-0.8, 0.8]
```

这些是教学示例，暂时不要把数值理解为米。

训练前，将不同范围映射到统一的 `[-1, 1]`，方便模型学习：

```text
原始动作范围                  归一化范围

数据集 A：[-0.2, 0.6] ──→      [-1, 1]

数据集 B：[-0.8, 0.8] ──→      [-1, 1]
```

推理后要变回去，就必须知道当时用了哪一套范围。这就是动作统计信息的作用。

**归一化数值相同，不意味着还原后的动作幅度相同。**

### 14.2 `unnorm_key` 选择哪一套动作统计信息

[查看读取动作统计信息的位置：第 527 行](E:/cpp_project/openvla/prismatic/extern/hf/modeling_prismatic.py:527)。

```python
action_norm_stats = self.get_action_stats(unnorm_key)
```

继续进入 [第 559 行的 `get_action_stats()`](E:/cpp_project/openvla/prismatic/extern/hf/modeling_prismatic.py:559)：

```python
return self.norm_stats[unnorm_key]["action"]
```

可以将 `norm_stats` 理解成一个分类保存的字典：

```text
norm_stats
├── 某数据集名称
│   └── action
│       ├── q01
│       ├── q99
│       └── mask
│
└── 另一数据集名称
    └── action
        ├── q01
        ├── q99
        └── mask
```

`unnorm_key` 就是选中的那个名称。

**它不是任务指令，也不是让模型切换成另一套网络权重。它选择的是动作还原需要的统计信息。**

在 LIBERO 评估脚本中，[第 100 行开始设置统计键](E:/cpp_project/openvla/experiments/robot/libero/run_libero_eval.py:100)：

```python
cfg.unnorm_key = cfg.task_suite_name
```

程序先尝试任务套件名称，例如 `libero_spatial`；随后还检查检查点中是否使用带 `_no_noops` 后缀的名称。

因此，最终必须匹配检查点实际包含的键，不能随便写一个任务名字。

### 14.3 `q01` 与 `q99` 是什么？

[查看第 529 行](E:/cpp_project/openvla/prismatic/extern/hf/modeling_prismatic.py:529)。原代码同时取出上下界，拆开理解如下：

```python
action_high = np.array(action_norm_stats["q99"])
action_low = np.array(action_norm_stats["q01"])
```

| 名称 | 含义 |
|---|---|
| `q01` | 该动作分量在统计数据中的第 1 百分位数 |
| `q99` | 该动作分量在统计数据中的第 99 百分位数 |

它们用于描述数据范围。相比直接使用最小值和最大值，这种分位数范围通常对少量极端数值不那么敏感。

**它们不是关节机械限位，也不是机器人绝对安全边界。**

每个动作分量有自己的统计值：

```text
X 分量有自己的 low、high。
Y 分量有自己的 low、high。
Z 分量有自己的 low、high。
……
```

不能拿 X 方向的范围还原所有动作分量。

### 14.4 用具体数字计算反归一化

[查看第 530 行开始的计算](E:/cpp_project/openvla/prismatic/extern/hf/modeling_prismatic.py:530)。

对于需要反归一化的分量，计算公式是：

$$
a=\frac{z+1}{2}(h-l)+l
$$

其中：

- $z$：token 解码得到的归一化数值。
- $l$：对应分量的 `q01`。
- $h$：对应分量的 `q99`。
- $a$：还原到数据动作尺度后的数值。

假设：

```text
z = 0.25
l = -0.2
h = 0.6
```

**第一步：把 `[-1, 1]` 中的位置换成 `[0, 1]` 中的比例。**

$$
r=\frac{z+1}{2}
=\frac{0.25+1}{2}
=0.625
$$

意思是处于整个区间的 **62.5% 位置**。

**第二步：计算原始区间的长度。**

$$
h-l=0.6-(-0.2)=0.8
$$

**第三步：从下限出发，前进区间长度的 62.5%。**

$$
a=-0.2+0.625\times0.8=0.3
$$

因此：

```text
token
  ↓ 解码
归一化数值 0.25
  ↓ 使用这套统计范围还原
动作数值 0.3
```

换成数据集 B 的范围 `[-0.8, 0.8]`，同一个 `0.25` 会还原为：

$$
a=-0.8+0.625\times1.6=0.2
$$

**这就是 `unnorm_key` 选错时，即使模型生成相同 token，执行动作幅度也可能出错的原因。**

这一步恢复的是数值尺度，不会消除前面动作离散化已经带来的量化误差，也不能保证恢复出原始示范中完全相同的数值。

### 14.5 为什么还有一个 `mask`？

同一段代码使用 `np.where`，逐分量选择输出。结构示意如下：

```text
actions = np.where(
    mask,
    反归一化后的数值,
    normalized_actions,
)
```

含义是：

```text
mask=True
    → 使用反归一化结果。

mask=False
    → 保留解码得到的数值。
```

不同动作分量可能采用不同约定，例如某些夹爪编码不需要与位姿分量使用同样的缩放方式。

**具体哪个分量跳过，要看实际统计信息中的 `mask`，不能仅凭“最后一维是夹爪”自行决定。**

### 14.6 反归一化结束，不等于可以直接把数值当成米

刚才算出：

```text
a = 0.3
```

准确含义是：还原到这份数据所采用的动作表示和尺度。

如果数据保存的是控制器输入，`0.3` 就是控制器输入值，之后还可能被控制器缩放为目标位置增量。

要分清两层处理：

```text
第一层：模型数据处理
归一化数值 → 数据原有的动作尺度

第二层：环境控制
动作指令 → 控制器解释、缩放并产生控制量
```

这不是重复做同一件事：前者对齐训练数据的表达，后者完成机器人控制。

### 14.7 LIBERO 脚本拿到动作后还做了什么？

[打开评估脚本第 219 行](E:/cpp_project/openvla/experiments/robot/libero/run_libero_eval.py:219)。

```python
action = normalize_gripper_action(action, binarize=True)
```

它先转换夹爪编码，并在这里启用二值化处理。接着针对 OpenVLA 执行：

```python
action = invert_gripper_action(action)
```

用于调整夹爪方向约定。

[查看夹爪转换函数：第 75 行](E:/cpp_project/openvla/experiments/robot/robot_utils.py:75)。

这里的夹爪编码转换与模型内部按数据统计信息做反归一化，是不同环节。不能看到函数名中都有 normalize，就将它们混为同一种处理。

最后，[第 228 行执行环境动作](E:/cpp_project/openvla/experiments/robot/libero/run_libero_eval.py:228)：

```python
obs, reward, done, info = env.step(action.tolist())
```

这一步把数值动作交给环境，返回新观察、奖励、结束标志和额外信息。

### 14.8 把输出链路完整展开

```text
图像＋任务指令
        ↓
OpenVLA 生成动作 token
        ↓
token 解码成区间中心值
        ↓
unnorm_key 选择动作统计信息
        ↓
按各分量的 q01、q99、mask 反归一化
        ↓
转换夹爪编码
        ↓
env.step(action)
        ↓
环境控制器计算控制量
        ↓
物理仿真推进
        ↓
返回新图像和状态
```

到这里，原版 OpenVLA 的“输入 → 动作生成 → 数值还原 → 仿真执行”已经串起来。

下一步进入训练样本构造代码，看示范动作怎样变成 `input_ids` 和 `labels`，以及训练时为什么不会直接“抄答案”。当前仍是代码阅读与原理学习，尚未运行实际评估。

## 15. 重点问答：反归一化得到 0.3 后，它到底代表什么？

这次的问题是：

> 算到 0.3 后，不知道它代表什么。后面经过夹爪转换、环境执行、控制器和物理仿真，到底哪些参数发生了变化？比如 0.3 是不是关节角？

前面的反归一化例子只说明数值尺度怎样还原，还没有给这个数值明确的物理含义。

**数字本身没有固定含义，必须看它位于动作数组的哪个位置，以及环境采用什么控制接口。**

下面沿着当前核对的 LIBERO 默认配置，完整追踪一次。配置依据是本地 LIBERO 源码和 robosuite 1.4.0 的默认 `OSC_POSE` 配置；这是教学推导，不是实际运行模型得到的测量结果。

### 15.1 先给 0.3 一个明确的位置

假设模型反归一化后得到：

```python
action = [0.3, 0, 0, 0, 0, 0, 1]
#          ↑                 ↑
#      第一个分量         夹爪分量
```

本地 LIBERO 默认使用 `OSC_POSE` 末端位姿控制器：

[查看默认控制器配置](E:/cpp_project/openpi/third_party/libero/libero/libero/envs/env_wrapper.py:12)。

在这个配置下：

```text
前面三个分量：末端平移指令。
接着三个分量：末端旋转指令。
最后一个分量：夹爪指令。
```

所以，本例的 `0.3` 表示：

> 要求机械臂末端沿 X 正方向调整的控制指令。

**它不是第一个关节的角度，也不是直接移动 0.3 米。**

这里是在明确动作接口后，用前面的示例数值继续讲解，不是说已经运行模型测得这个动作。换数据或换控制器后，需要重新核对动作含义。

### 15.2 转换夹爪编码不会修改这个 0.3

[查看夹爪转换代码](E:/cpp_project/openvla/experiments/robot/robot_utils.py:75)。

相关函数只处理：

```python
action[..., -1]
```

也就是最后一个分量。

本例中：

```text
转换前：
[0.3, 0, 0, 0, 0, 0,  1]

转换夹爪编码、反转符号后：
[0.3, 0, 0, 0, 0, 0, -1]
```

**第一维 0.3 原样保留。**

因此，之前的流程图不能理解成“每个动作数值都经过夹爪转换”。整个动作数组经过这段函数，但函数只修改其中的夹爪分量。

### 15.3 进入环境后，0.3 被换算成末端目标位置的增量

[查看动作进入环境的位置](E:/cpp_project/openvla/experiments/robot/libero/run_libero_eval.py:228)。

```python
obs, reward, done, info = env.step(action.tolist())
```

在核对过的 robosuite 1.4.0 默认 `OSC_POSE` 配置中：

```text
平移控制输入：[-1, 1]
        ↓
目标位置增量：[-0.05, 0.05] 米
```

因此：

$$
\Delta x=0.3\times0.05=0.015\text{ 米}
$$

也就是 **1.5 厘米**。

这是该版本默认配置的缩放关系，不是所有控制器都固定如此。[官方控制器配置](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/config/osc_pose.json)

假设当前末端位置为：

```text
p_current = [0.400, 0.100, 0.300] 米
```

则建立的新目标为：

```text
p_goal = p_current + [0.015, 0, 0]

       = [0.415, 0.100, 0.300] 米
```

**此时首先改变的是控制器里的目标位置，机器人实际位置还没有因此瞬间改变。**

官方目标位置计算采用“当前位置＋增量”。[目标位置计算源码](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/utils/control_utils.py#L101)

### 15.4 控制器比较目标与实际位置，计算关节力矩

现在控制器面对的是：

```text
实际 X 位置：0.400 米
目标 X 位置：0.415 米

误差：0.015 米
```

它需要解决：应该怎样驱动各关节，让末端向目标靠近？

```text
目标位置 p_goal ───┐
                  ├──→ 位置误差
实际位置 p ───────┘
                         │
当前速度、朝向等 ────────┤
                         ▼
                    控制器计算
                         │
                         ▼
                各关节的控制力矩 τ
```

这一步用到前面学习的反馈控制、惯性信息和雅可比。

曾经学习的关系：

$$
\tau=J^T F
$$

帮助建立末端控制作用与各关节力矩之间的对应。实际 OSC 还加入其他计算和补偿项，不能将完整算法缩减成只有这一条公式。[官方控制器实现](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/osc.py#L245)

**这条链路不会简单地设置“第一个关节角度 = 0.3”。** 多个关节可能共同运动，使末端沿 X 方向靠近目标。

仅凭 `0.3` 不能算出每个关节的具体力矩，还需要当前姿态、速度、机器人结构、控制参数等信息。

### 15.5 物理仿真收到控制量后，才更新实际运动状态

MuJoCo 根据关节力矩、惯性、重力、接触等条件计算运动。

```text
关节受到力矩
      ↓
产生关节加速度
      ↓
经过时间推进，关节速度发生变化
      ↓
关节位置发生变化
      ↓
机械臂姿态变化，末端位置也随之变化
```

实际发生变化的状态包括：

```text
q：关节位置，其中转动关节对应角度。
q̇：关节速度。
末端位置与朝向。
夹爪状态。
可能被碰到或抓住的物体状态。
```

**目标 X 位置是 0.415 米，不代表一次 `env.step()` 结束后，实际 X 位置就一定等于 0.415 米。** 它可能还在接近目标，或者受到接触等因素影响。

新的机器人和物体状态用于更新观察及渲染相机画面，程序获得新的 `obs`。

### 15.6 把哪个量变化列出来

| 阶段 | 发生变化的量 | 本例含义 |
|---|---|---|
| 夹爪编码转换 | 动作最后一个分量 | 第一维 `0.3` 不变 |
| 控制器缩放 | 末端位置增量 | `0.3 → 0.015 米` |
| 设置目标 | `p_goal` | X 目标由当前 `0.400` 建立为 `0.415` |
| 反馈控制计算 | 关节控制力矩 `τ` | 计算各关节需要怎样出力 |
| 物理仿真推进 | 实际 `q`、`q̇` 等状态 | 各关节实际运动 |
| 更新观察 | 末端状态、图像等 | 把运动后的情况交回程序 |

**整个推理过程中，OpenVLA 的网络权重没有更新。变化的是动作指令、控制目标、控制量和物理状态。**

这里说的“量发生变化”，不能与训练时“神经网络参数更新”混淆。默认固定增益控制器也不意味着每个控制步都重新学习其增益。

### 15.7 如果 0.3 真的是关节角，会怎样？

那属于另一种动作接口。

假设接口明确规定：

```text
action[0] = 第一个关节的绝对目标角度，单位为弧度。
```

此时 `0.3` 才表示：

$$
q_{d,1}=0.3\text{ rad}\approx17.2^\circ
$$

之后可能采用：

```text
第一个关节的目标角度 0.3 rad
                ↓
与第一个关节的实际角度比较
                ↓
关节控制器计算力矩
                ↓
物理运动使实际关节角向目标靠近
```

如果接口规定的是角度增量，那么 `0.3` 又表示在当前角度上增加 `0.3 rad`，含义不同。

因此判断动作含义，需要同时具备：

```text
数值 0.3
   ＋
它在动作中的位置
   ＋
控制接口对这个位置的定义
   ↓
才能确定具体含义
```

当前讨论的 LIBERO 默认链路可以具体记为：

```text
第一维 0.3
    ↓
末端 X 方向目标增量 1.5 厘米
    ↓
控制器计算关节力矩
    ↓
关节实际运动
    ↓
末端位置变化，返回新的观察
```
