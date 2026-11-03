# SAM 与深度估计交互实战

使用现有 Conda 环境 `embodied`。当前电脑为 Intel 核显，安装 CPU 版 PyTorch，默认采用 **SAM ViT-B + Intel/dpt-large**。相比教材 ViT-H，ViT-B 更适合本机体验；分割效果可能不同，但点击、掩码与深度统计流程相同。

## 启动

```powershell
conda activate embodied
cd E:\cpp_project\robot_ai_learn\实战练习代码
python 03_SAM与深度估计交互.py
```

使用自己的图片：

```powershell
python 03_SAM与深度估计交互.py --image "E:\图片\测试.jpg"
```

等待深度估计和 SAM 图像编码完成后才会出现窗口。每张图片只预计算一次，之后点击会复用图像特征。CPU 运行不保证实时。

本机示例验证：深度模型加载与计算约 4 秒，SAM 图像准备约 8 秒，单次点击分割约 0.2 秒；实际时间随图片和系统负载变化。

当前 Transformers 加载 DPT 时可能提示 `neck.fusion_stage.layers.0.residual_layer1` 的四个参数未初始化。已检查实现：第一层融合不传入 residual，因此不会调用这条分支；这些参数不参与本示例的推理。此说明仅针对上述参数，不适用于其他缺失权重提示。

- 左键：添加前景点，选中物体。
- 右键：添加背景点，排除错误区域。
- R：清空提示，重新选择物体。
- S：保存当前叠加图、黑白掩码和深度可视化。
- Q / Esc：退出。

窗口中文字用英文以避免 OpenCV 中文字体乱码；代码及终端说明使用中文。

深度值为相对逆深度相关值，**不是米，也不是物体中心的真实距离**。SAM score 是模型给出的掩码质量估计，不是“识别为某类别的概率”。该程序没有进行类别识别。

## 模型与结果

模型、默认示例和结果保存在脚本旁 `.cache/sam_depth/`，已有仓库规则忽略 `.cache`，不会将大型权重加入 Git。

- `sam_vit_b_01ec64.pth`：Meta 官方 SAM 权重。
- `truck.jpg`：Meta 官方示例图片。
- `dpt-large/`：DPT 权重与配置，加载时不联网。
- `result/`：按 S 保存的图片；重复保存覆盖本目录同名结果。

官方来源：

- [SAM 项目](https://github.com/facebookresearch/segment-anything)
- [ViT-B 权重](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- [示例图片](https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg)
- [DPT-Large 模型](https://huggingface.co/Intel/dpt-large)

## 环境复现

在已激活的 `embodied` 环境中：

```powershell
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==4.57.1 opencv-python==4.11.0.86 segment-anything==1.0 pillow numpy
```

SAM 权重与示例图片从上述官方链接下载至 `.cache/sam_depth/`。DPT 的 `config.json`、`preprocessor_config.json` 和 `model.safetensors` 从官方模型仓库下载至 `.cache/sam_depth/dpt-large/`。本次已准备本地模型后，脚本加载模型不再访问网络；迁移电脑时需同步这些文件或重新下载。

无窗口验证完整流程：

```powershell
python 03_SAM与深度估计交互.py --check
```

它会对默认图片中心点执行分割，检查非空掩码并保存预览；自选图片的中心可能不是你希望选择的物体。
