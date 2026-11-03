"""鼠标提示分割 + 相对逆深度统计；CPU 默认使用 SAM ViT-B。

运行：python 03_SAM与深度估计交互.py
换图：python 03_SAM与深度估计交互.py --image "图片路径"
无窗口验证：python 03_SAM与深度估计交互.py --check
"""

import argparse  # 读取图片路径和验证选项。
import gc  # 深度计算完成后释放不再使用的模型。
from pathlib import Path  # 路径以脚本位置为准，不依赖终端当前目录。
from time import perf_counter  # 测量实际推理时间。

import cv2  # 读取图像、显示窗口并处理鼠标事件。
import numpy as np  # 保存点坐标、掩码和计算区域统计。
import torch  # 执行模型推理。
from PIL import Image  # 为深度模型准备 RGB 图片。
from segment_anything import SamPredictor, sam_model_registry  # 官方 SAM 接口。
from transformers import DPTForDepthEstimation, DPTImageProcessor  # 教程的深度模型。


# ==================== 模块1：配置与图像读取 ====================
CACHE = Path(__file__).resolve().parent / ".cache" / "sam_depth"
DEPTH_MODEL = CACHE / "dpt-large"  # 官方 Intel/dpt-large 的本地副本，数值不是米。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=CACHE / "truck.jpg")
    parser.add_argument("--check", action="store_true", help="自动用中心点分割并保存预览，不弹窗")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 当前电脑默认走 CPU。
    torch.set_num_threads(4)  # 限制 CPU 推理线程，避免占满电脑。
    checkpoint = CACHE / "sam_vit_b_01ec64.pth"  # 与 vit_b 模型结构匹配。
    if not checkpoint.is_file():
        raise FileNotFoundError(f"缺少 SAM 权重：{checkpoint}，请按配套说明下载。")
    if not args.image.is_file():
        raise FileNotFoundError(f"找不到图片：{args.image}")
    # imdecode 支持 Windows 中文路径。
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码图片：{args.image}")
    scale = min(1.0, 960 / max(image.shape[:2]))  # 不放大小图，限制显示尺寸。
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 两个模型均使用 RGB。
    print(f"运行设备：{device}；图片：{args.image}", flush=True)

    # ==================== 模块2：预计算全图相对逆深度 ====================
    start = perf_counter()
    print("加载本地 DPT-Large，请等待……", flush=True)
    processor = DPTImageProcessor.from_pretrained(DEPTH_MODEL, local_files_only=True)
    depth_model = DPTForDepthEstimation.from_pretrained(
        DEPTH_MODEL, local_files_only=True, use_safetensors=True
    ).to(device).eval()
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(device)
    with torch.inference_mode():  # 推理无需梯度，减少内存占用。
        prediction = depth_model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=image.shape[:2], mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy().copy()  # 恢复到显示图尺寸，与 Mask 一一对应。
    if not np.isfinite(depth).all():
        raise ValueError("深度模型输出包含非有限值。")
    print(f"深度图准备完成，加载及计算耗时 {perf_counter() - start:.1f} 秒。", flush=True)
    del depth_model, inputs, prediction  # 深度只算一次，之后不必保留大模型。
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    depth_u8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)  # 仅供显示，不是米制距离。

    # ==================== 模块3：预计算 SAM 图像特征 ====================
    print("加载 SAM ViT-B 并编码图像，请等待……", flush=True)
    start = perf_counter()
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(device).eval()
    predictor = SamPredictor(sam)
    predictor.set_image(rgb)  # 最费时的图像编码，每张图片只进行一次。
    print(f"SAM 图像准备完成：{perf_counter() - start:.1f} 秒。", flush=True)
    points, labels = [], []  # 左键添加前景点，右键添加背景排除点。
    mask = None
    status = "Click object | Right: exclude | R: reset | S: save | Q/Esc: quit"

    # ==================== 模块4：点击后预测 Mask 并统计深度 ====================
    def select(event, x, y, flags, param):
        nonlocal mask, status
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        points.append([x, y])
        labels.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
        start = perf_counter()
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=np.asarray(points), point_labels=np.asarray(labels), multimask_output=False
            )
        mask = masks[0]
        values = depth[mask]  # 用分割区域选择深度值，不是自动修正深度。
        if values.size:
            status = f"Relative inverse depth: {values.mean():.3f} (NOT meters) | SAM score: {scores[0]:.3f}"
        else:
            status = "Empty mask: reset or add another foreground point"
        print(f"{status}；点击推理 {perf_counter() - start:.2f} 秒", flush=True)

    def render():
        display = image.copy()
        if mask is not None:
            display[mask] = (0.55 * display[mask] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (255, 255, 255), 1)
        for point, label in zip(points, labels):
            cv2.circle(display, tuple(point), 5, (0, 255, 0) if label else (0, 0, 255), -1)
        cv2.rectangle(display, (0, 0), (display.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(display, status, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, "Left: foreground | Right: background | R: reset | S: save | Q: quit",
                    (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return display

    def save_result():
        output = CACHE / "result"
        output.mkdir(parents=True, exist_ok=True)
        for name, data in [("interaction.png", render()), ("relative_depth.png", depth_color)]:
            cv2.imencode(".png", data)[1].tofile(output / name)
        if mask is not None:
            cv2.imencode(".png", mask.astype(np.uint8) * 255)[1].tofile(output / "mask.png")
        print(f"结果已保存：{output}", flush=True)

    # ==================== 模块5：验证模式与交互窗口 ====================
    if args.check:
        select(cv2.EVENT_LBUTTONDOWN, image.shape[1] // 2, image.shape[0] // 2, 0, None)
        if mask is None or not mask.any():
            raise RuntimeError("中心提示未产生有效掩码，请检查示例图片。")
        save_result()
        print("验证通过：深度估计、SAM 编码、点提示分割、结果保存。", flush=True)
        return
    window = "SAM + Depth - Interactive"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, select)
    cv2.imshow("Relative inverse depth (NOT meters)", depth_color)
    print("已就绪：左键选物体，右键排除，R重置，S保存，Q或Esc退出。", flush=True)
    try:
        while True:
            cv2.imshow(window, render())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r"):
                points.clear()
                labels.clear()
                mask = None
                status = "Reset. Click an object."
            elif key == ord("s"):
                save_result()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
