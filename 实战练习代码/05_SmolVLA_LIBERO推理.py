"""SmolVLA + LIBERO 仿真评估：按模块阅读的中文注释版。

来源：06-策略抓取或抓取VLA/大模型控制、VLA、VLM/
      01SmolVLA-LIBERO/01SmolVLA-libero.md 中第一份完整评估脚本。
版本：原教程指定 LeRobot d602e816、robosuite 1.4.0；不要直接套用新版接口。

阅读顺序：先读模块六的 while 循环，再回看模块二、三、四、五。
主线：环境观测 → 整理输入 → 策略取动作 → 夹爪编码转换 → 环境执行。
这是推理评估代码，没有 loss.backward()、优化器更新或训练过程。

运行前需要：匹配的 LeRobot/LIBERO 环境、LIBERO 资源、适配该任务和输入格式的模型。
policy_path 请指定训练后的 pretrained_model 目录，里面需要模型配置和权重。
仅有 smolvla_base 不代表已经学会本教程的 LIBERO 任务。

命令示例（将路径换成自己的；在已配置好的环境中运行）：
python 05_SmolVLA_LIBERO推理.py --policy_path=/实际路径/pretrained_model --device=cuda
可追加：--task_suite_name=libero_spatial --num_trials_per_task=1
默认视频写到本文件旁边的 .cache/smolvla_libero/videos。

本文件没有安装环境或启动服务器任务；整理时只进行静态语法检查。
图像方向、状态字段、夹爪编码均沿用原教程，换模型时必须核对其数据约定。
"""

# ==================== 模块一：导入工具 ====================
import dataclasses  # 用数据类集中声明命令行参数。
import logging  # 打印任务、结果和视频路径。
import math  # 四元数转换时使用反余弦与浮点数比较。
import os  # 设置当前进程的环境变量。
import pathlib  # 组合模型、任务描述文件与视频的路径。

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭 tokenizer 的内部并行。

import draccus  # 将命令行参数解析成下面的 Args 对象。
import imageio  # 把渲染画面写入视频；编码 MP4 还需要可用的 FFmpeg 后端。
import numpy as np  # 处理图像、状态数组与夹爪动作。
import torch  # 加载神经网络，并进行张量运算。
from libero.libero import benchmark, get_libero_path  # 读取任务套件和 LIBERO 资源路径。
from libero.libero.envs import OffScreenRenderEnv  # 创建可以离屏渲染的仿真环境。
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # SmolVLA 策略接口。


# ==================== 模块二：运行参数 ====================
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # 沿用原文的等待动作：前六维为零，最后一维控制夹爪。
LIBERO_ENV_RESOLUTION = 256  # 两路相机画面的高度和宽度，与原教程保持一致。


@dataclasses.dataclass  # 自动生成保存这些参数的数据对象。
class Args:
    policy_path: str  # 必填：模型目录或兼容的 Hub 模型 ID，避免无意中评估基础模型。
    task_suite_name: str = "libero_spatial"  # 任务集合，也可选 libero_object、libero_goal 等。
    num_steps_wait: int = 10  # 每回合开始时，先推进若干步，让场景里的物体稳定下来。
    num_trials_per_task: int = 1  # 每个任务尝试几回合；一次结果不能代表稳定成功率。
    video_out_path: str = str(
        pathlib.Path(__file__).resolve().parent / ".cache" / "smolvla_libero" / "videos"
    )  # 默认把生成的视频放在实战目录的缓存文件夹内。
    device: str = "cuda"  # 模型计算设备；选择 cpu 仍需要完整依赖及可用的仿真渲染环境。
    seed: int = 7  # 固定部分随机数来源，便于重复观察；不保证所有硬件上逐位一致。


# ==================== 模块三：转换夹爪动作的编码 ====================
def normalize_gripper_action(action, binarize=True):
    """沿用教程：将最后一维夹爪值从 [0, 1] 映射到 [-1, 1]。"""
    # 这里是夹爪接口编码转换，不是对整个动作向量做数据集反归一化。
    action[..., -1] = 2 * action[..., -1] - 1  # 最后一维是夹爪：0→-1，0.5→0，1→1。
    if binarize:  # 调用时传 False 就保留连续值，不进行二值化。
        action[..., -1] = np.sign(action[..., -1])  # 正数→1，负数→-1，零仍为零。
    return action  # 返回完成转换的动作数组。


def invert_gripper_action(action):
    """沿用教程：反转最后一维符号，匹配环境侧夹爪方向。"""
    action[..., -1] *= -1.0  # 与上一函数合起来，最终变换为 g_out = 1 - 2*g_in。
    return action  # 其余动作分量不在这两个函数中修改。


# ==================== 模块四：将末端四元数转换为旋转向量 ====================
def _quat2axisangle(quat):
    """输入 [qx, qy, qz, qw]，输出三维旋转向量：单位旋转轴乘旋转角。"""
    # 这不是欧拉角，也不是三个关节角；只是同一个末端朝向的另一种表示。
    quat = np.asarray(quat).copy()  # 使用副本，避免修改 obs 中原始的四元数数组。
    quat[3] = np.clip(quat[3], -1.0, 1.0)  # 限制数值误差，确保 acos 的输入在合法范围。
    den = np.sqrt(1.0 - quat[3] * quat[3])  # 对单位四元数，它对应 sin(旋转角/2)。
    if math.isclose(den, 0.0):  # 对应接近零旋转的特殊情况，避免除以零。
        return np.zeros(3)  # 用零向量表示零旋转。
    return quat[:3] * (2.0 * math.acos(quat[3])) / den  # 旋转轴乘角度，角度单位为弧度。


# ==================== 模块五：根据任务创建仿真场景 ====================
def _get_libero_env(task, resolution, seed):
    task_description = task.language  # 读取该任务的自然语言指令。
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )  # BDDL 文件描述任务场景、物体及目标条件，区别于神经网络权重。
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),  # 指定要加载的任务描述文件。
        camera_heights=resolution,  # 设置相机图像高度。
        camera_widths=resolution,  # 设置相机图像宽度。
    )  # 离屏渲染意味着可以获取图像，不要求弹出交互窗口。
    env.seed(seed)  # 设置环境随机种子。
    return env, task_description  # 同时返回仿真对象和输入模型的任务指令。


# ==================== 模块六：加载模型，循环观察、决策和执行 ====================
@draccus.wrap()  # 从命令行读取参数，并传给 eval_libero 的 args。
def eval_libero(args: Args) -> None:
    torch.manual_seed(args.seed)  # 设置 PyTorch 随机种子，动作生成的噪声也涉及随机数。
    np.random.seed(args.seed)  # 设置 NumPy 随机种子。

    # ---------- 6.1 加载已经训练好的策略 ----------
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)  # 读取模型配置和权重。
    policy.to(args.device)  # 将模型放到选定设备。
    policy.eval()  # 切换评估行为；它本身并不等于禁用梯度，下面还会用 inference_mode。

    # ---------- 6.2 选任务集合，并限制每个回合的最长执行步数 ----------
    benchmark_dict = benchmark.get_benchmark_dict()  # 取得名称到任务套件类的映射。
    task_suite = benchmark_dict[args.task_suite_name]()  # 实例化指定任务集合。
    max_steps = {
        "libero_spatial": 220,  # 沿用原文各任务集合的评估长度限制。
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[args.task_suite_name]  # 这里的步是环境控制步，不是训练更新步或流匹配求解步。
    video_dir = pathlib.Path(args.video_out_path)  # 把输出目录字符串变成路径对象。
    video_dir.mkdir(parents=True, exist_ok=True)  # 需要时创建视频目录。
    total_episodes, total_successes = 0, 0  # 分别累计已执行回合数和成功回合数。

    for task_id in range(task_suite.n_tasks):  # 逐个评估任务套件中的任务。
        task = task_suite.get_task(task_id)  # 读取本任务的描述。
        initial_states = task_suite.get_task_init_states(task_id)  # 读取预先准备的初始场景状态。
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        try:  # 无论正常完成还是报错，最后都释放当前任务的仿真环境。
            num_trials = min(args.num_trials_per_task, len(initial_states))  # 不超过已有初始状态数量。
            for episode_idx in range(num_trials):  # 一回合就是从初始状态开始尝试一次完整任务。
                logging.info("任务 %s / 回合 %s：%s", task_id, episode_idx, task_description)
                env.reset()  # 重置物理环境。
                policy.reset()  # 重置策略的回合缓存；不是清空学到的网络权重。
                obs = env.set_init_state(initial_states[episode_idx])  # 指定本回合物体等的初始状态。
                for _ in range(args.num_steps_wait):  # 等待初始场景稳定，暂时不调用模型。
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)  # 用等待动作推进物理仿真。

                t = 0  # 当前回合已执行的策略控制步数。
                frames = []  # 保存画面，回合结束后编码成视频。
                done = False  # 沿用此 LIBERO 教程的任务完成标志。

                while t < max_steps:  # 【重点】一次循环：观察 → 取动作 → 执行 → 获得新观察。
                    # ---------- 6.3 从环境读取两路图像 ----------
                    # 两个轴反向切片相当于旋转 180°，是本教程匹配训练图像方向的约定。
                    # ascontiguousarray 还会消除翻转切片产生的负步长，便于转换成 Tensor。
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    frames.append(agentview_image)  # 记录执行本次动作之前的外部相机画面。

                    # ---------- 6.4 从环境读取机器人状态 ----------
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],  # 末端当前位置，三个分量。
                            _quat2axisangle(obs["robot0_eef_quat"]),  # 末端朝向，转成三维旋转向量。
                            obs["robot0_gripper_qpos"],  # 夹爪关节的位置状态。
                        )
                    )  # 按固定顺序拼接；这里不是整条机械臂的全部关节角 q。

                    # ---------- 6.5 按模型需要的字段名与张量形状组织输入 ----------
                    observation = {
                        "observation.images.image": torch.from_numpy(agentview_image / 255.0)
                        .permute(2, 0, 1)  # 外部图像：[H,W,C] → [C,H,W]，像素已缩放到 [0,1]。
                        .to(torch.float32)  # 将像素保存为 float32。
                        .to(args.device)  # 把图像放到与模型相同的计算设备。
                        .unsqueeze(0),  # 加批次维度：[C,H,W] → [1,C,H,W]。
                        "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0)
                        .permute(2, 0, 1)  # 手腕图像采用相同的通道顺序。
                        .to(torch.float32)  # 使用 float32。
                        .to(args.device)  # 放到同一设备。
                        .unsqueeze(0),  # 一次只输入一个环境的画面。
                        "observation.state": torch.from_numpy(state)
                        .to(torch.float32)  # 机器人状态使用 float32。
                        .to(args.device)  # 状态也放到同一设备。
                        .unsqueeze(0),  # [状态维度] → [1,状态维度]。
                        "task": task_description,  # 任务指令；文字的进一步处理由策略内部完成。
                    }  # 图像回答“看到了什么”，状态回答“自身怎样”，语言回答“要做什么”。

                    # ---------- 6.6 调用策略，取得这一次要执行的动作 ----------
                    with torch.inference_mode():  # 推理时不记录反向传播需要的计算图。
                        action_tensor = policy.select_action(observation)  # 输入观察，取得当前一步动作。
                    # select_action 是执行接口；动作块生成与缓存管理在对应版本的策略内部。
                    # 不能把每次 select_action 都理解为重新生成整块动作，更不是更新网络参数。
                    action = action_tensor.cpu().numpy()[0]  # 转到 CPU/NumPy，去掉批次维度。

                    # ---------- 6.7 对齐夹爪指令的编码 ----------
                    action = normalize_gripper_action(action, binarize=False)  # 最后一维映射到 [-1,1]。
                    action = invert_gripper_action(action)  # 再反转方向，沿用教程与环境的约定。
                    # action 不是电机 ID，也不能直接认定为六个关节角：含义由动作接口决定。

                    # ---------- 6.8 交给环境控制器执行，并推进物理仿真 ----------
                    obs, _, done, _ = env.step(action)  # 返回新观察、奖励、完成标志、额外信息。
                    # 此处用 _ 表示不使用奖励与额外信息；本循环没有 PPO 或其他训练更新。
                    # 新 obs 会在下一次循环中成为模型输入，形成观察—执行闭环。
                    t += 1  # 实际执行了一步，增加控制步计数。
                    if done:  # 本教程按 LIBERO 的完成标志统计成功，不应套用于所有 Gym 环境。
                        break  # 本回合任务已完成，无须继续到 max_steps。

                # ---------- 6.9 保存结果：看视频，同时查看是否完成任务 ----------
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))  # 补上最后动作后的画面。
                total_episodes += 1  # 一个完整评估回合结束。
                total_successes += int(done)  # True 计为 1，False 计为 0。
                suffix = "success" if done else "failure"  # 用结果区分视频文件。
                video_path = video_dir / f"task_{task_id}_episode_{episode_idx}_{suffix}.mp4"
                with imageio.get_writer(str(video_path), fps=30) as writer:  # 30 是视频播放帧率，不等于控制频率。
                    for frame in frames:  # 按采集顺序逐帧写入。
                        writer.append_data(frame)  # 保存 RGB 图像；退出 with 后关闭视频文件。
                logging.info("完成=%s，执行步数=%s，视频=%s", done, t, video_path)
        finally:
            env.close()  # 释放本任务的渲染和仿真资源；出错时仍执行，异常继续向外抛出。

    if total_episodes:  # 至少完成过一个评估回合时，才能计算成功率。
        logging.info(
            "累计成功 %s / %s，成功率 %.1f%%",
            total_successes, total_episodes, 100.0 * total_successes / total_episodes,
        )  # 成功率衡量任务执行结果，不能用训练损失直接替代。


# ==================== 模块七：脚本启动入口 ====================
if __name__ == "__main__":  # 直接运行本文件时执行；作为模块导入时不自动启动仿真。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    eval_libero()  # draccus 解析命令行参数，进入模块六的评估主流程。
