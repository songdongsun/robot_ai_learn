"""在服务器执行作者策略并保存视频；复用 CooHOI 的环境和模型加载代码。"""

import argparse
import json
import sys
from pathlib import Path

# 此脚本放在服务器工作目录的 tools/ 中，项目位于相邻的 CooHOI/。
workspace = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace / "CooHOI" / "coohoi"))

import run as project  # 先由项目加载 Isaac Gym，再导入 PyTorch。
from isaacgym import gymapi
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--video", required=True)
    parser.add_argument("--steps", type=int, default=300)
    options, project_args = parser.parse_known_args()
    if options.steps <= 0:
        raise ValueError("steps 必须大于 0")
    sys.argv = [sys.argv[0]] + project_args
    project.args = project.get_args()
    torch.cuda.set_device(project.args.device_id)  # 保持 CUDA 与 Vulkan 使用相同的物理卡。
    project.cfg, project.cfg_train, _ = project.load_cfg(project.args)
    project.cfg_train["params"]["config"]["device_name"] = project.args.rl_device
    project.cfg["env"]["motion_file"] = project.args.motion_file
    project.cfg["env"]["eval_mode"] = False
    project.cfg["env"]["save_motions"] = False
    project.cfg["env"]["enableCameraSensors"] = True
    project.cfg_train["params"]["seed"] = project.set_seed(42)
    runner = project.build_alg_runner(project.RLGPUAlgoObserver())
    runner.load(project.cfg_train)
    runner.reset()
    player = runner.create_player()
    # 先在 CPU 读取，避免检查点里保存的 cuda:0 意外占用其他人的 GPU。
    checkpoint = torch.load(project.args.checkpoint, map_location="cpu")
    player.model.load_state_dict(checkpoint["model"])
    player.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
    player._amp_input_mean_std.load_state_dict(checkpoint["amp_input_mean_std"])
    task = player.env.task
    camera_props = gymapi.CameraProperties()
    if task.num_envs not in (1, 2):
        raise ValueError("录像入口只支持单人 1 槽或双人 2 槽")
    camera_props.width, camera_props.height = 640 // task.num_envs, 480
    cameras = [task.gym.create_camera_sensor(env, camera_props) for env in task.envs]
    if any(camera < 0 for camera in cameras):
        raise RuntimeError("无法创建离屏相机，请检查 NVIDIA Vulkan 图形设备")
    video = Path(options.video).resolve()
    video.parent.mkdir(parents=True, exist_ok=True)
    observation = player.env_reset()
    player.get_batch_size(observation["obs"], 1)  # 告知策略输入已有环境批次维度。
    done_indices = []
    rewards, resets = [], 0
    fps = round(1 / (task.gym.get_sim_params(task.sim).dt * task.control_freq_inv))
    # 写入器关闭后，再按相机、仿真的顺序释放图形资源。
    with imageio.get_writer(str(video), fps=fps, codec="libx264", quality=7) as writer:
        for step in range(options.steps):
            observation = player.env_reset(done_indices)
            with torch.no_grad():
                action = player.get_action(observation, True)
                observation, reward, done, info = player.env_step(player.env, action)
            rewards.append(float(reward.mean().item()))
            done_indices = done.nonzero(as_tuple=False)[:, 0]
            resets += int(done_indices.numel())
            # 相机只显示关联环境的物体；双人任务以左右双视角呈现两个环境槽。
            roots = task._humanoid_root_states[:, :3].detach().cpu().numpy()
            boxes = task._box_states[:, :3].detach().cpu().numpy()
            for index, (env, camera) in enumerate(zip(task.envs, cameras)):
                center = (roots[index] + boxes[index]) / 2
                distance = max(3.0, float(np.linalg.norm(roots[index, :2] - boxes[index, :2])))
                target = gymapi.Vec3(float(center[0]), float(center[1]), 1.0)
                eye = gymapi.Vec3(float(center[0]) + distance, float(center[1]) - distance, 1 + distance * 0.6)
                task.gym.set_camera_location(camera, env, eye, target)
            task.gym.fetch_results(task.sim, True)
            task.gym.step_graphics(task.sim)
            task.gym.render_all_camera_sensors(task.sim)
            views = []
            for index, (env, camera) in enumerate(zip(task.envs, cameras)):
                rgba = task.gym.get_camera_image(task.sim, env, camera, gymapi.IMAGE_COLOR)
                view = Image.fromarray(rgba.reshape(480, camera_props.width, 4)[:, :, :3])
                ImageDraw.Draw(view).text((8, 8), f"Actor {index + 1} | step {step + 1}", fill="white")
                views.append(np.asarray(view))
            writer.append_data(np.concatenate(views, axis=1))
            if (step + 1) % 60 == 0:
                print(f"Recorded {step + 1}/{options.steps} frames", flush=True)
    summary = {"frames": len(rewards), "fps": fps, "mean_task_reward": float(np.mean(rewards)),
               "episode_resets": resets, "checkpoint": project.args.checkpoint,
               "note": "推理演示，不是训练；奖励均值不是抓取成功率。"}
    video.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.cuda.synchronize()
    for env, camera in zip(task.envs, cameras):
        task.gym.destroy_camera_sensor(task.sim, env, camera)
    task.gym.destroy_sim(task.sim)
    print(f"Saved {video}", flush=True)


if __name__ == "__main__":
    main()
