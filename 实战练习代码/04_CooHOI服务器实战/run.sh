#!/usr/bin/env bash
# 在服务器运行：bash tools/run.sh GPU编号 single|double|infer-check|train-check [附加参数]
set -euo pipefail
GPU_ID=${1:?请提供 GPU 编号，先用 nvidia-smi 查看负载}
MODE=${2:-single}
shift "$(( $# >= 2 ? 2 : 1 ))"
WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /mnt/sdb-36T/sdsun/miniconda3/etc/profile.d/conda.sh
conda activate coohoi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export COOHOI_GRAPHICS_DEVICE="$GPU_ID"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export WANDB_MODE=offline
export WANDB_DIR="$WORKSPACE/logs"
export OMP_NUM_THREADS=2
export MAX_JOBS=2
export XDG_CACHE_HOME="$WORKSPACE/.cache"
export XDG_CONFIG_HOME="$WORKSPACE/.config"
export XDG_DATA_HOME="$WORKSPACE/.local/share"
export TORCH_EXTENSIONS_DIR="$WORKSPACE/.cache/torch_extensions"
export CUDA_CACHE_PATH="$WORKSPACE/.cache/cuda"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
export TMPDIR="$WORKSPACE/tmp"
export WANDB_CACHE_DIR="$WORKSPACE/.cache/wandb"
export WANDB_CONFIG_DIR="$WORKSPACE/.config/wandb"
export WANDB_DATA_DIR="$WORKSPACE/.local/share/wandb"
export PYTHONPYCACHEPREFIX="$WORKSPACE/.cache/pycache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
cd "$WORKSPACE/CooHOI"
mkdir -p "$WORKSPACE/results"
case "$MODE" in
  single)
    # 多 GPU 服务器录像时保留物理编号，避免 CUDA 屏蔽与 Vulkan 枚举不一致。
    unset CUDA_VISIBLE_DEVICES
    python "$WORKSPACE/tools/record_policy.py" --video "$WORKSPACE/results/single.mp4" \
      --sim_device "cuda:$GPU_ID" --rl_device "cuda:$GPU_ID" \
      --test --headless --num_envs 1 --task HumanoidAMPCarryObject \
      --cfg_env coohoi/data/cfg/humanoid_carrybox.yaml \
      --cfg_train coohoi/data/cfg/train/amp_humanoid_task.yaml \
      --motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
      --checkpoint coohoi/data/models/SingleAgent.pth "$@"
    ;;
  double)
    unset CUDA_VISIBLE_DEVICES
    python "$WORKSPACE/tools/record_policy.py" --video "$WORKSPACE/results/double.mp4" \
      --sim_device "cuda:$GPU_ID" --rl_device "cuda:$GPU_ID" \
      --test --headless --num_envs 2 --task ShareHumanoidCarryObject \
      --cfg_env coohoi/data/cfg/share_humanoid_carrybox.yaml \
      --cfg_train coohoi/data/cfg/train/share_humanoid_task_coohoi.yaml \
      --motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
      --checkpoint coohoi/data/models/TwoAgent.pth "$@"
    ;;
  train-check)
    # 仅验证采样、反向传播和保存检查点，不是正式训练或有效策略。
    python coohoi/run.py --headless --num_envs 16 --task HumanoidAMPCarryObject \
      --cfg_env coohoi/data/cfg/humanoid_carrybox.yaml \
      --cfg_train "$WORKSPACE/tools/train-check.yaml" \
      --motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
      --max_iterations 2 --wandb --wandb_name environment-check \
      --output_path "$WORKSPACE/results/training-check" "$@"
    ;;
  infer-check)
    python coohoi/run.py --test --headless --num_envs 1 --episode_length 120 \
      --task HumanoidAMPCarryObject --cfg_env coohoi/data/cfg/humanoid_carrybox.yaml \
      --cfg_train "$WORKSPACE/tools/train-check.yaml" \
      --motion_file coohoi/data/motions/coohoi_data/coohoi_data.yaml \
      --checkpoint coohoi/data/models/SingleAgent.pth "$@"
    ;;
  *) echo "模式必须是 single、double、infer-check 或 train-check" >&2; exit 2 ;;
esac
