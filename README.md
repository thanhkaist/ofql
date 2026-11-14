# One-Step Flow Q-Learning (OFQL)

Official implementation of:

**[One-Step Flow Q-Learning: Addressing the Diffusion Policy Bottleneck in Offline Reinforcement Learning](https://arxiv.org/abs/2508.13904)**
Thanh Nguyen, Chang D. Yoo — ICLR 2026

## Overview

Diffusion Q-Learning (DQL) is a strong offline RL method, but its multi-step denoising makes both training and inference expensive: every actor update has to backpropagate through the full sampling chain.

OFQL removes that bottleneck. Instead of learning the *marginal* velocity at a single flow time, OFQL learns the **average velocity** over an interval within the Flow Matching framework. Because the average velocity already integrates the flow across that interval, a single Euler step suffices to generate an action:

* ⚡ One-step action generation during **both training and inference**
* 🧩 No distillation, no auxiliary one-step network
* 🔁 No backpropagation-through-time in the actor update
* 📈 Outperforms multi-step DQL across D4RL

## 🛠️ Setup

### Create a Conda Environment
```sh
conda create -n ofql python=3.10 mesalib glew glfw pip=23 setuptools=63.2.0 wheel=0.38.4 protobuf=3.20 -c conda-forge -y
conda activate ofql
```

### Install MuJoCo Simulator and `mujoco-py` (Important)
Follow the official installation guide [here](https://github.com/openai/mujoco-py#install-mujoco).
Alternatively, use the script below:

```sh
#!/bin/bash
sudo apt-get update && sudo apt-get install -y wget tar libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf cmake
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

USER_DIR=$HOME
wget -c "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
mkdir -p $USER_DIR/.mujoco
cp mujoco210-linux-x86_64.tar.gz $USER_DIR/mujoco.tar.gz
rm mujoco210-linux-x86_64.tar.gz
tar -zxvf $USER_DIR/mujoco.tar.gz -C $USER_DIR/.mujoco

echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$USER_DIR/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export MUJOCO_PY_MUJOCO_PATH=$USER_DIR/.mujoco/mujoco210" >> ~/.bashrc
source ~/.bashrc
```

### Install Dependencies
```sh
pip install -r requirements.txt
pip install -e .
```
`requirements.txt` pins the versions this code was run with (PyTorch 2.7.1; D4RL and `mjrl` from source). For a CUDA build other than 12.8, install PyTorch per the [official guide](https://pytorch.org/) first.

## 💻 Training

One entry point per domain:

```sh
# MuJoCo locomotion
CUDA_VISIBLE_DEVICES=0 python pipelines/train_ofql_d4rl_mujoco.py \
    task=halfcheetah-medium-expert-v2 mode=train seed=0 name=hc_me_s0 save_dir=results
```

Configs live in `configs/ofql/<domain>/`; per-task settings in `configs/ofql/<domain>/task/`. Any field can be overridden on the command line, e.g. `task.eta=0.05` or `gradient_steps=1000000`.

Checkpoints are written to `<save_dir>/<pipeline_name>/<base_path>/<env_name>/`, where `base_path` encodes `time_dist`, `eta`, sampling steps and seed, so runs with different settings never collide.

To log to Weights & Biases add `enable_wandb=true project=OFQL group=<group>`; logging is off by default.

## 📊 Evaluation

Evaluate a trained checkpoint with single-step action generation:

```sh
CUDA_VISIBLE_DEVICES=0 python pipelines/train_ofql_d4rl_mujoco.py \
    task=halfcheetah-medium-v2 mode=inference seed=0 \
    name=hc_me_s0 save_dir=results ckpt=latest
```

`mode=inference` reconstructs the checkpoint path from `task`, `seed`, `save_dir` and the actor settings, so pass the same values used for training. Use `ckpt=<step>` for a specific checkpoint.

## 📌 Benchmark Tasks

Evaluated on three D4RL domains (15 datasets):

| Domain | Datasets |
| --- | --- |
| **MuJoCo** | halfcheetah / hopper / walker2d × medium, medium-replay, medium-expert |

Per-task `eta` (the Q-loss weight) and `weight_temperature` (the action-selection softmax temperature) are set in each task YAML.

## 🔬 Analysis

`pipelines/generalize_state_{ofql,dql}.py` measure action-prediction MSE on in-distribution versus out-of-distribution states:

```sh
python pipelines/generalize_state_ofql.py \
    --ckpt-dir results/ofql_d4rl_mujoco/<base_path>/halfcheetah-medium-v2/ --ckpt latest \
    --in-dataset halfcheetah-medium-v2 --ood-dataset halfcheetah-expert-v2
```

## Citation

```bibtex
@inproceedings{
    nguyen2026onestep,
    title={One-Step Flow Q-Learning: Addressing the Diffusion Policy Bottleneck in Offline Reinforcement Learning},
    author={Nguyen, Thanh and Yoo, Chang D},
    booktitle={ICLR},
    year={2026},
}
```

## 🏷️ Acknowledgements
This code is built upon the [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) and [Habi](https://github.com/Josh00-Lu/Habi.git) repos; please cite these works as well. See the [license](LICENSE) for details.
