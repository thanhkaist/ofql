import os
from copy import deepcopy

import d4rl
import gym
import hydra, wandb, uuid
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE, ContinuousRectifiedFlow, OFQL
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DQLMlp , OFQLMlp
from cleandiffuser.utils import report_parameters, DQLCritic, FreezeModules
from utils import set_seed
from tqdm import tqdm
import time


import argparse

parser = argparse.ArgumentParser(
    description="Measure OFQL action-prediction MSE on in-distribution vs out-of-distribution states.")
parser.add_argument("--ckpt-dir", required=True,
                    help="Directory holding diffusion_ckpt_<ckpt>.pt and critic_ckpt_<ckpt>.pt "
                         "(e.g. results/ofql_d4rl_mujoco/<base_path>/halfcheetah-medium-v2/)")
parser.add_argument("--ckpt", default="latest", help="Checkpoint tag, e.g. 'latest' or 1000000")
parser.add_argument("--in-dataset", default="halfcheetah-medium-v2")
parser.add_argument("--ood-dataset", default="halfcheetah-expert-v2")
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--hidden-dim", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--num-candidates", type=int, default=50)
parser.add_argument("--weight-temperature", type=float, default=50.0)
parser.add_argument("--resample", action="store_true",
                    help="Q-weighted resampling among candidates (off by default)")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
HIDDEN_DIM = args.hidden_dim
LR = args.lr
NUM_CANDIDATES = args.num_candidates
WEIGHT_TEMPERATURE = args.weight_temperature
RESAMPLE = args.resample

SAVE_PATH_OFQL = args.ckpt_dir if args.ckpt_dir.endswith("/") else args.ckpt_dir + "/"
CKPT = args.ckpt

# Prepare datasets
indistribution_dataset = args.in_dataset
outofdistribution_dataset = args.ood_dataset

env_in = gym.make(indistribution_dataset)
env_ood = gym.make(outofdistribution_dataset)

dataset_in = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env_in), normalize_reward=True)
dataset_ood = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env_ood), normalize_reward=True)

dataloader_in = DataLoader(
    dataset_in, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
dataloader_ood = DataLoader(
    dataset_ood, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

obs_dim, act_dim = dataset_in.o_dim, dataset_in.a_dim
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# prepare model 
# OFQL model 
nn_diffusion = OFQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(device)
nn_condition = IdentityCondition(dropout=0.0).to(device)

actor_ofql = OFQL(
    nn_diffusion, nn_condition, optim_params={"lr": LR},
    x_max=+1. * torch.ones((1, act_dim), device=device),
    x_min=-1. * torch.ones((1, act_dim), device=device), ema_rate=True, device=device, time_dist=["uniform",0.4,1],adaptive_l2_loss=False,flow_ratio=0.5)

# ------------------ Critic ---------------------
critic_ofql = DQLCritic(obs_dim, act_dim, hidden_dim=HIDDEN_DIM).to(device)



actor_ofql.load(SAVE_PATH_OFQL + f"diffusion_ckpt_{CKPT}.pt")
critic_ckpt = torch.load(SAVE_PATH_OFQL + f"critic_ckpt_{CKPT}.pt")
critic_ofql.load_state_dict(critic_ckpt["critic"])

actor_ofql.eval()
critic_ofql.eval()

log_mse = []

prior = torch.zeros((BATCH_SIZE*NUM_CANDIDATES, act_dim), device=device)


# Loop over in-distribution dataset
for batch in tqdm(dataloader_in):
    
    obs = batch["obs"]["state"].to(device)
    act_gt = batch["act"].to(device)

    obs = obs.unsqueeze(1).repeat(1, NUM_CANDIDATES, 1).view(-1, obs_dim)

    # sample actions
    act, log = actor_ofql.sample(
        prior,
        solver="euler",
        n_samples= obs.shape[0],
        sample_steps=1,
        condition_cfg=obs, w_cfg=1.0,
        use_ema=True, temperature=1.0)

    # resample
    if RESAMPLE:
        with torch.no_grad():
            q = critic_ofql.q_min(obs, act)
            q = q.view(-1, NUM_CANDIDATES, 1)
            w = torch.softmax(q * WEIGHT_TEMPERATURE, 1)
            act = act.view(-1, NUM_CANDIDATES, act_dim)

            indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
            sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()
    else:
        act = act.view(-1, NUM_CANDIDATES, act_dim)
        indices = torch.zeros((act.shape[0],), dtype=torch.long, device=device)
        sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()
    # calculate MSE numpy
    mse = np.mean((sampled_act - act_gt.cpu().numpy())**2)
    log_mse.append(mse)


# Report In-distribution MSE
mean_mse_in = np.mean(log_mse)
std_mse_in = np.std(log_mse) / np.sqrt(len(log_mse))
print(f"In-distribution MSE: Mean = {mean_mse_in:.6f}, Std Error = {std_mse_in:.6f}")

# Loop over out-of-distribution dataset
log_mse = []
for batch in tqdm(dataloader_ood):
    
    obs = batch["obs"]["state"].to(device)
    act_gt = batch["act"].to(device)

    obs = obs.unsqueeze(1).repeat(1, NUM_CANDIDATES, 1).view(-1, obs_dim)

    # sample actions
    act, log = actor_ofql.sample(
        prior,
        solver="euler",
        n_samples= obs.shape[0],
        sample_steps=1,
        condition_cfg=obs, w_cfg=1.0,
        use_ema=True, temperature=1.0)

    # resample
    if RESAMPLE:
        with torch.no_grad():
            q = critic_ofql.q_min(obs, act)
            q = q.view(-1, NUM_CANDIDATES, 1)
            w = torch.softmax(q * WEIGHT_TEMPERATURE, 1)
            act = act.view(-1, NUM_CANDIDATES, act_dim)

            indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
            sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()
    else:
        act = act.view(-1, NUM_CANDIDATES, act_dim)
        indices = torch.zeros((act.shape[0],), dtype=torch.long, device=device)
        sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

    mse = np.mean((sampled_act - act_gt.cpu().numpy())**2)
    log_mse.append(mse)

# Report Out-of-distribution MSE
mean_mse_ood = np.mean(log_mse)
std_mse_ood = np.std(log_mse) / np.sqrt(len(log_mse))
print(f"Out-of-distribution MSE: Mean = {mean_mse_ood:.6f}, Std Error = {std_mse_ood:.6f}")