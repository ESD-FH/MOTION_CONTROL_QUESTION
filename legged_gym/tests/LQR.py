# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch


def test_env(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs =  min(env_cfg.env.num_envs, 10)
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, _ = env.reset()
    for i in range(int(10*env.max_episode_length)):

        #/************************************************** INPUT OBSERVATIONS **************************************************/
        base_pitch = obs[..., 0]           # (N,)
        base_ang_vel = obs[..., 1:4]       # (N,3)
        gyro_pitch = base_ang_vel[..., 1]  # (N,)
        dof_vel = obs[..., 4:]             # (N,2) -> left/right wheel joint qd
        #/************************************************** LQR **************************************************/








        #在此处补充LQR控制器









        #/************************************************** OUTPUT TORQUES TO ACTIONS **************************************************/
        wheel_T_left = torch.zeros(env.num_envs, device=env.device)
        wheel_T_right = torch.zeros(env.num_envs, device=env.device)
        wheel_T_left = torch.clamp(wheel_T_left, -0.18, 0.18)
        wheel_T_right = torch.clamp(wheel_T_right, -0.18, 0.18)

        #/************************************************** ENV STEP **************************************************/
        actions = torch.stack([wheel_T_left, wheel_T_right], dim=-1)
        obs, _, rew, done, info = env.step(actions)

if __name__ == '__main__':
    args = get_args()
    args.task = "balance_car"
    test_env(args)
