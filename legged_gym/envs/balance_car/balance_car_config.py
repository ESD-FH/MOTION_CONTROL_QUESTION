# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BalanceCarFlatCfg(LeggedRobotCfg):
    """平地上双轮小车：资源为 resources/robots/balance/test.urdf。"""

    class env(LeggedRobotCfg.env):
        num_envs = 1
        num_actions = 2
        # 无高度采样时：12 + 2*2 + 2 = 18
        num_observations = 18
        env_spacing = 2.0
        episode_length_s = 1000.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class commands(LeggedRobotCfg.commands):
        heading_command = True
        resampling_time = 4.0

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-2.0, 2.0]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.10]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = "T"
        stiffness = {"wheel": 0.0}
        damping = {"wheel": 0.0}
        action_scale = 1.0
        decimation = 1

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/balance/test.urdf"
        name = "balance_car"
        foot_name = "wheel"
        penalize_contacts_on = ["body"]
        terminate_after_contacts_on = []
        self_collisions = 1
        replace_cylinder_with_capsule = False

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.10
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.5
            ang_vel_xy = -0.05  
            orientation = -1.0
            torques = -0.0002
            collision = -0.5
            action_rate = -0.01
            feet_air_time = 0.0
            base_height = 0.0

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            dof_vel = 1.0

    class viewer(LeggedRobotCfg.viewer):
        pos = [5.0, 0.0, 2.0]
        lookat = [0.0, 0.0, 0.0]

    class noise(LeggedRobotCfg.noise):
        add_noise = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 0.5]
        randomize_base_mass = False
        added_mass_range = [0.0, 0.0]
        push_robots = False
        push_interval_s = 0.0
        max_push_vel_xy = 0.0


class BalanceCarFlatCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "flat_balance_car"
        max_iterations = 500
