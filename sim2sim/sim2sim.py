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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import random
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from ruamel.yaml import YAML

from agent import Agent

import torch
import argparse


import csv
import pandas as pd

def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[-1]
    q_vec = q[:3]
    a = (2.0 * (q_w ** 2) - 1.0) * v
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

# def quaternion_to_euler_array(quat):
#     # Ensure quaternion is in the correct format [x, y, z, w]
#     x, y, z, w = quat
    
#     # Roll (x-axis rotation)
#     t0 = +2.0 * (w * x + y * z)
#     t1 = +1.0 - 2.0 * (x * x + y * y)
#     roll_x = np.arctan2(t0, t1)
    
#     # Pitch (y-axis rotation)
#     t2 = +2.0 * (w * y - z * x)
#     t2 = np.clip(t2, -1.0, 1.0)
#     pitch_y = np.arcsin(t2)
    
#     # Yaw (z-axis rotation)
#     t3 = +2.0 * (w * z + x * y)
#     t4 = +1.0 - 2.0 * (y * y + z * z)
#     yaw_z = np.arctan2(t3, t4)
    
#     # Returns roll, pitch, yaw in a NumPy array in radians
#     return np.array([roll_x, pitch_y, yaw_z]) 


# sim2real: get_obs需要改成实际的观测值
def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    
    name_sensor_pos = ['r_hip_pitch_joint_p', 'r_hip_roll_joint_p', 'r_thigh_joint_p', 'r_calf_joint_p', 'r_ankle_pitch_joint_p', 'r_ankle_roll_joint_p', 
        'l_hip_pitch_joint_p', 'l_hip_roll_joint_p', 'l_thigh_joint_p', 'l_calf_joint_p', 'l_ankle_pitch_joint_p', 'l_ankle_roll_joint_p']
    name_sensor_vel = ['r_hip_pitch_joint_v', 'r_hip_roll_joint_v', 'r_thigh_joint_v', 'r_calf_joint_v', 'r_ankle_pitch_joint_v', 'r_ankle_roll_joint_v', 
        'l_hip_pitch_joint_v', 'l_hip_roll_joint_v', 'l_thigh_joint_v', 'l_calf_joint_v', 'l_ankle_pitch_joint_v', 'l_ankle_roll_joint_v']
    # q = np.array([data.sensor(name).data[0] for name in name_sensor_pos])
    # dq = np.array([data.sensor(name).data[0] for name in name_sensor_vel])
    
    # print("dq:", dq, "qvel:", data.qvel.astype(np.double)[-12:])
    # print("q:", q, "q2:", data.qpos.astype(np.double)[-12:])
    
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    # print("p:", (target_q - q) * kp )
    # print("d", (target_dq - dq) * kd)
    
    # print("target_q: ", target_q)

    
    # print("p: ", (target_q - q) * kp, "d: ", (target_dq - dq) * kd)
    return (target_q- q) * kp + (target_dq - dq) * kd

def run_mujoco(agent, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        agent: The agent used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    target_q_filtered = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
    
    count_lowlevel = 0
    
    # jump_time = 3.0 * np.random.rand() + 2.0
    jump_period = 0.9
    cmd = [jump_period, 2.0, 0]
    world_z = np.array([0, 0, 1.0])
    dq_filtered = np.zeros(12, dtype=np.double)
    
    lag_joint_target_buffer = [np.zeros((cfg.env.num_actions), dtype=np.double) for _ in range(cfg.control.n_lag_action_steps + 1)]
    
    count_csv = 0
    first_flag=1
    with open('sim2sim_robot_states.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # csvwriter.writerow([f'q_{i}' for i in range(19)])
        csvwriter.writerow([
            "sim2sim_base_euler_roll", "sim2sim_base_euler_pitch", "sim2sim_base_euler_yaw",
            # "sim2sim_base_quat_x", "sim2sim_base_quat_y", "sim2sim_base_quat_z", "sim2sim_base_quat_w",
            "sim2sim_dof_pos_0", "sim2sim_dof_pos_1", "sim2sim_dof_pos_2", "sim2sim_dof_pos_3",
            "sim2sim_dof_pos_4", "sim2sim_dof_pos_5", "sim2sim_dof_pos_6", "sim2sim_dof_pos_7",
            "sim2sim_dof_pos_8", "sim2sim_dof_pos_9", "sim2sim_dof_pos_10", "sim2sim_dof_pos_11",
            "sim2sim_target_dof_pos_0", "sim2sim_target_dof_pos_1", "sim2sim_target_dof_pos_2", "sim2sim_target_dof_pos_3",
            "sim2sim_target_dof_pos_4", "sim2sim_target_dof_pos_5", "sim2sim_target_dof_pos_6", "sim2sim_target_dof_pos_7",
            "sim2sim_target_dof_pos_8", "sim2sim_target_dof_pos_9", "sim2sim_target_dof_pos_10", "sim2sim_target_dof_pos_11",
        ])
        
        with torch.no_grad():

            for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
                # Obtain an observation
                
                
                
                q, dq, quat, v, omega, gvec = get_obs(data)
                q = q[-cfg.env.num_actions:]
                dq = dq[-cfg.env.num_actions:]
                
                sim_time = count_lowlevel * cfg.sim_config.dt
                
                    
                for i in range(6):
                    tmpq = q[i]
                    q[i] = q[i+6]
                    q[i+6] = tmpq

                    tmpdq = dq[i]
                    dq[i] = dq[i+6]
                    dq[i+6] = tmpdq
                    
                dq_filtered = 1.0 * dq + 0.0 * dq_filtered

                # 1000hz -> 50hz
                if count_lowlevel % cfg.sim_config.decimation == 0:
                    obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                    # eu_ang = quaternion_to_euler_array(quat)
                    # eu_ang[eu_ang > math.pi] -= 2 * math.pi
                    body_ori = quat_rotate_inverse(quat, world_z)

                    obs[0, 0:3] = body_ori
                    obs[0, 3:15] = q
                    obs[0, 15:27] = dq_filtered * cfg.normalization.obs_scales.dof_vel
                    obs[0, 27:39] = action
                    obs[0, 39] = (sim_time / jump_period + 0.80) % 1.0
                    
                    obs[0, 40:43] = cmd

                    # obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                    if first_flag:
                        for _ in range(cfg.env.frame_stack):
                            hist_obs.append(obs)
                            hist_obs.popleft()
                        first_flag = 0
                    hist_obs.append(obs)
                    hist_obs.popleft()

                    policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                    for i in range(cfg.env.frame_stack):
                        policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
                    # print("obs:",policy_input)
                    action[:] = agent.getAction(torch.Tensor(policy_input).to(agent.device), True).detach().cpu().numpy()
                    # action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                    # target_q = action * 0
                    target_q = action * cfg.control.action_scale
                    
                    # if count_csv < 500:
                    #     csv_q = np.zeros(27)
                    #     csv_ori = quat_rotate_inverse(quat, world_z)
                    #     # csv_euler_ang = quaternion_to_euler_array(q[3:7])
                    #     csv_q[0:3] = csv_ori
                    #     csv_q[3:15] = q[:]
                    #     csv_q[15:] = target_q[:]
                    #     csvwriter.writerow(csv_q.tolist())
                    # count_csv += 1

                    target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
                    
                    # Generate PD control
                    target_q_filtered = cfg.control.action_smooth_weight*target_q + (1.0 - cfg.control.action_smooth_weight)*target_q_filtered
                
                lag_joint_target_buffer = lag_joint_target_buffer[1:] + [target_q_filtered]
                joint_targets = lag_joint_target_buffer[0]
                
                tau = pd_control(joint_targets, q, cfg.robot_config.kps,
                                target_dq, dq, cfg.robot_config.kds)  # Calc torques
                # print("tau:", tau)

                tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
                for i in range(6):
                    tmptau = tau[i]
                    tau[i] = tau[i+6]
                    tau[i+6] = tmptau
                data.ctrl = tau
                # print("action:", action)

                
                # sim2real: tau需要给到电机
                # <--------------------------
                mujoco.mj_step(model, data)
                viewer.render()
                # --------------------------->
                count_lowlevel += 1
                

    viewer.close()


if __name__ == '__main__':
    
    # seed = 1
    
    # np.random.seed(seed)
    # random.seed(seed)    
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Deactuatorvelployment script.')
    parser.add_argument('--device_type', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    parser.add_argument('--model_num', type=int, default=0, help='num model.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg():

        class sim_config:
            
            mujoco_model_path = f'../assets/robot_urdf/mjcf/pi_12dof_release_v1.xml'
            sim_duration = 60.0
            dt = 0.0001
            control_dt = 0.02
            decimation = control_dt / dt # control_dt = 0.02s

        class robot_config:
            kps = np.array([36]*(12), dtype=np.double)
            kds = np.array([1.03]*(12), dtype=np.double)
            tau_limit = 21 * np.ones(12, dtype=np.double)
            
            
        class env:
            num_actions = 12
            frame_stack = 15
            num_single_obs = 3 + 12*3 + 1 + 3
            num_observations = (3 + 12*3 + 1 + 3) * 15
            
        class normalization:
            class obs_scales:
                lin_vel = 1.
                ang_vel = 1.
                dof_pos = 1
                dof_vel = 1
                quat = 1.
            clip_observations = 100.
            clip_actions = 100.
            
        class control:
            action_scale = 0.3
            action_smooth_weight = 0.3
            n_lag_action_steps = 5
    
    args.name = 'sim2sim'
    # deviceresults/piforjump_student/seed_1_student_Nov06_11-06-44/checkpoint/model_100001792.pt
    if torch.cuda.is_available() and args.device_type == 'gpu':
        device_name = f'cuda:{args.gpu_idx}'
        print('[torch] cuda is used.')
    else:
        device_name = 'cpu'
        print('[torch] cpu is used.')
    args.device = device_name
    args.obs_dim = Sim2simCfg().env.num_observations
    args.action_dim = Sim2simCfg().env.num_actions
    args.action_bound_min = -np.ones(args.action_dim)
    args.action_bound_max = np.ones(args.action_dim)
    args.history_len = Sim2simCfg().env.frame_stack
    with open('comoppo_model.yaml', 'r') as f:
        algo_cfg = YAML().load(f)
    for key in algo_cfg.keys():
        args.__dict__[key] = algo_cfg[key]
    agent = Agent(args)
    agent.load(args.model_num)
    
    agent.obs_rms.save_yaml(args.model_num) # sim2real: uncomment this line to save the mean and var to csv file
    run_mujoco(agent, Sim2simCfg())
