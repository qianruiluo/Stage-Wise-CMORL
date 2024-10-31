from common.actor_gaussian import ActorGaussian as Actor

from normalizer import ObsRMS

import numpy as np
import torch
import os

EPS = 1e-8

class Agent:
    def __init__(self, args) -> None:
        # for base
        self.name = args.name
        self.device = args.device

        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.action_bound_min = args.action_bound_min
        self.action_bound_max = args.action_bound_max



        # for normalization
        self.history_len = args.history_len
        self.obs_rms = ObsRMS('obs', self.obs_dim, self.history_len, self.device)

        # declare actor
        model_cfg = args.model
        print("model cfg:", model_cfg)
        self.actor = Actor(
            self.device, self.obs_dim, self.action_dim, self.action_bound_min, 
            self.action_bound_max, model_cfg['actor']).to(self.device)


    ################
    # Public Methods
    ################

    @torch.no_grad()
    def getAction(self, obs_tensor:torch.tensor, deterministic:bool) -> torch.tensor:
        
        norm_obs_tensor = self.obs_rms.normalize(obs_tensor) 
        
        last_obs = obs_tensor[0][-46:].detach().cpu().numpy()
        print("----------------------------------------------")
        print("obs_ori:", last_obs[0:3])
        print("obs_q:", last_obs[3:15])
        print("obs_dq:", last_obs[15:27])
        print("obs_action:", last_obs[27:39])
        print("obs_phase:", last_obs[39:43])
        print("obs_command:", last_obs[43:46])
        
        
        
        epsilon_tensor = torch.randn(norm_obs_tensor.shape[:-1] + (self.action_dim,), device=self.device)
        self.actor.updateActionDist(norm_obs_tensor, epsilon_tensor)
        _, unnorm_action_tensor = self.actor.sample(deterministic)
        return unnorm_action_tensor
    
    def copyObsRMS(self, obs_rms):
        self.obs_rms.mean[:] = obs_rms.mean
        self.obs_rms.var[:] = obs_rms.var
        self.obs_rms.upgrade()

    def load(self, model_num):
        # load rms
        self.obs_rms.load(model_num)

        # load network models
        checkpoint_file = f"checkpoint/model_{model_num}.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            print(f'[{self.name}] load success.')
            return int(model_num)
        else:
            self.actor.initialize()
            print(f'[{self.name}] load fail.')
            return 0

    ################
    # private method
    ################