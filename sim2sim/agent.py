from common.actor_gaussian import ActorGaussian as Actor

from normalizer import ObsRMS

import numpy as np
import torch
import os
import copy

EPS = 1e-8
torch.set_printoptions(precision=3,linewidth=200, sci_mode=False)
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
        
        last_obs = obs_tensor[0][-43:].detach().cpu().numpy()
        print("----------------------------------------------")
        # print("obs_tensor:\n", obs_tensor)
        # print("norm_obs_tensor:\n", norm_obs_tensor)
        print("obs_ori:", last_obs[0:3])
        print("obs_q:", last_obs[3:15])
        print("obs_dq:", last_obs[15:27])
        print("obs_action:", last_obs[27:39])
        print("obs_phase:", last_obs[39])
        print("obs_command:", last_obs[40:43])
        
        epsilon_tensor = torch.randn(norm_obs_tensor.shape[:-1] + (self.action_dim,), device=self.device)
        self.actor.updateActionDist(norm_obs_tensor, epsilon_tensor)
        _, unnorm_action_tensor = self.actor.sample(deterministic)
        # print("unnorm action: ", unnorm_action_tensor)
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
            self.save(model_num)
            return int(model_num)
        else:
            self.actor.initialize()
            print(f'[{self.name}] load fail.')
            return 0
        
    def save(self, model_num):

        example_input = torch.randn(self.obs_dim, device="cpu")
        path = "checkpoint/pi_forjump_1_policy.pt"
        actor_copy = copy.deepcopy(self.actor).to("cpu")
        actor_copy.eval()
        
        
        
        traced_script_module = torch.jit.trace(actor_copy, example_input)
        traced_script_module.save(path)
        
        model = torch.jit.load(path)
        # print("save success. \ntest output: ", model(example_input), "\nmodel output: ", self.actor(example_input))
        
        
        # torch.save(self.actor.model, f"checkpoint/model_full_{model_num}.pt")

    ################
    # private method
    ################
