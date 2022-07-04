import os
import os.path as osp

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from lapal.utils.replay_buffer import ReplayBuffer
from lapal.utils import utils, types
import lapal.utils.pytorch_utils as ptu

from rl_plotter.logger import Logger

class LAPAL_Agent:
    def __init__(self, 
        params, 
        venv, 
        ac_vae, 
        disc, 
        policy, 
        logger, 
    ):
        self.params = params

        self.logger = logger  
        self.venv = venv
        self.ac_vae = ac_vae
        self.disc = disc
        self.policy = policy 

        self.demo_buffer = ReplayBuffer()

        # bool
        self.use_disc = params['discriminator']['use_disc']
        self.use_ac_vae = params['ac_vae']['use_ac_vae']

        # plot
        self.rl_logger = Logger(
            log_dir='./logs',
            exp_name=params['suffix'],
            env_name=params['env_name'],
            seed=params['seed']
        )

    def train(self):
        # Run expert policy to collect demonstration paths

        if self.use_disc or self.use_ac_vae:
            demo_paths = utils.collect_demo_trajectories(
                self.venv,
                self.params['expert_policy'], 
                self.params['demo_size']
            )
            self.demo_buffer.add_rollouts(demo_paths)
            if self.use_ac_vae: 
                self.pretrain_ac_vae() 

        # Warm up generator replay buffer
        self.policy.learn(total_timesteps=self.params['generator']['learning_starts'])

        self.timesteps = 0
        while self.timesteps < self.params['total_timesteps']:
            self.train_generator()

            if self.use_disc:
                self.train_discriminator()

            self.timesteps += self.params['generator']['batch_size']

            # Evaluation
            if self.timesteps % self.params['evaluation']['interval'] < self.params['generator']['batch_size']:
                self.perform_logging(self.policy)
            self.logger.dump(step=self.timesteps)

            if self.timesteps % self.params['evaluation']['save_interval'] < self.params['generator']['batch_size']:
                self.save_models()
        

    def pretrain_ac_vae(self):
        batch_size = self.params['ac_vae']['batch_size']
        
        for i in range(self.params['ac_vae']['n_iters']):
            demo_transitions = self.demo_buffer.sample_random_transitions(batch_size)
            obs = ptu.from_numpy(np.stack([t.observation for t in demo_transitions], axis=0))
            acs = ptu.from_numpy(np.stack([t.action for t in demo_transitions], axis=0))
            metrics = self.ac_vae.train(obs, acs)

            if (i + 1) % 1000 == 0:
                for k, v in metrics.items():
                    self.logger.record(f"ac_vae/{k}", v)
            else:
                for k, v in metrics.items():
                    self.logger.record(f"ac_vae/{k}", v, exclude='stdout')
            
            self.logger.dump(step=i)


    def train_generator(self):
        """
        Train the policy/actor using learned reward
        """
        self.policy.learn(
            total_timesteps=self.params['generator']['batch_size'], 
            reset_num_timesteps=False
        )


    def train_discriminator(self):
        batch_size = self.params['discriminator']['batch_size']  
        train_args = ()

        # Demo buffer contains ob, ac in original space
        demo_transitions = self.demo_buffer.sample_random_transitions(batch_size)
        demo_obs, demo_acs, _ = utils.extract_transitions(demo_transitions)
        train_args += (demo_obs,)
        if self.use_ac_vae:
            demo_lat_acs = self.ac_vae.ac_encoder(demo_obs, demo_acs)
            train_args += (demo_lat_acs,)
        else:
            train_args += (demo_acs,)

        # Agent buffer contains ob, ac in original space
        agent_transitions = self.policy.replay_buffer.sample(batch_size)
        agent_obs = agent_transitions.observations.float()
        agent_acs = agent_transitions.actions.float()
        train_args += (agent_obs,)
        if self.use_ac_vae:
            agent_lat_acs = self.ac_vae.ac_encoder(agent_obs, agent_acs)
            train_args += (agent_lat_acs,)
        else:
            train_args += (agent_acs,)

        metrics = self.disc.train(*train_args)

        for k, v in metrics.items():
            self.logger.record(f"disc/{k}", v)

    def perform_logging(self, eval_policy):

        #######################
        # Evaluate the agent policy in true environment
        print("\nCollecting data for eval...")
        eval_kwargs = {}
        if self.use_ac_vae:
            eval_kwargs.update(dict(ac_decoder=self.ac_vae.ac_decoder))
        eval_paths = utils.sample_trajectories(
            self.venv, 
            eval_policy, 
            self.params['evaluation']['batch_size'],
            **eval_kwargs
        )  

        eval_returns = [path.rewards.sum() for path in eval_paths]
        eval_ep_lens = [len(path) for path in eval_paths]

        logs = {}
        logs["Eval/AverageReturn"] = np.mean(eval_returns)
        logs["Eval/StdReturn"] = np.std(eval_returns)
        logs["Eval/MaxReturn"] = np.max(eval_returns)
        logs["Eval/MinReturn"] = np.min(eval_returns)
        logs["Eval/AverageEpLen"] = np.mean(eval_ep_lens)

        for key, value in logs.items():
            self.logger.record(key, value)

        self.rl_logger.update(score=eval_returns, total_steps=self.timesteps)

    def save_models(self):   

        folder = self.params['logdir'] + f"/checkpoints/{self.timesteps:06d}"
        os.makedirs(folder, exist_ok=True) 

        th.save(self.disc.state_dict(), osp.join(folder, "disc.pt"))
        self.policy.save(osp.join(folder, "policy.pt"))
        if self.ac_vae is not None:
            th.save(self.ac_vae.state_dict(), osp.join(folder, "ac_vae.pt"))

    def load_models(self, folder):

        self.disc.load_state_dict(th.load(folder + "/disc.pt"))
        self.policy = SAC.load(folder + "/policy.pt")
        
        if self.ac_vae is not None:
            self.ac_vae.load_state_dict(th.load(folder + "/ac_vae.pt"))
