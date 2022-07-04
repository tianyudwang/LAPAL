import argparse
import os
import os.path as osp
import numpy as np

import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from lapal.utils import utils

def build_env(env_name, n_envs, env_kwargs=None, wrapper=None, wrapper_kwargs=None):
    """
    Make env and add env wrappers
    """

    if env_name in ["Door"]:
        import robosuite as suite
        from robosuite.wrappers import GymWrapper

        def make_env():
            env = suite.make(
                env_name=env_name, # try with other tasks like "Stack" and "Door"
                robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
                reward_shaping=True,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
            )
            env = GymWrapper(env)
            return env
        env = make_vec_env(make_env, vec_env_cls=SubprocVecEnv, n_envs=n_envs, wrapper=wrapper, wrapper_kwargs=wrapper_kwargs)  
        return env


    if env_name in ['Hoppper-v3', 'Walker2d-v3', 'Ant-v3', 'Humanoid-v3']:
        env_kwargs = dict(terminate_when_unhealthy=False)
    else:
        env_kwargs = None

    if utils.get_gym_env_type(env_name) == 'mujoco':
        env = make_vec_env(
            env_name, 
            n_envs=n_envs, 
            env_kwargs=env_kwargs,
            wrapper_class=wrapper, 
            wrapper_kwargs=wrapper_kwargs
        )

    else:
        raise ValueError('Environment {} not supported yet ...'.format(env_name))
    return env

def train_policy(env, algo, policy_name, resume_from=None, timesteps=50000):
    """
    Train the expert policy in RL
    """
    if algo == 'SAC':
        from stable_baselines3 import SAC

        model_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../expert_models'))
        os.makedirs(model_path, exist_ok=True)

        if resume_from is not None:
            model = SAC.load(resume_from, env=env, print_system_info=True)
        else:
            model = SAC("MlpPolicy", env, gradient_steps=1, verbose=1)
        
        from stable_baselines3.common.logger import configure
        data_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../data'))
        tmp_path = data_path + f'/{policy_name}'
        # set up logger
        new_logger = configure(tmp_path, ["stdout", "csv", "log", "json", "tensorboard"])
        model.set_logger(new_logger)
        model.learn(total_timesteps=timesteps, log_interval=4)

        policy_name = model_path + f'/{policy_name}'
        model.save(policy_name)
    else:
        raise ValueError('RL algorithm {} not supported yet ...'.format(algo))
    return model




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalCheetah-v3')
    parser.add_argument('--algo', type=str, default='SAC')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--total_timesteps', type=int, default=6000000)
    parser.add_argument('--n_envs', type=int, default=8)
    args = parser.parse_args()
    
    env = build_env(args.env_name, args.n_envs)

    model = train_policy(env, args.algo, policy_name, resume_from=args.resume_from, timesteps=args.total_timesteps)


if __name__ == '__main__':

    main()