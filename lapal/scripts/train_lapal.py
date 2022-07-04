import sys, os
import os.path as osp
import time
from ruamel.yaml import YAML

import gym 
import robosuite as suite
from robosuite.wrappers import GymWrapper

from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import SAC

from lapal.agents.discriminator import Discriminator
from lapal.agents.lapal_agent import LAPAL_Agent
from lapal.agents.vae import CVAE
from lapal.utils import utils
import lapal.utils.pytorch_utils as ptu


def make_robosuite_env():
    env = suite.make(
        env_name="Door", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    env = GymWrapper(env)
    return env


def build_venv(env_name, n_envs, norm_obs=False, wrapper=None, wrapper_kwargs=None):
    """
    Make vectorized env and add env wrappers
    """
    if env_name in ["Door"]:
        env = make_vec_env(
            make_robosuite_env, 
            vec_env_cls=SubprocVecEnv, 
            n_envs=n_envs,
            wrapper_class=wrapper, 
            wrapper_kwargs=wrapper_kwargs
        )  
        return env

    if env_name in ['Hoppper-v3', 'Walker2d-v3', 'Ant-v3', 'Humanoid-v3']:
        env_kwargs = dict(terminate_when_unhealthy=False)
    else:
        env_kwargs = None

    if utils.get_gym_env_type(env_name) == 'mujoco':
        venv = make_vec_env(
            env_name, 
            n_envs=n_envs, 
            env_kwargs=env_kwargs,
            wrapper_class=wrapper, 
            wrapper_kwargs=wrapper_kwargs
        ) 
    else:
        raise ValueError('Environment {} not supported yet ...'.format(env_name))

    return venv


def main():

    yaml = YAML(typ='safe')
    params = yaml.load(open(sys.argv[1]))

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################
    model_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../expert_models/', params['expert_policy']))
    assert osp.exists(model_path + '.zip'), f"Trained expert model not saved as {model_path}"
    params['expert_policy'] = model_path

    data_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../data'))
    logdir = params['env_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = osp.join(data_path, logdir)
    if params['suffix'] is None:
        params['suffix'] = params['discriminator']['reward_type'] + '_' + params['generator']['type']
    logdir += '_' + params['suffix']
    params['logdir'] = logdir

    print(params)

    # dump params
    os.makedirs(logdir, exist_ok=True)
    import yaml
    with open(osp.join(logdir, 'params.yml'), 'w') as fp:
        yaml.safe_dump(params, fp, sort_keys=False)

    ##################################
    ### SETUP ENV, DISCRIMINATOR, GENERATOR
    ##################################
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['which_gpu'])

    if params['env_name'] == 'Door':
        env = make_robosuite_env()
    else:
        env = gym.make(params['env_name'])
    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]

    ################################################
    # Action encoder/decoder
    ################################################
    ac_vae_params = params['ac_vae']
    if ac_vae_params['use_ac_vae']:
        ac_vae = CVAE(
            params['ob_dim'],
            params['ac_dim'],
            ac_vae_params['latent_dim'],
            lr=ac_vae_params['learning_rate'],
            kl_coef=ac_vae_params['kl_coef'],
        )
    else:
        ac_vae = None

    ################################################
    # Discriminator
    ################################################
    disc_params = params['discriminator']
    disc_input_size = params['ob_dim']
    if ac_vae_params['use_ac_vae']:
        disc_input_size += ac_vae_params['latent_dim']
    else:
        disc_input_size += params['ac_dim']

    disc = Discriminator(
        input_size=disc_input_size,
        learning_rate=disc_params['learning_rate'],
        batch_size=disc_params['batch_size'],
        reward_type=disc_params['reward_type'],
    )

    ################################################
    # Environment
    ################################################
    # SubprocVecEnv must be wrapped in if __name__ == "__main__":
    venv = build_venv(params['env_name'], params['n_envs'])
    venv.seed(params['seed'])
    logger = configure(params['logdir'], ["stdout", "csv", "log", "tensorboard"])

    ################################################
    # Generator
    ################################################
    gen_params = params['generator']
    policy_kwargs = {}
    if disc_params['use_disc']:
        policy_kwargs.update(dict(reward=disc.reward))
    if ac_vae_params['use_ac_vae']:
        policy_kwargs.update(
            dict(
                latent_ac_dim=ac_vae_params['latent_dim'],
                ac_encoder=ac_vae.ac_encoder,
                ac_decoder=ac_vae.ac_decoder
            )
        )

    policy = SAC(
        "MlpPolicy",
        venv,
        learning_rate=gen_params['learning_rate'],
        buffer_size=1000000,
        learning_starts=gen_params['learning_starts'],
        batch_size=gen_params['batch_size'],
        gradient_steps=gen_params['gradient_steps'],
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=params['seed'],
        **policy_kwargs
    )
    policy.set_logger(logger)

    print(f"Environment state space dimension: {params['ob_dim']}, action space dimension: {params['ac_dim']}")
    print(f"SAC policy state space dimension: {policy.policy.observation_space.shape[0]}, action space dimension: {policy.policy.action_space.shape[0]}")
    print(f"Discriminator input size: {disc_input_size}")

    ###################
    ### RUN TRAINING
    ###################

    irl_model = LAPAL_Agent(
        params, 
        venv, 
        ac_vae, 
        disc, 
        policy, 
        logger
    )
    irl_model.train()

if __name__ == '__main__':
    main()