from typing import Optional, Union, Any, Tuple, List, Callable, Dict

import gym
import numpy as np
import torch as th

import gym 
from gym.envs import registry

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from lapal.utils import pytorch_utils as ptu
from lapal.utils import types

def sample_trajectories(
    venv: VecEnv,
    policy: Union[OffPolicyAlgorithm, Any],
    num_trajectories: int,
    deterministic: Optional[bool] = True,
    ac_decoder: Callable[[np.ndarray], np.ndarray] = None,

) -> List[types.TrajectoryWithReward]:
    """
    Currently only works for fixed horizon envs, all envs should return done=True 
    """

    num_envs = venv.num_envs
    paths = []
    for i in range(num_trajectories // num_envs):
        obs, acs, rewards, next_obs, infos = [], [], [], [], []

        done = [False]
        ob = venv.reset()
        while not any(done):

            ac, _ = policy.predict(ob, deterministic=deterministic)

            if ac_decoder is not None:
                ob_tensor = ptu.from_numpy(ob)
                ac_tensor = ptu.from_numpy(ac)
                ac = ptu.to_numpy(ac_decoder(ob_tensor, ac_tensor))

            next_ob, reward, done, info = venv.step(ac)

            obs.append(ob)
            acs.append(ac)
            rewards.append(reward)
            next_obs.append(next_ob)
            infos.append(info)

            ob = next_ob

        obs = np.stack(obs, axis=0)
        acs = np.stack(acs, axis=0)
        next_obs = np.stack(next_obs, axis=0)
        rewards = np.stack(rewards, axis=0)

        for j in range(num_envs):
            if isinstance(infos[0][0], dict):
                keys = infos[0][0].keys()
                new_infos = {}
                for key in keys:
                    new_infos[key] = np.array([var[j][key] for var in infos])
            else:
                new_infos = None

            paths.append(
                types.TrajectoryWithReward(
                    observations=obs[:,j], 
                    actions=acs[:,j], 
                    next_observations=next_obs[:,j],
                    rewards=rewards[:,j],
                    infos=new_infos,
                    log_probs=None
                )
            )
    return paths


def check_demo_performance(paths):
    assert type(paths[0]) == types.TrajectoryWithReward, "Demo path type is not types.TrajectoryWithReward"
    returns = [path.rewards.sum() for path in paths]
    lens = [len(path) for path in paths]
    print(f"Collected {len(returns)} expert demonstrations")
    print(f"Demonstration length {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
    print(f"Demonstration return {np.mean(returns):.2f} +/- {np.std(returns):.2f}")


def collect_demo_trajectories(env: gym.Env, expert_policy: str, batch_size: int):
    expert_policy = SAC.load(expert_policy)
    print('\nRunning expert policy to collect demonstrations...')
    demo_paths = sample_trajectories(env, expert_policy, batch_size)
    check_demo_performance(demo_paths)
    return demo_paths

def collect_d4rl_trajectories(env: gym.Env, batch_size: int):
    dataset = env.get_dataset()
    timeouts = np.where(dataset['timeouts'])[0]
    demo_paths = []
    
    start = 0
    for i in range(batch_size):
        end = timeouts[i] + 1
        observations = dataset['observations'][start:end]
        actions = dataset['actions'][start:end]
        next_observations = np.zeros_like(observations)     # we don't use next observations in GAIL
        rewards = dataset['rewards'][start:end]
        demo_paths.append(types.TrajectoryWithReward(
            observations=observations, 
            actions=actions, 
            next_observations=next_observations,
            rewards=rewards,
            infos=None,
            log_probs=None
        ))
        start = end
    check_demo_performance(demo_paths)
    return demo_paths    


def extract_paths(paths: List[types.Trajectory]) -> List[th.Tensor]:
    obs = ptu.from_numpy(np.array([path.observations for path in paths]))
    # Drop the last terminal state
    obs = obs[:, :-1, :]
    act = ptu.from_numpy(np.array([path.actions for path in paths]))
    if paths[0].log_probs is not None:
        log_probs = ptu.from_numpy(np.array([path.log_probs for path in paths]))
    else: 
        log_probs = None
    assert obs.shape[0] == act.shape[0], (
        "Batch size is not same for extracted paths"
    )
    assert obs.shape[1] == act.shape[1], (
        "Episode length is not same for extracted paths"
    )
    return obs, act, log_probs

def extract_transitions(transitions: List[types.Transition]) -> List[th.Tensor]:
    obs = ptu.from_numpy(np.array([transition.observation for transition in transitions]))
    act = ptu.from_numpy(np.array([transition.action for transition in transitions]))
    if transitions[0].log_prob is not None:
        log_probs = ptu.from_numpy(np.array([transition.log_prob for transition in transitions]))
    else: 
        log_probs = None
    assert obs.shape[0] == act.shape[0], (
        "Batch size is not same for extracted paths"
    )
    return obs, act, log_probs

def log_metrics(logger, metrics: Dict[str, np.ndarray], namespace: str):
    for k, v in metrics.items():
        if v.ndim < 1 or (v.dim == 1 and v.shape[0] <= 1):
            logger.record_mean(f"{namespace}/{k}", v)
        else:
            logger.record_mean(f"{namespace}/{k}Max", th.amax(v).item())
            logger.record_mean(f"{namespace}/{k}Min", th.amin(v).item())
            logger.record_mean(f"{namespace}/{k}Mean", th.mean(v).item())
            logger.record_mean(f"{namespace}/{k}Std", th.std(v).item())

def get_gym_env_type(env_name):
    if env_name not in registry.env_specs:
        raise ValueError("No such env")
    entry_point = registry.env_specs[env_name].entry_point
    if entry_point.startswith("gym.envs."):
        type_name = entry_point[len("gym.envs."):].split(":")[0].split('.')[0]
    else:
        type_name = entry_point.split('.')[0]
    return type_name