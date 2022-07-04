from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import torch as th

@dataclass(frozen=True)
class Trajectory:
    """A trajectory, e.g. a one episode rollout from an expert policy."""
    
    observations: np.ndarray
    """Observations, shape (trajectory_len, ) + observation_shape."""

    actions: np.ndarray
    """Actions, shape (trajectory_len, ) + action_shape."""

    next_observations: np.ndarray
    """Next observations, shape (trajectory_len, ) + observation_shape"""

    log_probs: np.ndarray
    """Action log probabilities, shape (trajectory_len, )."""

    infos: Dict[str, np.ndarray]
    """Infos, shape (trajectory_len, val_dim)"""


    def __len__(self):
        """Returns number of transitions, equal to the number of actions."""
        return len(self.actions)

    def __post_init__(self):
        """Performs input validation: check shapes are as specified in docstring."""
        if len(self.observations) != len(self.actions):
            raise ValueError(
                f"Observations {len(self.observations)}, actions {len(self.actions)} ",
            )
        if len(self.actions) == 0:
            raise ValueError("Degenerate trajectory: must have at least one action.")

        if self.next_observations is not None and len(self.observations) != len(self.next_observations):
            raise ValueError(
                f"Observations {len(self.observations)}, next observations {len(self.next_observations)} ",
            )
        if self.infos is not None:
            for key, val in self.infos.items():
                if len(val) != len(self.actions):
                    print(val.shape)
                    raise ValueError(f"Infos shape {len(val)} does not match actions {len(self.actions)}")

        if self.log_probs is not None:
            if len(self.log_probs) != len(self.actions):
                raise ValueError(
                    f"Action log_probs shape {len(self.log_probs)} does not match actions {len(self.actions)}"
                )

        if self.infos is not None:
            assert isinstance(self.infos, dict)
            for k, v in self.infos.items():
                if len(v) != len(self.actions):
                    raise ValueError(f"Info {k} shape {len(v)} does not match actions {len(self.actions)}")



@dataclass(frozen=True)
class TrajectoryWithReward(Trajectory):
    """A `Trajectory` that additionally includes reward information."""

    rewards: np.ndarray
    """Reward, shape (trajectory_len, ). dtype float."""

    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()

        if self.rewards.shape != (len(self.actions),):
            raise ValueError(
                "rewards must be 1D array, one entry for each action: "
                f"{self.rewards.shape} != ({len(self.actions)},)",
            )

@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    """Observation, shape (observation_shape, )."""

    action: np.ndarray
    """Action, shape (action_shape, )."""

    # next_observation: np.ndarray
    """Next observation, shape (observation_shape, )."""

    reward: np.ndarray
    """Reward, shape (1, )."""

    log_prob: np.ndarray
    """Action log probability, shape (1, ) """

    info: Dict[str, np.ndarray]
    """Info"""


    def __len__(self):
        """Length of a transition is always 1"""
        return 1

    def __post_init__(self):
        """Performs input validation: check shapes are as specified in docstring."""
        if len(self.observation.shape) > 1 or len(self.action.shape) > 1:
            raise ValueError(
                "Initialiazed more than one transition"
            )
        # if self.observation.shape != self.next_observation.shape:
        #     raise ValueError(
        #         "Observations have different dimensions in one transition"
        #     )


def convert_trajectories_to_transitions(trajectories: List[Trajectory]) -> List[Transition]:
    """Flatten a series of trajectories to a series of transitions"""
    assert len(trajectories) >= 1, "Cannot convert empty trajectory"

    transitions = []
    for traj in trajectories:
        for i in range(len(traj)):
            if traj.infos is None:
                info = None
            else:
                info = {}
                for key, val in traj.infos.items():
                    info[key] = val[i]

            if traj.log_probs is None:
                log_prob = None
            else:
                log_prob = traj.log_probs[i]
            
            transition = Transition(
                observation=traj.observations[i], 
                action=traj.actions[i], 
                reward=traj.rewards[i],
                # next_observation=traj.next_observations[i], 
                info=info,
                log_prob=log_prob
            )
            transitions.append(transition)

    assert len(transitions) == sum([len(traj) for traj in trajectories]), (
        "Number of transitions does not match after conversion"
    )
    return transitions