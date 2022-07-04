from typing import List, Optional

import numpy as np

from lapal.utils import types 

class ReplayBuffer:

    def __init__(self, max_size=1024):
        self.max_size = max_size        # Maximum number of episodes to store in buffer
        self.trajectories = []
        self.transitions = []

    def reset(self):
        self.trajectories = []
        self.transitions = [] 

    def __len__(self):
        return len(self.transitions)       

    def add_rollouts(self, trajectories: List[types.Trajectory]):
        """Add trajectories to buffer"""
        self.trajectories.extend(trajectories)
        transitions = types.convert_trajectories_to_transitions(trajectories)
        self.transitions.extend(transitions)

        if len(self.trajectories) > self.max_size:
            self.trajectories = self.trajectories[-self.max_size:]

        if len(self.transitions) > self.max_size*len(self.trajectories[0]):
            self.transitions = self.transitions[-self.max_size*len(self.trajectories[0]):]

        print(f"Replay buffer contains {len(self.trajectories)} trajectories of {len(self.transitions)} transitions")

    def sample_random_transitions(self, batch_size: int) -> List[types.Transition]:
        assert batch_size <= len(self.transitions), (
            "Sampling batch size larger than transitions in replay buffer"
        )
        rand_indices = np.random.permutation(len(self.transitions))[:batch_size]
        return [self.transitions[i] for i in rand_indices]

    def sample_recent_transitions(self, batch_size: int) -> List[types.Transition]:
        assert batch_size <= len(self.transitions), (
            "Sampling batch size larger than transitions in replay buffer"
        )
        return self.transitions[-batch_size:]

    def sample_random_trajectories(self, batch_size: int) -> List[types.Trajectory]:
        assert batch_size <= len(self.trajectories), (
            "Sampling batch size larger than trajectories in replay buffer"
        )
        rand_indices = np.random.permutation(len(self.trajectories))[:batch_size]
        return [self.trajectories[i] for i in rand_indices]

    def sample_recent_trajectories(self, batch_size: int) -> List[types.Trajectory]:
        assert batch_size <= len(self.trajectories), (
            "Sampling batch size larger than trajectories in replay buffer"
        )
        return self.trajectories[-batch_size:]

    def sample_trajectory_chunks(
        self, 
        batch_size: int, 
        horizon: Optional[int] = 1
    ) -> List[types.Trajectory]:

        num_trajs = len(self.trajectories)
        num_steps = len(self.trajectories[0]) - horizon     # Assume all trajectories have same length

        traj_chunks = []
        for i in range(batch_size):
            traj_ind, step_ind = np.random.randint([num_trajs, num_steps])
            traj = self.trajectories[traj_ind]
            traj_chunks.append(
                types.TrajectoryWithReward(
                    observations=traj.observations[step_ind:step_ind+horizon],
                    actions=traj.actions[step_ind:step_ind+horizon],
                    next_observations=None,
                    log_probs=None,
                    infos={k: v[step_ind:step_ind+horizon] for k, v in traj.infos.items()},
                    rewards=traj.rewards[step_ind:step_ind+horizon],
                )
            )
        return traj_chunks