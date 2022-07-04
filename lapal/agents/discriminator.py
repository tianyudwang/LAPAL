from typing import Optional, List, Dict, Any

import numpy as np
import torch as th
from torch import nn
from torch import optim

from lapal.utils import utils, types
import lapal.utils.pytorch_utils as ptu

class Discriminator(nn.Module):
    def __init__(
        self, 
        input_size: int,
        activation: Optional[str] = 'tanh',
        n_layers: Optional[int] = 2,
        size: Optional[int] = 256,
        learning_rate: Optional[float] = 3e-4,
        batch_size: Optional[int] = 256,
        reward_type: Optional[str] = 'GAIL',
        max_logit: Optional[float] = 10.0,
        clip_reward_range: Optional[float] = -1.0,
        max_grad_norm: Optional[float] = 10.0,
    ):
        super().__init__()

        self.reward_type = reward_type
        self.device = ptu.device
        self.batch_size = batch_size
        self.max_logit = max_logit
        self.max_grad_norm = max_grad_norm

        self.model = ptu.build_mlp(
            input_size=input_size,
            output_size=1,
            n_layers=n_layers,
            size=size,
            activation=activation,
            output_activation='identity',
        ).to('cuda') 

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.loss = nn.BCELoss()
        
        # D = 0 for expert and D = 1 for agent
        demo_labels = th.zeros((batch_size, 1), device=self.device)
        agent_labels = th.ones((batch_size, 1), device=self.device) 
        self.labels = th.cat((demo_labels, agent_labels), dim=0)


    def forward(self, state: th.Tensor, action: th.Tensor):
        logits = self.model(th.cat([state, action], dim=-1))
        logits = th.clamp(logits, -self.max_logit, self.max_logit)
        return logits

    def reward(self, states: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """Recompute reward after collected rollouts for off-policy algorithm"""
        with th.no_grad():
            logits = self(states.float(), actions.float())
            if self.reward_type == 'GAIL':
                rewards = -th.log(self.sigmoid(logits))
            elif self.reward_type == 'SOFTPLUS':
                rewards = -self.softplus(logits)
            elif self.reward_type == 'AIRL':
                rewards = -logits
            else:
                assert False
        return rewards


    def train(
        self,
        demo_states: th.Tensor,
        demo_actions: th.Tensor,
        agent_states: th.Tensor,
        agent_actions: th.Tensor
    ) -> Dict[str, th.Tensor]:  
        assert demo_states.dim() == 2
        assert demo_states.shape[0] == self.batch_size

        states = th.cat([demo_states, agent_states], dim=0)
        actions = th.cat([demo_actions, agent_actions], dim=0)
        logits = self(states, actions)
        D = self.sigmoid(logits)
        loss = self.loss(D, self.labels)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        grad_norms = []
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norms.append(p.grad.detach().data.norm(2))
        grad_norms = th.stack(grad_norms)
        
        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log metrics
        demo_D, agent_D = D[:self.batch_size], D[self.batch_size:]
        disc_expert_acc = th.mean((demo_D < 0.5).float())
        disc_agent_acc = th.mean((agent_D > 0.5).float())
        disc_expert_logit = th.mean(logits[:self.batch_size])
        disc_agent_logit = th.mean(logits[self.batch_size:])
        metrics = {
            'disc_loss': loss.item(),
            'disc_expert_acc': disc_expert_acc.item(),
            'disc_agent_acc': disc_agent_acc.item(),  
            'disc_expert_logit': disc_expert_logit.item(),
            'disc_agent_logit': disc_agent_logit.item(),
            'grad': grad_norms.mean().item(),
        }
        return metrics
