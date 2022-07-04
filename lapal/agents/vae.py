from typing import Optional, List, Dict
import itertools

import numpy as np
import torch as th
from torch import nn
from torch import optim
import torch.nn.functional as F

from lapal.utils import types
import lapal.utils.pytorch_utils as ptu

class CVAE(nn.Module):
    """
    Conditional VAE to encode and reconstruct actions, conditioned on state
    """
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        lat_ac_dim: int,
        size: Optional[int] = 256,
        n_layers: Optional[int] = 2,
        activation: Optional[str] = 'leaky_relu',
        output_activation: Optional[str] = 'identity',
        lr: Optional[float] = 3e-4,
        kl_coef: Optional[float] = 0.1,
    ):
        super().__init__()

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.lat_ac_dim = lat_ac_dim
        self.device = ptu.device
        self.log_std_min = -5
        self.log_std_max = 5
        self.kl_coef = kl_coef

        self.encoder = ptu.build_mlp(
            input_size=self.ob_dim + self.ac_dim,
            output_size=self.lat_ac_dim * 2,      # mu and log_std
            size=size,
            n_layers=n_layers,
            activation=activation,
            output_activation=output_activation,
        ).to(self.device)

        self.decoder = ptu.build_mlp(
            input_size=self.ob_dim + self.lat_ac_dim,
            output_size=self.ac_dim,
            size=size,
            n_layers=n_layers,
            activation=activation,
            output_activation=output_activation,
        ).to(self.device)

        self.optimizer = optim.Adam(
            itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ),
            lr=lr,
        )

        self.MSELoss = nn.MSELoss(reduction='none')


    def ac_encoder(self, obs, acs):
        
        obs = obs.float()
        acs = acs.float()

        with th.no_grad():
            x = th.cat([obs, acs], dim=-1)
            mu_log_std = self.encoder(x)
            mu, log_std = mu_log_std[:, :self.lat_ac_dim], mu_log_std[:, self.lat_ac_dim:]
            log_std = th.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
            std = th.exp(log_std)
            z = mu + th.randn_like(std) * std       
        return z 

    def ac_decoder(self, obs, lat_acs):
        lat_acs_shape = lat_acs.shape

        if isinstance(obs, np.ndarray):
            if obs.ndim < 2:
                obs = obs[None, ...]
            assert obs.ndim == 2, f"obs shape {obs.shape}"
            obs_tensor = ptu.from_numpy(obs)
        else:
            obs_tensor = obs.float()

        if isinstance(lat_acs, np.ndarray):
            if lat_acs.ndim < 2:
                lat_acs = lat_acs[None, ...]
            assert lat_acs.ndim == 2, f"act shape {lat_ac.shape}"
            lat_acs_tensor = ptu.from_numpy(lat_acs)
        else:
            lat_acs_tensor = lat_acs.float()
        
        acs = self.decoder(th.cat([obs_tensor, lat_acs_tensor], dim=-1))
        acs = th.clamp(acs, -1, 1)        # assumes original action space is bounded by [-1, 1]
        
        if isinstance(lat_acs, np.ndarray):
            acs = ptu.to_numpy(acs)
        if len(lat_acs_shape) == 1:
            acs = acs[0]
        return acs

    def forward(self, obs: th.Tensor,  acs: th.Tensor) -> List[th.Tensor]:
        """Encode and decode, latent observation is used as conditional variable"""
        x = th.cat([obs, acs], dim=-1)
        mu_log_std = self.encoder(x)
        mu, log_std = mu_log_std[:, :self.lat_ac_dim], mu_log_std[:, self.lat_ac_dim:]
        log_std = th.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = th.exp(log_std)
        z = mu + th.randn_like(std) * std
        return self.decoder(th.cat([obs, z], dim=-1)), mu, std


    def train(self, obs: th.Tensor, acs: th.Tensor) -> Dict[str, np.ndarray]:
        """
        Sample a minibatch of observations
        obs: (batch_size, ob_dim)
        """

        pred_acs, mu, std = self(obs, acs)
        recon_loss = F.mse_loss(pred_acs, acs)
        kld_loss = -0.5 * th.mean(th.sum(1 + th.log(std**2) - mu**2 - std**2, dim=1))
        loss = recon_loss + self.kl_coef * kld_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {
            'loss': loss.item(), 
            'recon_loss': recon_loss.item(), 
            'kld_loss': kld_loss.item()
        }

        return metrics