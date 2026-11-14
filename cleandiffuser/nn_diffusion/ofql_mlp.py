from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import SUPPORTED_TIMESTEP_EMBEDDING

__all__ = ["OFQLMlp"]


class OFQLMlp(BaseNNDiffusion):
    """MLP backbone for the OFQL policy.

    Predicts the average velocity of the flow over the interval `[r, t]`, conditioned
    on the observation. Because the average velocity is learned directly, the induced
    ODE can be integrated in a single Euler step at action-selection time.

    Args:
        obs_dim (int):
            The dimension of the observation, used as the conditioning input.
        act_dim (int):
            The dimension of the action, i.e. the dimension of the generated sample.
        emb_dim (int):
            The dimension of the `t` and `r` timestep embeddings.
        timestep_emb_type (str):
            The type of the timestep embedding. Default: "positional"
        timestep_emb_params (Optional[dict]):
            The parameters of the timestep embedding. Default: None

    Examples:
        >>> model = OFQLMlp(obs_dim=16, act_dim=10, emb_dim=16)
        >>> x = torch.randn((2, 10))
        >>> t, r = torch.rand((2,)), torch.rand((2,))
        >>> condition = torch.randn((2, 16))
        >>> model(x, t, r, condition).shape
        torch.Size([2, 10])
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        emb_dim: int = 16,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.Mish(), nn.Linear(emb_dim * 2, emb_dim)
        )

        self.r_make_noise = self.map_noise = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](emb_dim)
        self.r_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.Mish(), nn.Linear(emb_dim * 2, emb_dim)
        )

        self.mid_layer = nn.Sequential(
            nn.Linear(obs_dim+ act_dim + emb_dim+ emb_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
        )

        self.final_layer = nn.Linear(256, act_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, r: torch.Tensor, condition: Optional[torch.Tensor] = None):
        t = self.time_mlp(self.map_noise(t))
        r = self.r_mlp(self.r_make_noise(r))

        # if condition is not None:
        #     t += condition
        
        x = torch.cat([x, t, r, condition], -1)
        x = self.mid_layer(x)

        return self.final_layer(x)


if __name__ == "__main__":
    # Example usage
    model = OFQLMlp(obs_dim=16, act_dim=10, emb_dim=16)
    x = torch.randn((2, 10))
    t, r = torch.rand((2,)), torch.rand((2,))
    condition = torch.randn((2, 16))
    print(model(x, t, r, condition).shape)  # Should output: torch.Size([2, 10])
    # print(model(x, t, r, None).shape)  # Should output: torch.Size([2, 10])