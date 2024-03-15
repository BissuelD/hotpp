import torch
from torch import nn


def rotation_matrix_3d(angles):
    x, y, z = angles
    R = torch.stack(
        [ torch.cos(y) * torch.cos(z),
         -torch.cos(y) * torch.sin(z),
          torch.sin(y),
          torch.sin(x) * torch.sin(y) * torch.cos(z) + torch.cos(x) * torch.sin(z),
         -torch.sin(x) * torch.sin(y) * torch.sin(z) + torch.cos(x) * torch.cos(z),
         -torch.sin(x) * torch.cos(y),
         -torch.cos(x) * torch.sin(y) * torch.cos(z) + torch.sin(x) * torch.cos(z),
          torch.cos(x) * torch.sin(y) * torch.sin(z) + torch.sin(x) * torch.cos(z),
          torch.cos(x) * torch.cos(y),        
    ], dim=1)
    return R.reshape(-1, 3, 3)


class E3RotateLayer(nn.Module):
    def __init__(self,
                 channels: int,
                 ) -> None:
        super().__init__()
        angles = torch.empty((3, channels), dtype=torch.float32)
        nn.init.normal_(angles, mean=0.0, std=0.01)
        self.angles = nn.Parameter(angles)

    def forward(self,
                input_tensors : torch.Tensor,  # [n, 3]
                ) -> torch.Tensor:             # [n , c, 3]
        rotation_matrix = rotation_matrix_3d(self.angles)
        return torch.matmul(rotation_matrix.unsqueeze(0), input_tensors[:, None, :, None]).squeeze(-1)
