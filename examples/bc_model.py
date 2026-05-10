"""PilotNet-style CNN for end-to-end driving (CARLA Behavioral Cloning).

Architecture follows NVIDIA's "End to End Learning for Self-Driving Cars"
(Bojarski et al., 2016) with three regression heads: steer, throttle, brake.

Input:  RGB image (3, IMG_H, IMG_W) normalised to [-1, 1]  (IMG_H=66, IMG_W=200)
Output: dict {'steer': tanh in [-1, 1],
              'throttle': sigmoid in [0, 1],
              'brake': sigmoid in [0, 1]}
"""

from __future__ import print_function

import torch
import torch.nn as nn

IMG_H = 66
IMG_W = 200


class PilotNet(nn.Module):
    def __init__(self, dropout=0.3):
        super(PilotNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ELU(),
        )
        # Determined from a forward pass on a (1,3,66,200) tensor:
        # output shape (1, 64, 1, 18) -> flatten = 1152
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 3, IMG_H, IMG_W)).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_flat, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
        )
        self.head_steer = nn.Linear(10, 1)
        self.head_throttle = nn.Linear(10, 1)
        self.head_brake = nn.Linear(10, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return {
            'steer': torch.tanh(self.head_steer(x)).squeeze(-1),
            'throttle': torch.sigmoid(self.head_throttle(x)).squeeze(-1),
            'brake': torch.sigmoid(self.head_brake(x)).squeeze(-1),
        }


def build_model(dropout=0.3):
    return PilotNet(dropout=dropout)


if __name__ == '__main__':
    m = build_model()
    n_params = sum(p.numel() for p in m.parameters())
    print('PilotNet parameters: {:,}'.format(n_params))
    out = m(torch.randn(2, 3, IMG_H, IMG_W))
    for k, v in out.items():
        print(k, tuple(v.shape), 'min={:.3f} max={:.3f}'.format(v.min().item(), v.max().item()))
