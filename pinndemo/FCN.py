import torch
import torch.nn as nn


class FCN(nn.Module):
    "define a fully connected network"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(FCN, self).__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(
            *[nn.Linear(N_INPUT, N_HIDDEN, dtype=torch.float64), activation()]
        )
        self.fch = nn.Sequential(
            *[
                nn.Sequential(
                    *[nn.Linear(N_HIDDEN, N_HIDDEN, dtype=torch.float64), activation()]
                )
                for _ in range(N_LAYERS - 1)
            ]
        )
        self.fce = nn.Linear(N_HIDDEN, N_INPUT, dtype=torch.float64)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
