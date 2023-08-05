import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())


class FCN(nn.Module):
    "def a fully connected network"

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


def exact_solution(d, w0, t):
    "def a analytical solution"
    assert d < w0
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    cos = torch.cos(phi + w * t)
    exp = torch.exp(-d * t)
    u = exp * 2 * A * cos
    return u


pinn = FCN(1, 1, 32, 3)
pinn.to(torch.device("cuda"))

t_bond = torch.tensor(0.0, requires_grad=True, dtype=torch.float64, device="cuda").view(
    -1, 1
)

t_phyi = torch.linspace(
    0, 1, 30, requires_grad=True, dtype=torch.float64, device="cuda"
).view(-1, 1)

d, w0 = 2, 20
mu, k = 2 * d, w0**2
T = torch.linspace(0, 1, 300, dtype=torch.float64, device="cuda").view(-1, 1)
U = exact_solution(d, w0, T)
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
for i in range(15000):
    optimizer.zero_grad()

    lambda1, lambda2 = 1e-1, 1e-3

    u = pinn(t_bond)
    loss1 = (torch.squeeze(u) - 1) ** 2

    dudt = torch.autograd.grad(u, t_bond, torch.ones_like(u), create_graph=True)[0]
    loss2 = (torch.squeeze(dudt) - 0) ** 2

    u = pinn(t_phyi)
    dudt = torch.autograd.grad(u, t_phyi, torch.ones_like(u), create_graph=True)[0]
    d2udt2 = torch.autograd.grad(
        dudt, t_phyi, torch.ones_like(dudt), create_graph=True
    )[0]
    loss3 = torch.mean((d2udt2 + mu * dudt + k * u) ** 2)

    loss = loss1 + lambda1 * loss2 + lambda2 * loss3
    loss.backward()

    optimizer.step()
    print(i, loss.tolist())
torch.save(pinn, "fistpinn.ntdt")

plt.plot(T.tolist(), U.tolist())
plt.show()
plt.plot(T.tolist(), (pinn(T)).tolist())
plt.show()

plt.plot(T.tolist(), U.tolist())
plt.plot(T.tolist(), (pinn(T)).tolist())
plt.show()
