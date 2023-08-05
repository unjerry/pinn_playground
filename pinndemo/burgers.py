import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())

import FCN

pinn = FCN.FCN(2, 1, 64, 4)
pinn.to(torch.device("cuda"))

T, X = np.meshgrid(np.arange(0.0, 1.0, 0.01), np.linspace(-1.0, 1.0, 256))
bc = np.stack((T, X), axis=2)
I_phy = torch.from_numpy(bc)
I_phy.requires_grad = True
I_phy = I_phy.to(torch.device("cuda"))

T, X = np.meshgrid(np.array(0.0), np.linspace(-1.0, 1.0, 256))
bc = np.stack((T, X), axis=2)
I_bond_x = torch.from_numpy(bc)
I_bond_x.requires_grad = True
I_bond_x = I_bond_x.to(torch.device("cuda"))
I_bond_x_u = -torch.sin(torch.pi * I_bond_x[:, :, 1:2])

T, X = np.meshgrid(np.arange(0.0, 1.0, 0.01), np.array(-1.0))
bc = np.stack((T, X), axis=2)
I_bond_t0 = torch.from_numpy(bc)
I_bond_t0.requires_grad = True
I_bond_t0 = I_bond_t0.to(torch.device("cuda"))
I_bond_t0_u = torch.zeros_like(I_bond_t0[:, :, 1:2])

T, X = np.meshgrid(np.arange(0.0, 1.0, 0.01), np.array(1.0))
bc = np.stack((T, X), axis=2)
I_bond_t1 = torch.from_numpy(bc)
I_bond_t1.requires_grad = True
I_bond_t1 = I_bond_t1.to(torch.device("cuda"))
I_bond_t1_u = torch.zeros_like(I_bond_t1[:, :, 1:2])
print(I_phy.shape)
print(I_bond_x.shape)
print(I_bond_x_u.shape)
print(I_bond_t0.shape)
print(I_bond_t0_u.shape)
print(I_bond_t1.shape)
print(I_bond_t1_u.shape)

plt.plot(I_bond_x[:, :, 1:2].squeeze().tolist(), I_bond_x_u.squeeze().tolist())
plt.plot(I_bond_t0[:, :, 0:1].squeeze().tolist(), I_bond_t0_u.squeeze().tolist())
plt.plot(I_bond_t1[:, :, 0:1].squeeze().tolist(), I_bond_t1_u.squeeze().tolist())
plt.show()

optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-4)

ni = 0.01 / torch.pi
cnt = 0
while True:
    optimizer.zero_grad()

    u = pinn(I_phy)
    dudi = torch.autograd.grad(u, I_phy, torch.ones_like(u), create_graph=True)[0]
    d2udi2 = torch.autograd.grad(dudi, I_phy, torch.ones_like(dudi), create_graph=True)[
        0
    ]

    # print(dudi.shape)
    # print(d2udi2.shape)
    # print("u", u.shape)
    # print(I_phy.shape)
    loss_f = torch.mean(
        (dudi[:, :, 0] + u[:, :, 0] * dudi[:, :, 1] - ni * d2udi2[:, :, 1]) ** 2
    )

    u = pinn(I_bond_x)
    loss_x = torch.mean((u - I_bond_x_u) ** 2)
    u = pinn(I_bond_t0)
    loss_t0 = torch.mean((u - I_bond_t0_u) ** 2)
    u = pinn(I_bond_t1)
    loss_t1 = torch.mean((u - I_bond_t1_u) ** 2)

    loss = loss_f + loss_t0 + loss_t1 + loss_x
    loss.backward(retain_graph=True)
    optimizer.step()

    cnt += 1
    print(cnt, loss.tolist())
    if cnt % 100 == 0:
        torch.save(pinn, "pinn{}".format(cnt // 100))
