import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())

import FCN

pinn = None
try:
    pinn = torch.load("fluidcoli.ntdt")
except:
    pinn = FCN.FCN(3, 3, 32, 8)
pinn.to(torch.device("cuda"))


def showpic(T):
    X, Y = np.meshgrid(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 10.0, 10))
    bc = np.stack((X, Y), axis=2)
    bc = np.concatenate((bc, T * np.ones_like(X)[:, :, np.newaxis]), axis=2)
    # print(bc.shape)
    # print(np.ones_like(Y)[:, :, np.newaxis].shape)
    # print(bc)
    Pice = torch.from_numpy(bc)
    Pice = Pice.to(torch.device("cuda"))

    with torch.no_grad():
        out = pinn(Pice)
        # print(out.shape)
        H = torch.einsum("ijk,ijk->ij", out[:, :, 0:2], out[:, :, 0:2])

    fig = plt.figure(figsize=(2 * 7, 7))
    # ax1 = plt.axes(projection="3d")
    ax1 = fig.add_subplot(121)  # 这种方法也可以画多个子图
    ax2 = fig.add_subplot(122)  # 这种方法也可以画多个子图
    ax1.streamplot(
        X,
        Y,
        out[:, :, 0].squeeze().cpu().detach().numpy(),
        out[:, :, 1].squeeze().cpu().detach().numpy(),
        density=1,
        broken_streamlines=False,
    )
    p1 = ax1.contourf(
        X, Y, H.squeeze().cpu().detach().numpy(), 1000, cmap="rainbow"
    )  # 等高线图，要设置offset，为Z的最小值
    plt.colorbar(p1)
    p = ax2.contourf(
        X, Y, out[:, :, 2].squeeze().cpu().detach().numpy(), 1000, cmap="rainbow"
    )  # 等高线图，要设置offset，为Z的最小值
    plt.colorbar(p)
    plt.grid()
    plt.show()


def showheight(T):
    X, Y = np.meshgrid(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10))
    bc = np.stack((X, Y), axis=2)
    bc = np.concatenate((bc, T * np.ones_like(X)[:, :, np.newaxis]), axis=2)
    # print(bc.shape)
    # print(np.ones_like(Y)[:, :, np.newaxis].shape)
    # print(bc)
    Pice = torch.from_numpy(bc)
    Pice = Pice.to(torch.device("cuda"))

    with torch.no_grad():
        out = pinn(Pice)
        # print(out.shape)
        H = torch.einsum("ijk,ijk->ij", out[:, :, 0:2], out[:, :, 0:2])

    fig = plt.figure(figsize=(2 * 7, 7))
    # ax1 = plt.axes(projection="3d")
    ax1 = fig.add_subplot(121)  # 这种方法也可以画多个子图
    ax2 = fig.add_subplot(122)  # 这种方法也可以画多个子图
    ax1.streamplot(
        X,
        Y,
        out[:, :, 0].squeeze().cpu().detach().numpy(),
        out[:, :, 1].squeeze().cpu().detach().numpy(),
        density=1,
        broken_streamlines=False,
    )
    p1 = ax1.contourf(
        X, Y, H.squeeze().cpu().detach().numpy(), 1000, cmap="rainbow"
    )  # 等高线图，要设置offset，为Z的最小值
    plt.colorbar(p1)
    p = ax2.contourf(
        X, Y, out[:, :, 2].squeeze().cpu().detach().numpy(), 1000, cmap="rainbow"
    )  # 等高线图，要设置offset，为Z的最小值
    plt.colorbar(p)
    plt.grid()
    plt.show()


def showfourheight(T):
    X, Y = np.meshgrid(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10))
    bc = np.stack((X, Y), axis=2)
    bc = np.concatenate((bc, T * np.ones_like(X)[:, :, np.newaxis]), axis=2)
    # print(bc.shape)
    # print(np.ones_like(Y)[:, :, np.newaxis].shape)
    # print(bc)
    Pice = torch.from_numpy(bc)
    Pice = Pice.to(torch.device("cuda"))

    with torch.no_grad():
        out = pinn(Pice)
        # print(out.shape)
        H = torch.einsum("ijk,ijk->ij", out[:, :, 0:2], out[:, :, 0:2])

    fig = plt.figure(figsize=(2 * 7, 7))
    # ax1 = plt.axes(projection="3d")
    ax1 = fig.add_subplot(221)  # 这种方法也可以画多个子图
    ax2 = fig.add_subplot(222)  # 这种方法也可以画多个子图
    ax3 = fig.add_subplot(223, projection="3d")  # 这种方法也可以画多个子图
    ax4 = fig.add_subplot(224, projection="3d")  # 这种方法也可以画多个子图

    ax1.streamplot(
        X,
        Y,
        out[:, :, 0].squeeze().cpu().detach().numpy(),
        out[:, :, 1].squeeze().cpu().detach().numpy(),
        density=1,
        broken_streamlines=False,
    )
    p1 = ax1.contourf(
        X, Y, H.squeeze().cpu().detach().numpy(), 1000, cmap="rainbow"
    )  # 等高线图，要设置offset，为Z的最小值
    plt.colorbar(p1)

    p2 = ax2.contourf(
        X, Y, out[:, :, 2].squeeze().cpu().detach().numpy(), 1000, cmap="rainbow"
    )  # 等高线图，要设置offset，为Z的最小值
    plt.colorbar(p2)

    p3 = ax3.plot_surface(X, Y, H.squeeze().cpu().detach().numpy(), cmap="rainbow")
    plt.colorbar(p3)

    p4 = ax4.plot_surface(
        X, Y, out[:, :, 2].squeeze().cpu().detach().numpy(), cmap="rainbow"
    )  # 等高线图，要设置offset，为Z的最小值
    plt.colorbar(p4)

    plt.grid()
    plt.show()


V_X_0 = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0],
        [0.0, 0.1, 0.5, 0.1, 0.0, 0.0, -0.1, -0.5, -0.1, 0.0],
        [0.0, 0.2, 1.0, 0.2, 0.0, 0.0, -0.2, -1.0, -0.2, 0.0],
        [0.0, 0.1, 0.5, 0.1, 0.0, 0.0, -0.1, -0.5, -0.1, 0.0],
        [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
V_Y_0 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
P_0 = np.array(
    [
        [0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
D_T_0_ANS = np.stack((V_X_0, V_Y_0, P_0), axis=2)
T_0_ANS = torch.from_numpy(D_T_0_ANS[:, :, np.newaxis, :])
T_0_ANS = T_0_ANS.to(torch.device("cuda"))
print("T_0_ANS", T_0_ANS)

Ng = 10
Tg = 100

X, Y, T = np.meshgrid(
    np.linspace(0.0, 1.0, Ng), np.linspace(0.0, 1.0, Ng), np.linspace(0.0, 1.0, Tg)
)
bc = np.stack((X, Y, T), axis=3)
# print(bc)
# print(bc.shape)
PHY_GRID = torch.from_numpy(bc)
PHY_GRID.requires_grad = True
PHY_GRID = PHY_GRID.to(torch.device("cuda"))
T_0 = PHY_GRID[:, :, 0:1, :]
print("T_0", T_0.shape)
print("T_0_ANS", T_0_ANS.shape)
WALL_0Y = PHY_GRID[0:1, :, :, :]
WALL_0Y_ANS = torch.zeros([1, Ng, Tg, 3], dtype=torch.float64, device="cuda")
WALL_1Y = PHY_GRID[-2:-1, :, :, :]
WALL_1Y_ANS = torch.zeros([1, Ng, Tg, 3], dtype=torch.float64, device="cuda")
WALL_X0 = PHY_GRID[:, 0:1, :, :]
WALL_X0_ANS = torch.zeros([Ng, 1, Tg, 3], dtype=torch.float64, device="cuda")
WALL_X1 = PHY_GRID[:, -2:-1, :, :]
WALL_X1_ANS = torch.zeros([Ng, 1, Tg, 3], dtype=torch.float64, device="cuda")
mu = 0.001
# print(WALL_0Y.shape)
# print(WALL_0Y_ANS.shape)
# print(WALL_1Y.shape)
# print(WALL_1Y_ANS.shape)
# print(WALL_X0.shape)
# print(WALL_X0_ANS.shape)
# print(WALL_X1.shape)
# print(WALL_X1_ANS.shape)
# print(T_0.shape)
# print(T_0_ANS.shape)

optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-4)

for i in range(0):
    optimizer.zero_grad()
    out = pinn(T_0)
    loss_T_0_U = torch.mean((out[:, :, :, 0:2] - T_0_ANS[:, :, :, 0:2]) ** 2)
    # loss_T_0_P = torch.mean((out[:, :, :, 2]) ** 2)
    out = pinn(WALL_0Y)
    loss_WALL_0Y = torch.mean((out[:, :, :, 0:2] - WALL_0Y_ANS[:, :, :, 0:2]) ** 2)
    out = pinn(WALL_1Y)
    loss_WALL_1Y = torch.mean((out[:, :, :, 0:2] - WALL_1Y_ANS[:, :, :, 0:2]) ** 2)
    out = pinn(WALL_X0)
    loss_WALL_X0 = torch.mean((out[:, :, :, 0:2] - WALL_X0_ANS[:, :, :, 0:2]) ** 2)
    out = pinn(WALL_X1)
    loss_WALL_X1 = torch.mean((out[:, :, :, 0:2] - WALL_X1_ANS[:, :, :, 0:2]) ** 2)

    out = pinn(PHY_GRID)
    bt = torch.func.vmap
    jac = torch.func.jacrev
    JAC = bt(bt(bt(jac(pinn))))
    HES = bt(bt(bt(jac(jac(pinn)))))
    J = JAC(PHY_GRID)
    H = HES(PHY_GRID)
    I = torch.diag(torch.tensor([1, 1, 1], dtype=torch.float64, device="cuda"))
    Lp = torch.einsum("...ii->...i", H * I)
    Ut = torch.einsum("...ij->...ji", J[:, :, :, 0:2, 2:3]).squeeze()
    MU = J[:, :, :, 0:2, 0:2]
    MU2 = Lp[:, :, :, 0:2, 0:2]
    U = out[:, :, :, 0:2, np.newaxis]
    UdU = torch.einsum("...ij->...ji", MU @ U).squeeze()
    ddU = torch.einsum(
        "...ij,j->...i", MU2, mu * torch.ones([2], dtype=torch.float64, device="cuda")
    ).squeeze()
    dP = J[:, :, :, 2:3, 0:2].squeeze()
    dU = torch.einsum("...ii->...i", J[:, :, :, 0:2, 0:2]).squeeze()
    LpP = torch.einsum("...i->...", Lp[:, :, :, 2:3, 0:2].squeeze()).squeeze()
    MIXEDU = J[:, :, :, 0, 0] * J[:, :, :, 0, 0] + Lp[:, :, :, 0, 0] * out[:, :, :, 0]
    +J[:, :, :, 1, 0] * J[:, :, :, 0, 1] * 2
    +out[:, :, :, 1] * H[:, :, :, 0, 0, 1] + out[:, :, :, 0] * H[:, :, :, 1, 0, 1]
    +J[:, :, :, 1, 1] * J[:, :, :, 1, 1] + Lp[:, :, :, 1, 1] * out[:, :, :, 1]
    # print(dU.shape)
    """
    print(PHY_GRID.shape)
    print(out.shape)
    print(J.shape)
    print("Ut", Ut.shape)
    print("UdU", UdU.shape)
    print("J", J.shape)
    print("MU", MU.shape)
    print("out", out.shape)
    print("U", U.shape)
    print("UdU", UdU.shape)
    print("dP", dP.shape)
    print("Lp", Lp.shape)
    print("MU2", MU2.shape)
    print("ddU", ddU.shape)
    print(I)
    """
    loss_phy = torch.mean((Ut + UdU + dP - ddU) ** 2)
    loss_U = torch.mean((torch.einsum("...i->...", dU)) ** 2)
    loss_P = torch.mean((LpP + MIXEDU) ** 2)

    loss = (
        10 * (loss_phy + 3 * loss_U + 2 * loss_P)
        + (loss_T_0_U)  # + loss_T_0_P)
        + (loss_WALL_0Y + loss_WALL_1Y + loss_WALL_X0 + loss_WALL_X1)
    )
    print(i, loss.tolist())
    # print(out[0, 0, 0, :].tolist())
    loss.backward()
    # for i in pinn.parameters():
    #    print(i.grad)
    optimizer.step()
    if (i + 1) % 1000 == 0:
        torch.save(pinn, "fluidcoli{}.ntdt".format((i + 1) // 1000))

torch.save(pinn, "fluidcoli.ntdt")

# for i in range(5):
# showheight(i / 10)
for i in range(10):
    showheight(i / 10)
