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
    X, Y = np.meshgrid(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10))
    bc = np.stack((X, Y), axis=2)
    bc = np.concatenate((bc, T * np.ones_like(X)[:, :, np.newaxis]), axis=2)
    # print(bc.shape)
    # print(np.ones_like(Y)[:, :, np.newaxis].shape)
    # print(bc)
    Pice = torch.from_numpy(bc)
    Pice = Pice.to(torch.device("cuda"))

    out = pinn(Pice)
    # print(out.shape)

    fig = plt.figure(figsize=(2 * 7, 7))
    # ax1 = plt.axes(projection="3d")
    ax1 = fig.add_subplot(121)  # 这种方法也可以画多个子图
    ax2 = fig.add_subplot(122)  # 这种方法也可以画多个子图
    ax1.streamplot(
        X,
        Y,
        out[:, :, 0].squeeze().cpu().detach().numpy(),
        out[:, :, 1].squeeze().cpu().detach().numpy(),
        density=5,
        broken_streamlines=False
    )
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
