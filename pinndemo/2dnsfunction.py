import torch
import FCN
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ans_data_dict_file = ".\\data_dict.dt"
prd_data_dict_file = ".\\data_dict.dt"
pinn_net_data_file = ".\\data_dict.ntdt"

ans_data_dict = torch.load(ans_data_dict_file)
prd_data_dict = {}


def export_anime(filename, data_dict, secd):
    fig = plt.figure("fluid flow", figsize=(8, 4))
    ax = plt.subplot()

    def anime(i):
        norm = torch.hypot(
            data_dict["U"][:, :, i, 0],
            data_dict["U"][:, :, i, 1],
        )
        pc = ax.pcolormesh(
            data_dict["D"][:, :, i, 0].cpu().detach().numpy(),
            data_dict["D"][:, :, i, 1].cpu().detach().numpy(),
            norm.cpu().detach().numpy(),
            cmap="rainbow",
        )
        ax.quiver(
            data_dict["D"][:, :, i, 0].cpu().detach().numpy(),
            data_dict["D"][:, :, i, 1].cpu().detach().numpy(),
            data_dict["U"][:, :, i, 0].cpu().detach().numpy()
            / norm.cpu().detach().numpy(),
            data_dict["U"][:, :, i, 1].cpu().detach().numpy()
            / norm.cpu().detach().numpy(),
            headaxislength=5,
        )
        ax.set_title("time={}".format(i / 10))

    ani = animation.FuncAnimation(fig=fig, func=anime, frames=secd * 10, interval=10)
    ani.save(filename=filename + ".mp4", fps=10, writer="ffmpeg")
    # plt.show()


def export_compare_anime(filename, data_dict, ans_dict, secd):
    fig = plt.figure("fluid flow", figsize=(16, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    def anime(i):
        norm1 = torch.hypot(
            data_dict["U"][:, :, i, 0],
            data_dict["U"][:, :, i, 1],
        )
        norm2 = torch.hypot(
            ans_dict["U"][:, :, i, 0],
            ans_dict["U"][:, :, i, 1],
        )
        ax1.pcolormesh(
            data_dict["D"][:, :, i, 0].cpu().detach().numpy(),
            data_dict["D"][:, :, i, 1].cpu().detach().numpy(),
            norm1.cpu().detach().numpy(),
            cmap="rainbow",
        )
        ax2.pcolormesh(
            ans_dict["D"][:, :, i, 0].cpu().detach().numpy(),
            ans_dict["D"][:, :, i, 1].cpu().detach().numpy(),
            norm2.cpu().detach().numpy(),
            cmap="rainbow",
        )
        ax1.quiver(
            data_dict["D"][:, :, i, 0].cpu().detach().numpy(),
            data_dict["D"][:, :, i, 1].cpu().detach().numpy(),
            data_dict["U"][:, :, i, 0].cpu().detach().numpy()
            / norm1.cpu().detach().numpy(),
            data_dict["U"][:, :, i, 1].cpu().detach().numpy()
            / norm1.cpu().detach().numpy(),
            headaxislength=5,
        )
        ax2.quiver(
            ans_dict["D"][:, :, i, 0].cpu().detach().numpy(),
            ans_dict["D"][:, :, i, 1].cpu().detach().numpy(),
            ans_dict["U"][:, :, i, 0].cpu().detach().numpy()
            / norm2.cpu().detach().numpy(),
            ans_dict["U"][:, :, i, 1].cpu().detach().numpy()
            / norm2.cpu().detach().numpy(),
            headaxislength=5,
        )
        ax1.set_title("time={}".format(i / 10))
        ax2.set_title("time={}".format(i / 10))

    ani = animation.FuncAnimation(fig=fig, func=anime, frames=secd * 10, interval=10)
    ani.save(filename=filename + ".mp4", fps=10, writer="ffmpeg")
    # plt.show()


try:
    pinn = torch.load(pinn_net_data_file)
except:
    pinn = FCN.FCN(3, 3, 16, 4)
pinn.to(torch.device("cuda"))

prd_data_dict["D"] = ans_data_dict["D"]

mu = 0.01
jac = torch.func.jacrev
vmp = torch.vmap
es = torch.einsum
optimizer = torch.optim.Adam(pinn.parameters(), lr=5e-5)
R = 0
LEN = 200
while R <= 0:
    New_Init = None
    if R >= 1:
        with torch.no_grad():
            New_Init = pinn(prd_data_dict["D"][:, :, 1 : R + 1, :])
    i = 0
    while True:
        optimizer.zero_grad()

        out = pinn(prd_data_dict["D"][:, :, 0:1, :])
        loss_init = torch.mean(
            (out[:, :, 0:1, 2:3] - ans_data_dict["P"][:, :, 0:1, :]) ** 2
        ) + torch.mean((out[:, :, 0:1, 0:2] - ans_data_dict["U"][:, :, 0:1, :]) ** 2)
        loss_bond = 0
        out = pinn(prd_data_dict["D"][:, 0, 0:LEN, :])
        loss_bond += torch.mean(
            (out[:, 0:LEN, 2:3] - ans_data_dict["P"][:, 0, 0:LEN, :]) ** 2
        ) + torch.mean((out[:, 0:LEN, 0:2] - ans_data_dict["U"][:, 0, 0:LEN, :]) ** 2)
        out = pinn(prd_data_dict["D"][:, -1, 0:LEN, :])
        loss_bond += torch.mean(
            (out[:, 0:LEN, 2:3] - ans_data_dict["P"][:, -1, 0:LEN, :]) ** 2
        ) + torch.mean((out[:, 0:LEN, 0:2] - ans_data_dict["U"][:, -1, 0:LEN, :]) ** 2)
        out = pinn(prd_data_dict["D"][0, :, 0:LEN, :])
        loss_bond += torch.mean(
            (out[:, 0:LEN, 2:3] - ans_data_dict["P"][0, :, 0:LEN, :]) ** 2
        ) + torch.mean((out[:, 0:LEN, 0:2] - ans_data_dict["U"][0, :, 0:LEN, :]) ** 2)
        out = pinn(prd_data_dict["D"][-1, :, 0:LEN, :])
        loss_bond += torch.mean(
            (out[:, 0:LEN, 2:3] - ans_data_dict["P"][-1, :, 0:LEN, :]) ** 2
        ) + torch.mean((out[:, 0:LEN, 0:2] - ans_data_dict["U"][-1, :, 0:LEN, :]) ** 2)
        if R == 0:
            loss_proc = 0
        else:
            out = pinn(prd_data_dict["D"][:, :, R : R + 1, :])
            loss_proc = torch.mean(
                (out[:, :, 1 : R + 1, :] - New_Init[:, :, R : R + 1, :]) ** 2
            )
        loss = 10 * (loss_init + loss_proc + loss_bond)
        for j in range(LEN):
            out = pinn(prd_data_dict["D"][:, :, R + LEN : R + LEN + 1, :])
            JAC = vmp(vmp(vmp((jac(pinn)))))(
                prd_data_dict["D"][:, :, R + LEN : R + LEN + 1, :]
            )
            HES = vmp(vmp(vmp((jac(jac(pinn))))))(
                prd_data_dict["D"][:, :, R + LEN : R + LEN + 1, :]
            )
            # print(JAC.shape)
            # print((JAC[:, :, :, 0:2, 2]).shape)
            # print(es("...ij,...j->...i", JAC[:, :, :, 0:2, 0:2], out[:, :, :, 0:2]).shape)
            # print((JAC[:, :, :, 2, 0:2]).shape)
            # print(HES.shape)
            # print(es("...ijj->...i", HES[:, :, :, 0:2, 0:2, 0:2]).shape)

            # print("div", es("...ii->...", JAC[:, :, :, 0:2, 0:2]).shape)

            loss_ns = torch.mean(
                (
                    (JAC[:, :, :, 0:2, 2])
                    + es("...ij,...j->...i", JAC[:, :, :, 0:2, 0:2], out[:, :, :, 0:2])
                    + (JAC[:, :, :, 2, 0:2])
                    - mu * es("...ijj->...i", HES[:, :, :, 0:2, 0:2, 0:2])
                )
                ** 2
            )
            loss_div = torch.mean(es("...ii->...", JAC[:, :, :, 0:2, 0:2]) ** 2)
            loss_p = torch.mean(
                (
                    es("...ijj->...i", HES[:, :, :, 2, 0:2, 0:2])
                    + JAC[:, :, :, 0, 0] * JAC[:, :, :, 0, 0]
                    + JAC[:, :, :, 1, 1] * JAC[:, :, :, 1, 1]
                    + 2 * JAC[:, :, :, 1, 0] * JAC[:, :, :, 0, 1]
                    + out[:, :, :, 0] * HES[:, :, :, 0, 0, 0]
                    + out[:, :, :, 0] * HES[:, :, :, 1, 0, 1]
                    + out[:, :, :, 1] * HES[:, :, :, 0, 0, 1]
                    + out[:, :, :, 1] * HES[:, :, :, 1, 1, 1]
                )
                ** 2
            )
            loss += (loss_ns + loss_div + loss_p) / LEN

        loss.backward()
        optimizer.step()

        print(R, i + 1, loss_init.tolist())
        i += 1
        if loss_init.tolist() < 1e-4:
            R = R + 1
            break


out = pinn(prd_data_dict["D"][:, :, :, :])
prd_data_dict["U"] = out[:, :, :, 0:2]
prd_data_dict["P"] = out[:, :, :, 2:3]

export_compare_anime("2dnsfuncconpare", prd_data_dict, ans_data_dict, 2)

torch.save(pinn, pinn_net_data_file)
