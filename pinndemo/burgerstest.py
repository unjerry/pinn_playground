import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())

import FCN

pinn = torch.load("pinndt\\pinn320.ntdt")
pinn.to(torch.device("cuda"))

T, X = np.meshgrid(np.arange(0.0, 1.0, 0.01), np.linspace(-1.0, 1.0, 256))
bc = np.stack((T, X), axis=2)
I_phy = torch.from_numpy(bc)
I_phy = I_phy.to(torch.device("cuda"))

print(I_phy.shape)
u = pinn(I_phy)
tt = torch.squeeze(u).cpu()
U = tt.detach().numpy()

fig = plt.figure()
# ax1 = plt.axes(projection="3d")
ax2 = fig.add_subplot(122)  # 这种方法也可以画多个子图
ax1 = fig.add_subplot(121, projection="3d")  # 这种方法也可以画多个子图
# 定义三维数据
# 作图
ax1.plot_surface(X, T, U, cmap="rainbow")
p = ax2.contourf(X, T, U,np.arange(-1.,1.,0.001), cmap="rainbow")  # 等高线图，要设置offset，为Z的最小值
plt.colorbar(p)
plt.show()
