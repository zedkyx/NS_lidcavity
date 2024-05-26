import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import functional as F
import numpy as np
import matplotlib.pyplot as plt

# 定义batch计算jacobian的函数
def batch_jacobian(func, x, create_graph=False):
    def _func_sum(x):
        # func(x) 的输出形状为 [Batch, Channels, Height, Width]，即 [1, 3, 40, 40]
        return func(x).sum(dim=[0, 2, 3])  # Sum over all dimensions except the batch dimension and channel dimension
    #[3, 1, 2, 40, 40]
    jacobian = F.jacobian(_func_sum, x, create_graph=create_graph)
    #[1, 3, 2, 40, 40]
    jacobian = jacobian.permute(1, 0, 2, 3, 4)
    return jacobian


# 卷积神经网络模型
class CNNPINN(nn.Module):
    def __init__(self, input_channels=2):  # Input channels: (x, y)
        super(CNNPINN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 40 * 40, 256)  # 输入尺寸为 (40, 40)
        self.fc2 = nn.Linear(256, 3 * 40 * 40)  # 输出: (u, v, p) * 网格尺寸 * 网格尺寸

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), 3, 40, 40)  # 拟输出形状
        return x

# 定义Navier-Stokes方程损失函数
def ns_loss(model, X, Re=100):
    batch_size = 1
    grid_size = 40

    # in_channels=2
    X = X.view(batch_size, grid_size, grid_size, 2)
    # 调整顺序为 (batch_size, in_channels, grid_size, grid_size)
    X = X.permute(0, 3, 1, 2).contiguous()

    xyt = X
    #print(xyt.shape)
    out = model(xyt)
    #print(out.shape)
    u = out[:, 0, :, :].unsqueeze(1)
    #print(u.shape) #torch.Size([1, 1, 40, 40])
    v = out[:, 1, :, :].unsqueeze(1)
    p = out[:, 2, :, :].unsqueeze(1)

    sol_d1_pde = batch_jacobian(model, xyt, create_graph=True)
    #print(sol_d1_pde.shape) #torch.Size([1, 3, 2, 40, 40])
    dudx = sol_d1_pde[:, 0, 0, :, :] #torch.Size([1, 40, 40])
    dudy = sol_d1_pde[:, 0, 1, :, :]
    dvdx = sol_d1_pde[:, 1, 0, :, :]
    dvdy = sol_d1_pde[:, 1, 1, :, :]
    dpdx = sol_d1_pde[:, 2, 0, :, :]
    dpdy = sol_d1_pde[:, 2, 1, :, :]

    sol_d2_u = batch_jacobian(lambda x: sol_d1_pde[:, 0, :, :, :], xyt, create_graph=True)
    sol_d2_v = batch_jacobian(lambda x: sol_d1_pde[:, 1, :, :, :], xyt, create_graph=True)
    #print(sol_d2_u.shape)

    lap_u = sol_d2_u[:, 0, 0, :, :] + sol_d2_u[:, 1, 1, :, :]
    lap_v = sol_d2_v[:, 0, 0, :, :] + sol_d2_v[:, 1, 1, :, :]

    pde_loss1 = (dudx + dvdy) ** 2
    # pde_loss2 = (dudt + u * dudx + v * dudy + dpdx - 1 / Re * lap_u) ** 2
    # pde_loss3 = (dvdt + u * dvdx + v * dvdy + dpdy - 1 / Re * lap_v) ** 2
    pde_loss2 = (u * dudx + v * dudy + dpdx - 1 / Re * lap_u) ** 2
    pde_loss3 = (u * dvdx + v * dvdy + dpdy - 1 / Re * lap_v) ** 2

    pde_loss = torch.mean(pde_loss1 + pde_loss2 + pde_loss3)

    # 边界条件损失
    lid_velocity = torch.ones_like(u[:, :, 0, :])
    zero_velocity = torch.zeros_like(u[:, :, 0, :])

    # p=0的约束
    p_zero_constraint = torch.mean(p[:, :, 0, 0] ** 2)  # 在 x=0, y=0 处的 p=0 约束

    # 单独处理角点
    boundary_loss = (
            torch.mean((u[:, :, 0, :] - zero_velocity) ** 2) +  # lower boundary: u = 0
            torch.mean((u[:, :, -1, :] - lid_velocity) ** 2) +  # upper boundary: u = 1 (lid)
            torch.mean((u[:, :, :-1, 0]) ** 2) +  # left boundary: u = 0
            torch.mean((u[:, :, :-1, -1]) ** 2) +  # right boundary: u = 0

            torch.mean((v[:, :, 0, :] - zero_velocity) ** 2) +  # lower boundary: v = 0
            torch.mean((v[:, :, -1, :] - zero_velocity) ** 2) +  # upper boundary: v = 0
            torch.mean((v[:, :, :, 0] - zero_velocity) ** 2) +  # left boundary: v = 0
            torch.mean((v[:, :, :, -1] - zero_velocity) ** 2) +  # right boundary: v = 0
            p_zero_constraint
    )
    # 总损失
    total_loss = pde_loss + boundary_loss
    return total_loss

# 生成训练数据的函数
def generate_data(grid_size=40, batch_size=1):
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy= np.meshgrid(x, y, indexing='ij')
    X = np.stack([xx.ravel(), yy.ravel()], axis=1)
    X = np.tile(X, (batch_size, 1))
    return torch.tensor(X, dtype=torch.float32, requires_grad=True)

# 训练模型的函数
def train_pinn_model(model, optimizer, epochs, X_train):
    # 初始化最小损失值
    min_loss = float('inf')
    best_model_path = 'best_model.pth'

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = ns_loss(model, X_train)
        loss.backward()
        optimizer.step()

        # 保存损失最小时的模型
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': min_loss,
            }, best_model_path)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    print(f'Minimum Loss: {min_loss}')


if __name__ == "__main__":
    # 参数设置
    grid_size = 40
    batch_size = 1
    epochs = 2000
    learning_rate = 0.001
    Re = 100  # 雷诺数

    # 生成训练数据 (用于内部约束)
    X_train = generate_data(grid_size,batch_size)

    # 初始化模型和优化器
    model = CNNPINN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_pinn_model(model, optimizer, epochs, X_train)

    # 保存模型
    torch.save(model.state_dict(), 'best_model.pth')
