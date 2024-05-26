# 特征回旋 (vortices)：在标准的正方形腔体问题中，可以看到主回旋（大回旋）位于腔体的中心偏上部，同时在四个角可能会出现次回旋（较小的涡旋）。
# 速度分布：在流场中，流动在顶部以水平方向为主，然后逐渐向下扩散，逐渐在各个边边形成复杂的流动模式。

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class CNNPINN(nn.Module):
    def __init__(self, input_channels=2):  # Input channels: (x, y)
        super(CNNPINN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 40 * 40, 256)  # 假设输入尺寸为 (40, 40)
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

# 加载模型
model = CNNPINN()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 定义网格和时间步
grid_size = 40

# 生成网格点
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(x, y)

# 准备输入（假设输入是空间点和时间组合，这里需要根据实际情况调整）
def create_input_grid(X, Y, grid_size):
    input_tensor = np.zeros((1, 2, grid_size, grid_size))  # channels: x, y,
    input_tensor[0, 0, :, :] = X  # First channel: X
    input_tensor[0, 1, :, :] = Y  # Second channel: Y
    return torch.tensor(input_tensor, dtype=torch.float32)

# 预测并绘图
input_grid = create_input_grid(X, Y, grid_size)
with torch.no_grad():
    output = model(input_grid).detach().cpu().numpy()

u_pred = output[0, 0, :, :]
v_pred = output[0, 1, :, :]

# 计算速度大小
speed = np.sqrt(u_pred**2 + v_pred**2)

# 绘制速度大小的彩虹色图及流场图
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, speed, cmap='rainbow')  # 速度大小的彩虹色图
plt.colorbar(label='Speed')  # 添加颜色条
plt.quiver(X, Y, u_pred, v_pred, color='k')  # 流场矢量图，白色矢量
plt.title("Lid-driven Cavity Velocity Field")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("lid_cavity_xy_magnitude-flow-direction_bjplus.png")
plt.close()

