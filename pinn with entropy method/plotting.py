import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class ANN_PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(ANN_PINN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


class PINN(nn.Module):
    def __init__(self, conv_net):
        super(PINN, self).__init__()
        self.conv_net = conv_net

    def forward(self, x):
        return self.conv_net(x)


def plot_flow_field(pinn, grid_size=512):
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    points = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32)
    print(points)
    points = points.view(-1, 2)  # 调整为卷积网络需要的输入形状

    with torch.no_grad():
        uvp = pinn(points).numpy()
        u = uvp[:, 0].reshape(grid_size, grid_size)
        v = uvp[:, 1].reshape(grid_size, grid_size)

    magnitude = np.sqrt(u ** 2 + v ** 2)
    print(magnitude)


    plt.figure(figsize=(8, 8))
    plt.streamplot(X, Y, u, v, density=2, linewidth=1, arrowsize=1)
    # plt.contourf(X, Y, magnitude, cmap='rainbow')
    # plt.colorbar(label='Velocity Magnitude')
    # plt.quiver(X, Y, u, v, color='white')
    plt.title('Flow Field Velocity Magnitude')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('square')  # 确保xy轴比例一致
    plt.show()



if __name__ == "__main__":
    input_dim = 2
    hidden_dim = 80
    output_dim = 5
    num_hidden_layers = 6
    ann_net = ANN_PINN(input_dim, hidden_dim, output_dim, num_hidden_layers)
    pinn = PINN(ann_net)
    pinn.load_state_dict(torch.load(r'reduiliu_ra_1e3_ev.pth'))
    pinn.eval()  # 切换模型到评估模式
    plot_flow_field(pinn)