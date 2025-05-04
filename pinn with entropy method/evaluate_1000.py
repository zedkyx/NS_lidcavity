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
    x = 0.5
    y = [1, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
         0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0]
    u_theory = np.array([1.00000, 0.65928, 0.57492, 0.51117, 0.46604,
                         0.33304, 0.18719, 0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.00000])

    X, Y = np.meshgrid(x, y)
    points = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32)
    print(points)
    points = points.view(-1, 2)  # 调整为卷积网络需要的输入形状

    with torch.no_grad():
        uvp = pinn(points).numpy()
        u = uvp[:, 0]
        v = uvp[:, 1]

    print("u", u)
    print("v", v)
    magnitude = np.sqrt(u ** 2 + v ** 2)
    print(magnitude)

    # 绘制速度剖面比较图
    plt.figure(figsize=(10, 6))
    plt.plot(u_theory, y, marker='o', label='Theoretical')
    plt.plot(u, y, marker='x', label='Model Prediction')
    plt.xlabel('Velocity u')
    plt.ylabel('Position y')
    plt.title('Comparison of Velocity Profile at x = 0.5')
    # plt.gca().invert_yaxis()  # 反转y轴，使y=1在上方
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    input_dim = 2
    hidden_dim = 80
    output_dim = 3
    num_hidden_layers = 6
    ann_net = ANN_PINN(input_dim, hidden_dim, output_dim, num_hidden_layers)
    pinn = PINN(ann_net)
    pinn.load_state_dict(torch.load(r'pinn_model_1000_grid_5.pth'))
    pinn.eval()  # 切换模型到评估模式
    plot_flow_field(pinn)