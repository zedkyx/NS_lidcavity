import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


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

    def loss(self, x, boundary_points, re):
        uvp = self.forward(x)
        # print("uvp",uvp.shape)
        u = uvp[:, 0]
        v = uvp[:, 1]
        p = uvp[:, 2]
        t = uvp[:, 3]
        e = uvp[:, 4]

        # print("x",x.shape)

        u_grad = torch.autograd.grad(uvp[:, 0], x, torch.ones_like(uvp[:, 0]), create_graph=True, retain_graph=True)[0]
        v_grad = torch.autograd.grad(uvp[:, 1], x, torch.ones_like(uvp[:, 1]), create_graph=True, retain_graph=True)[0]
        p_grad = torch.autograd.grad(uvp[:, 2], x, torch.ones_like(uvp[:, 2]), create_graph=True, retain_graph=True)[0]
        t_grad = torch.autograd.grad(uvp[:, 3], x, torch.ones_like(uvp[:, 3]), create_graph=True, retain_graph=True)[0]

        # print(u_grad)
        # print(u_grad.shape) #torch.Size([400, 2])
        u_x = u_grad[:, 0]
        u_y = u_grad[:, 1]
        v_x = v_grad[:, 0]
        v_y = v_grad[:, 1]
        p_x = p_grad[:, 0]
        p_y = p_grad[:, 1]
        t_x = t_grad[:, 0]
        t_y = t_grad[:, 1]

        u_grad2 = torch.autograd.grad(u_grad, x, torch.ones_like(u_grad), create_graph=True, retain_graph=True)[0]
        v_grad2 = torch.autograd.grad(v_grad, x, torch.ones_like(v_grad), create_graph=True, retain_graph=True)[0]
        t_grad2 = torch.autograd.grad(t_grad, x, torch.ones_like(t_grad), create_graph=True, retain_graph=True)[0]
        # print(u_grad2.shape)
        u_xx = u_grad2[:, 0]
        u_yy = u_grad2[:, 1]
        v_xx = v_grad2[:, 0]
        v_yy = v_grad2[:, 1]
        t_xx = t_grad2[:, 0]
        t_yy = t_grad2[:, 1]
        # print("u_xx",u_xx)

        # Navier-Stokes方程残差 Ra = 1
        pr = 7
        gr = 1e4 / pr
        # ev(remain 0)
        alpha_evm = 0.03
        vis_t_minus = alpha_evm * torch.abs(e)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vis_t0 = torch.tensor(5 * 1e-7 * 7, device=device)
        ve = torch.minimum(vis_t0, vis_t_minus)
        # Save vis_t_minus for computing vis_t in the next step

        momentum_x = u * u_x + v * u_y + p_x - (1.0 + ve) * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_x - (1.0 + ve) * (v_xx + v_yy) - gr * t
        continuity = u_x + v_y
        energy = u * t_x + v * t_y - (1.0 / pr) * (t_xx + t_yy)
        residual = u * momentum_x + v * momentum_y - e


        # 内部点损失
        loss_interior = torch.mean(momentum_x ** 2) + torch.mean(momentum_y ** 2) + torch.mean(continuity ** 2) \
                        + torch.mean(energy ** 2) + torch.mean(residual ** 2)

        # 边界点损失
        uvp_boundary = self.forward(boundary_points)
        u_boundary = uvp_boundary[:, 0]
        v_boundary = uvp_boundary[:, 1]
        t_boundary = uvp_boundary[:, 3]
        t_left_boundary = t_boundary[:512]
        t_right_boundary = t_boundary[512:2 * 512]

        t_bottom_boundary = t_boundary[2 * 512:3 * 512]
        t_top_boundary = t_boundary[3 * 512:4 * 512]

        bottom_boundary_value = 1

        # 按照等温1-y设置
        y = boundary_points[:, 1]
        left_boundary_value = 1 - y[:512]
        # print(left_boundary_value)
        right_boundary_value = 1 - y[512:2 * 512]

        # 左右绝热边界条件
        loss_boundary_t = torch.mean((t_bottom_boundary - bottom_boundary_value) ** 2) + torch.mean(
            t_top_boundary ** 2) + torch.mean((t_left_boundary - left_boundary_value) ** 2) \
            + torch.mean((t_right_boundary - right_boundary_value) ** 2)
        # print("bc:", loss_boundary)
        # print("interior:", loss_interior)
        loss_boundary_u = torch.mean(u_boundary ** 2)
        loss_boundary_v = torch.mean(v_boundary ** 2)
        loss_boundary = loss_boundary_t + loss_boundary_u + loss_boundary_v
        # 总损失
        in_weight = 1
        bc_weight = 1
        loss = loss_interior * in_weight + loss_boundary * bc_weight
        return loss
