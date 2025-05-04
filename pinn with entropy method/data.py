import torch
import torch.nn as nn
import numpy as np


def generate_training_data(num_points=200):
    x = torch.linspace(0, 1, num_points, dtype=torch.float32, requires_grad=True)
    y = torch.linspace(0, 1, num_points, dtype=torch.float32, requires_grad=True)

    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid_x = grid_x.reshape(-1, 1)
    grid_y = grid_y.reshape(-1, 1)
    x = torch.cat((grid_x, grid_y), dim=1)

    x = x.view(-1, 2)  # 调整为卷积网络需要的输入形状
    return x

# 生成每条边上的点
def generate_boundary_points(num_points_per_side=512):
    x_left = torch.zeros(num_points_per_side, dtype=torch.float32).view(-1, 1)
    y_left = torch.linspace(0, 1, num_points_per_side, dtype=torch.float32).view(-1, 1)
    left_boundary = torch.cat((x_left, y_left), dim=1)

    x_right = torch.ones(num_points_per_side, dtype=torch.float32).view(-1, 1)
    y_right = torch.linspace(0, 1, num_points_per_side, dtype=torch.float32).view(-1, 1)
    right_boundary = torch.cat((x_right, y_right), dim=1)

    x_bottom = torch.linspace(0, 1, num_points_per_side, dtype=torch.float32).view(-1, 1)
    y_bottom = torch.zeros(num_points_per_side, dtype=torch.float32).view(-1, 1)
    bottom_boundary = torch.cat((x_bottom, y_bottom), dim=1)

    x_top = torch.linspace(0, 1, num_points_per_side, dtype=torch.float32).view(-1, 1)
    y_top = torch.ones(num_points_per_side, dtype=torch.float32).view(-1, 1)
    top_boundary = torch.cat((x_top, y_top), dim=1)

    boundary_points = torch.cat((left_boundary, right_boundary, bottom_boundary, top_boundary), dim=0)
    boundary_points = boundary_points.view(-1,2)  # 调整为卷积网络需要的输入形状
    #print(boundary_points)

    return boundary_points


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = generate_training_data().to(device)
print(x.shape)
boundary_points = generate_boundary_points().to(device)
print(boundary_points.shape)
