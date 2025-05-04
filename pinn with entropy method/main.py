import torch
import torch.nn as nn
import torch.optim as optim
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from net import ANN_PINN, PINN
from data import x, boundary_points

import psutil

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # 以MB为单位

def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.9

def plot_velocity_field(pinn, x, device, epoch, figure_path):
    pinn.eval()
    with torch.no_grad():
        x = x.to(device)
        uvp = pinn(x)
        u = uvp[:, 0].cpu().numpy()
        v = uvp[:, 1].cpu().numpy()

        # 计算速度大小
        speed = np.sqrt(u ** 2 + v ** 2)

        x = x.cpu().numpy()
        x_coords = x[:, 0]
        y_coords = x[:, 1]

        plt.figure(figsize=(10, 8))

        # 使用scatter绘制速度大小的彩虹图
        scatter = plt.scatter(x_coords, y_coords, c=speed, cmap='rainbow', alpha=0.5)
        plt.colorbar(scatter, label='Speed')

        plt.title(f'Velocity Field at Epoch {epoch}')
        plt.xlabel('x')
        plt.ylabel('y')

        # 保存图像
        image_save_path = os.path.join(figure_path, f'velocity_field_epoch_{epoch}.png')
        plt.savefig(image_save_path)
        plt.close()  # 关闭图像，释放内存
        print(f'Velocity field plot saved to {image_save_path}')
    pinn.train()

def train_pinn(pinn, optimizer, epochs, x, boundary_points, device, save_path, figure_path, re=1000, lr_decay=None):
    pinn = pinn.to(device)
    x = x.to(device)
    boundary_points = boundary_points.to(device)

    # 创建保存文件夹
    os.makedirs(figure_path, exist_ok=True)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        optimizer.zero_grad()
        loss = pinn.loss(x, boundary_points, re=re)
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

        if epoch % 1000 == 0:
            torch.save(pinn.state_dict(), save_path)
            plot_velocity_field(pinn, x, device, epoch, figure_path)  # 调用绘图函数并保存图像

        del loss
        if epoch % 1000 == 0:
            gc.collect()  # 控制GC的频率

        if lr_decay and epoch % lr_decay == 0:
            adjust_learning_rate(optimizer, lr_decay)

    # 最终保存模型参数
    torch.save(pinn.state_dict(), save_path)
    print(f'Final model parameters saved to {save_path}')

if __name__ == '__main__':
    input_dim = 2
    hidden_dim = 80
    output_dim = 5
    num_hidden_layers = 6
    epochs = 300000
    lr = 1e-4
    figure_path = r'C:\Users\Administrator\Desktop\ra_1e3'
    save_path = 'ra_1e3.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # memory_usage = get_memory_usage()
    # print(f"当前进程的内存使用情况: {memory_usage:.2f} MB")
    gc.collect()
    torch.cuda.empty_cache()

    ann_net = ANN_PINN(input_dim, hidden_dim, output_dim, num_hidden_layers)
    pinn = PINN(ann_net)
    pinn.load_state_dict(torch.load('reduiliu_ra_1.5e5_ev_lr_1e-3.pth', map_location=device))
    #
    optimizer = optim.Adam(pinn.parameters(), lr=lr)
    train_pinn(pinn, optimizer, epochs, x, boundary_points, device, save_path, figure_path)
