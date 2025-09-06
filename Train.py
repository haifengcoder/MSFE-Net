# Train_UnetV5_modify.py
# Step3: 训练脚本（无监督重建 + 第一层MFE梯度一致性正则）
# 使用说明：
#   python Train_UnetV5_modify.py
# 依赖：
#   - nets/UnetV5_modify.py（Step1已替换）
#   - nets/Convs/GradientAwareMFE.py（Step2已替换）
#   - nets/coco_dataset.py
# 路径：
#   训练集：data/coco2014/train2014a
#   验证集：data/coco2014/val2014a
#   权重输出：nets/parameters/UnetV5_modify.pkl

import os
import time
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 项目内模块
from nets.UnetV5_modify import UNetV5
from nets.coco_dataset import COCODataset

# -------------------- 可配置参数 --------------------
experiment_name = 'UnetV5_modify'
gpu_device = "cuda:0" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-4
epochs = 4
batch_size = 16
display_step = 100
shuffle = True

# 数据与日志路径
project_addrsss = os.getcwd()
train_dir = os.path.join(project_addrsss, "data", "coco2014", "train2014a")
val_dir   = os.path.join(project_addrsss, "data", "coco2014", "val2014a")
log_dir   = os.path.join(project_addrsss, "nets", "train_record")
os.makedirs(log_dir, exist_ok=True)
log_path  = os.path.join(log_dir, experiment_name + "_log.txt")
param_dir = os.path.join(project_addrsss, "nets", "parameters")
os.makedirs(param_dir, exist_ok=True)
save_path = os.path.join(param_dir, "UnetV5_modify.pkl")

# 归一化（需与数据集transforms一致）
MEAN = 0.4500517361627943
STD  = 0.26465333914691797

# 损失权重
SSIM_WEIGHT = 0.2     # 重建中的SSIM权重
LAMBDA_G    = 0.1     # 梯度一致性正则权重


# -------------------- 工具函数 --------------------
def print_and_log(msg: str):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adjust_learning_rate(optimizer, base_lr, epoch):
    # 每2个epoch衰减到原来的0.8
    lr = base_lr * (0.8 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def denorm01(x_norm):
    # x_norm: [B,1,H,W]，将归一化张量反归一化到[0,1]
    x = x_norm * STD + MEAN
    return torch.clamp(x, 0.0, 1.0)


# --- SSIM 实现（与常见实现一致，要求输入[0,1]范围） ---
def create_gaussian_window(window_size, channel, sigma=1.5, device='cpu', dtype=torch.float32):
    gauss_1d = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
        dtype=dtype, device=device
    )
    gauss_1d = gauss_1d / gauss_1d.sum()
    _1D_window = gauss_1d.unsqueeze(1)  # [K,1]
    _2D_window = _1D_window @ _1D_window.t()  # [K,K]
    window = _2D_window.unsqueeze(0).unsqueeze(0).to(dtype=dtype, device=device)  # [1,1,K,K]
    window = window.expand(channel, 1, window_size, window_size).contiguous()     # [C,1,K,K] depthwise
    return window


def ssim(img1, img2, window_size=5, size_average=True):
    # img1,img2: [B,C,H,W]，要求在[0,1]
    B, C, H, W = img1.shape
    device = img1.device
    dtype  = img1.dtype
    window = create_gaussian_window(window_size, C, device=device, dtype=dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)

    mu1_sq  = mu1.pow(2)
    mu2_sq  = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def recon_loss(x_pred_norm, x_gt_norm):
    # 将两者反归一化到[0,1]后计算：MSE + SSIM_WEIGHT * (1-SSIM)
    x_pred = denorm01(x_pred_norm)
    x_gt   = denorm01(x_gt_norm)

    mse = F.mse_loss(x_pred, x_gt)
    ssim_val = ssim(x_pred, x_gt, window_size=5, size_average=True)
    l_ssim = 1.0 - ssim_val
    return mse + SSIM_WEIGHT * l_ssim, mse.detach(), l_ssim.detach()


def sobel_mag(img):
    # 输入 img: [B,1,H,W]，建议在[0,1]空间计算（这里直接用反归一化后的）
    B, C, H, W = img.shape
    device = img.device
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=img.dtype, device=device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(img, kx, padding=1)
    gy = F.conv2d(img, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    # 每张图按空间做min-max归一化到[0,1]
    mag_min = mag.amin(dim=(2, 3), keepdim=True)
    mag_max = mag.amax(dim=(2, 3), keepdim=True)
    mag01 = (mag - mag_min) / (mag_max - mag_min + 1e-6)
    return mag01


# -------------------- 前向辅助：获取第一层MFE的grad_map --------------------
def forward_with_grad_map(model: UNetV5, img):
    """
    重走一次与模型一致的前向，但在第一层MFE处取出 grad_map。
    返回：out_norm（重建输出，归一化域）、grad_map（[B,1,H,W]）
    """
    # 编码
    x1_conv = model.left_conv_1(img)
    x1_mfe, grad_map = model.left_MFEblock_1(x1_conv, return_grad=True)  # 关键：return_grad=True
    x1_att = model.left_att_1(x1_mfe)
    x1_down = model.down_1(x1_att)

    x2 = model.left_conv_2(x1_down)
    x2 = model.left_MFEblock_2(x2)
    x2 = model.left_att_2(x2)
    x2_down = model.down_2(x2)

    x3 = model.left_conv_3(x2_down)
    x3 = model.left_MFEblock_3(x3)
    x3 = model.left_att_3(x3)
    x3_down = model.down_3(x3)

    x4 = model.left_conv_4(x3_down)
    x4 = model.left_MFEblock_4(x4)
    x4 = model.left_att_4(x4)
    x4_down = model.down_4(x4)

    # 中间
    x5 = model.center_conv(x4_down)

    # 解码
    x6_up = model.up_1(x5, x4)
    x6 = model.right_conv_1(torch.cat([x6_up, x4], dim=1))

    x7_up = model.up_2(x6, x3)
    x7 = model.right_conv_2(torch.cat([x7_up, x3], dim=1))

    x8_up = model.up_3(x7, x2)
    x8 = model.right_conv_3(torch.cat([x8_up, x2], dim=1))

    x9_up = model.up_4(x8, x1_att)
    x9 = model.right_conv_4(torch.cat([x9_up, x1_att], dim=1))

    out = model.output(x9)  # 归一化域的一通道重建图
    return out, grad_map


# -------------------- 训练与验证 --------------------
def main():
    # 数据集与加载器
    data_transforms = None  # 在 COCODataset 内部已实现ToTensor+Normalize
    train_set = COCODataset(train_dir, transform=data_transforms, need_crop=False, need_augment=False)
    val_set   = COCODataset(val_dir,   transform=data_transforms, need_crop=False, need_augment=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,   num_workers=2, pin_memory=True)

    print_and_log(f"Train size: {len(train_set)} | Val size: {len(val_set)}")

    # 模型
    model = UNetV5().to(gpu_device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val = float('inf')
    since = time.time()

    for epoch in range(epochs):
        # 调整学习率
        adjust_learning_rate(optimizer, learning_rate, epoch)
        # 训练
        model.train()
        running_loss, running_mse, running_ssim, running_g = 0.0, 0.0, 0.0, 0.0
        iters = 0

        for i, input_norm in enumerate(train_loader):
            input_norm = input_norm.to(gpu_device, non_blocking=True)  # [B,1,H,W] 归一化域

            # 前向，取第一层grad_map
            out_norm, grad_map = forward_with_grad_map(model, input_norm)

            # 重建损失（反归一化到[0,1]后计算）
            loss_recon, mse_val, ssim_val = recon_loss(out_norm, input_norm)

            # 梯度一致性正则：target 用反归一化后的输入图做Sobel幅值
            input_denorm = denorm01(input_norm)
            grad_target  = sobel_mag(input_denorm).detach()   # [B,1,H,W] in [0,1]
            loss_grad    = F.l1_loss(grad_map, grad_target)

            loss = loss_recon + LAMBDA_G * loss_grad

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mse  += mse_val.item()
            running_ssim += (1.0 - ssim_val.item())  # 记录为SSIM损失
            running_g    += loss_grad.item()
            iters += 1

            if i % display_step == 0:
                print_and_log(f"[Train] Epoch {epoch+1}/{epochs} Iter {i}: "
                              f"Loss={loss.item():.4f} | MSE={mse_val.item():.4f} "
                              f"| Lssim={(1.0-ssim_val.item()):.4f} | Lgrad={loss_grad.item():.4f}")

        epoch_train_loss = running_loss / max(1, iters)
        print_and_log(f"[Train] Epoch {epoch+1} Avg: Loss={epoch_train_loss:.6f} "
                      f"| MSE={running_mse/max(1,iters):.6f} "
                      f"| Lssim={running_ssim/max(1,iters):.6f} "
                      f"| Lgrad={running_g/max(1,iters):.6f}")

        # 验证
        model.eval()
        val_loss, val_mse, val_lssim, val_lgrad = 0.0, 0.0, 0.0, 0.0
        v_iters = 0
        with torch.no_grad():
            for input_norm in val_loader:
                input_norm = input_norm.to(gpu_device, non_blocking=True)
                out_norm, grad_map = forward_with_grad_map(model, input_norm)

                loss_recon, mse_val, ssim_val = recon_loss(out_norm, input_norm)
                input_denorm = denorm01(input_norm)
                grad_target  = sobel_mag(input_denorm)
                loss_grad    = F.l1_loss(grad_map, grad_target)

                loss = loss_recon + LAMBDA_G * loss_grad

                val_loss  += loss.item()
                val_mse   += mse_val.item()
                val_lssim += (1.0 - ssim_val.item())
                val_lgrad += loss_grad.item()
                v_iters += 1

        epoch_val_loss = val_loss / max(1, v_iters)
        print_and_log(f"[Val]   Epoch {epoch+1} Avg: Loss={epoch_val_loss:.6f} "
                      f"| MSE={val_mse/max(1,v_iters):.6f} "
                      f"| Lssim={val_lssim/max(1,v_iters):.6f} "
                      f"| Lgrad={val_lgrad/max(1,v_iters):.6f}")

        # 保存最优
        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print_and_log(f"[Save] Updated best model at epoch {epoch+1}, val_loss={best_val:.6f}")

        # 计时
        elapsed = time.time() - since
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        print_and_log(f"[Time] {h}h {m}m {s}s")
        print_and_log("-" * 60)

    print_and_log(f"Training done. Best val loss: {best_val:.6f}")
    print_and_log(f"Best weights saved to: {save_path}")


if __name__ == "__main__":
    set_seed(1)
    main()
