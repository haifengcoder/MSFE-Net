# nets/lp_lssim_loss.py
# 可选：稳定版重建损失（MSE + ssim_weight * (1-SSIM)），输入张量为Normalize后的域
# 若使用本Loss，可在训练脚本中替换自定义的recon_loss调用

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconSSIMLoss(nn.Module):
    def __init__(self,
                 mean: float = 0.4500517361627943,
                 std: float = 0.26465333914691797,
                 ssim_weight: float = 0.2,
                 window_size: int = 5,
                 reduction: str = "mean"):
        """
        mean/std    : 反归一化参数（与数据预处理一致）
        ssim_weight : SSIM项的权重（建议 0.1~1.0）
        window_size : SSIM的高斯窗口
        reduction   : 'mean'（目前仅支持）
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.ssim_weight = ssim_weight
        self.window_size = window_size
        self.reduction = reduction

    @staticmethod
    def _create_gaussian_window(window_size, channel, sigma=1.5, device='cpu', dtype=torch.float32):
        gauss_1d = torch.tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
            dtype=dtype, device=device
        )
        gauss_1d = gauss_1d / gauss_1d.sum()
        _1D = gauss_1d.unsqueeze(1)                    # [K,1]
        _2D = _1D @ _1D.t()                            # [K,K]
        window = _2D.unsqueeze(0).unsqueeze(0)         # [1,1,K,K]
        window = window.expand(channel, 1, window_size, window_size).contiguous()  # [C,1,K,K]
        return window

    def _ssim(self, x, y):
        # x,y ∈ [0,1]
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        K = self.window_size
        window = self._create_gaussian_window(K, C, device=device, dtype=dtype)

        mu_x = F.conv2d(x, window, padding=K // 2, groups=C)
        mu_y = F.conv2d(y, window, padding=K // 2, groups=C)

        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=K // 2, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=K // 2, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=K // 2, groups=C) - mu_xy

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
        return ssim_map.mean()

    def _denorm01(self, x_norm):
        # x_norm: Normalize域 -> 反归一化到[0,1]
        x = x_norm * self.std + self.mean
        return torch.clamp(x, 0.0, 1.0)

    def forward(self, image_in, image_out):
        """
        image_in  : 目标（Normalize域）
        image_out : 预测（Normalize域）
        返回：total_loss, mse_loss, ssim_loss
        """
        x = self._denorm01(image_in)
        y = self._denorm01(image_out)

        mse = F.mse_loss(y, x, reduction='mean')
        ssim_val = self._ssim(y, x)
        l_ssim = 1.0 - ssim_val

        total = mse + self.ssim_weight * l_ssim
        return total, mse.detach(), l_ssim.detach()
