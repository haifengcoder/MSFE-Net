import torch
import torch.nn as nn
import torch.nn.functional as F

class FRA(nn.Module):
    """
    Focus-Reliability Attention
    1) 空间权重 W_s  – 由局部方差估计“清晰度可靠性”
    2) 通道权重 W_c – 经典 SE
    输出  F_out = X * W_s * W_c + X
    """

    def __init__(self, channels, k_size=3, r=16, eps=1e-5):
        """
        channels : 输入通道数 C
        k_size   : 计算局部统计的窗口大小
        r        : SE 降维比例
        """
        super().__init__()
        self.k = k_size
        self.eps = eps

        # 将方差图 (B×1×H×W) 转为空间注意力 W_s
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # SE 通道注意力
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x : B×C×H×W
        """
        # ---------- 1. 计算局部方差 ----------
        k = self.k
        mu   = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k//2)           # E[X]
        mu_2 = F.avg_pool2d(x * x, kernel_size=k, stride=1, padding=k//2)       # E[X²]
        var  = torch.clamp(mu_2 - mu * mu, min=0.0)                             # Var[X]

        # ---------- 2. 通道均值 → 1 通道 ----------
        var_mean = var.mean(dim=1, keepdim=True)                                # B×1×H×W

        # ---------- 3. 归一化到 [0,1] ----------
        v_min = var_mean.amin(dim=(2,3), keepdim=True)
        v_max = var_mean.amax(dim=(2,3), keepdim=True)
        var_norm = (var_mean - v_min) / (v_max - v_min + self.eps)              # B×1×H×W

        # ---------- 4. 空间注意力 ----------
        W_s = self.spatial_conv(var_norm)                                       # B×1×H×W

        # ---------- 5. 通道注意力 ----------
        W_c = self.channel_se(x)                                                # B×C×1×1

        # ---------- 6. 融合 ----------
        out = x * W_s * W_c + x
        return out
