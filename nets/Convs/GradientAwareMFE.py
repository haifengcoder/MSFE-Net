import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class GradientAwareMFE(nn.Module):
    def __init__(self, in_channels, base_rates=[2, 4, 8]):
        """
        梯度感知的多尺度特征提取模块（3 尺度融合：y1, y2, y3；y0 不参与尺度注意力）
        Args:
            in_channels: 输入通道数
            base_rates: 低梯度分支的空洞率设置（需至少 3 个，默认 [2,4,8]）
        """
        super(GradientAwareMFE, self).__init__()
        assert len(base_rates) >= 3, "base_rates 至少需要 3 个空洞率"
        self.out_channels = in_channels
        self.base_rates = base_rates

        # 基础卷积层（y0）
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

        # 梯度估计模块
        self.gradient_estimator = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 高梯度分支（细节）：d = {1,2,3}
        self.high_grad_aspp = nn.ModuleList([
            ASPPConv(in_channels, self.out_channels, rate)
            for rate in [1, 2, 3]
        ])

        # 低梯度分支（上下文）：d = {2,4,8}
        self.low_grad_aspp = nn.ModuleList([
            ASPPConv(in_channels, self.out_channels, rate)
            for rate in base_rates[:3]
        ])

        # 输出投影
        self.project = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

        # 尺度注意力（仅对 y1,y2,y3 做权重）
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.SE1 = nn.Conv2d(in_channels, in_channels, 1, bias=True)  # y1
        self.SE2 = nn.Conv2d(in_channels, in_channels, 1, bias=True)  # y2
        self.SE3 = nn.Conv2d(in_channels, in_channels, 1, bias=True)  # y3

        # 中间特征（可视化用）
        self.high_grad_features = []
        self.low_grad_features = []

    def forward(self, x):
        # 基础卷积层
        y0 = self.layer1(x)

        # 估计梯度图
        grad_map = self.gradient_estimator(x)

        # 清空中间特征缓存
        self.high_grad_features = []
        self.low_grad_features = []

        # 高梯度卷积特征（3 个尺度）
        for i, conv in enumerate(self.high_grad_aspp):
            if i == 0:
                y_high = conv(y0 + x)
            else:
                y_high = conv(self.high_grad_features[-1] + x)
            self.high_grad_features.append(y_high)

        # 低梯度卷积特征（3 个尺度）
        for i, conv in enumerate(self.low_grad_aspp):
            if i == 0:
                y_low = conv(y0 + x)
            else:
                y_low = conv(self.low_grad_features[-1] + x)
            self.low_grad_features.append(y_low)

        # 门控融合得到 3 个尺度
        y1 = self.high_grad_features[0] * grad_map + self.low_grad_features[0] * (1 - grad_map)
        y2 = self.high_grad_features[1] * grad_map + self.low_grad_features[1] * (1 - grad_map)
        y3 = self.high_grad_features[2] * grad_map + self.low_grad_features[2] * (1 - grad_map)

        # 仅对 y1,y2,y3 做尺度注意力
        y1_weight = self.SE1(self.gap(y1))
        y2_weight = self.SE2(self.gap(y2))
        y3_weight = self.SE3(self.gap(y3))

        weight = torch.cat([y1_weight, y2_weight, y3_weight], dim=2)  # [N,C,3,1]
        weight = self.softmax(self.sigmoid(weight))

        w1 = weight[:, :, 0].unsqueeze(2)  # [N,C,1,1]
        w2 = weight[:, :, 1].unsqueeze(2)
        w3 = weight[:, :, 2].unsqueeze(2)

        # 三尺度加权融合（y0 不参与权重；如需可无权重地加上 y0：x_att += y0）
        x_att = w1 * y1 + w2 * y2 + w3 * y3

        # 投影 + 与输入残差相加
        return self.project(x_att + x)
