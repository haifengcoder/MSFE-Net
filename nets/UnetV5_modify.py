# nets/UnetV5_modify.py

import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image
import torchvision.transforms as transforms
from skimage.color import rgb2gray

# 绝对导入（与项目结构保持一致）
from nets.Attention.FRA import FRA
from nets.Convs.GradientAwareMFE import GradientAwareMFE


# ------------------------------ 基础模块 ------------------------------
class up_conv(nn.Module):
    """
    上采样 + 3x3卷积 + BN + ReLU
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y_skip):
        # 双线性上采样到skip的尺寸
        x = F.interpolate(x, y_skip.size()[2:], mode='bilinear', align_corners=True)
        z = self.conv(x)
        z = self.bn(z)
        z = self.relu(z)
        return z


class DoubleConv(nn.Module):
    """
    3x3卷积-BN-ReLU x2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OneConv(nn.Module):
    """
    单层3x3卷积-BN-ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


# ------------------------------ 主网络 ------------------------------
class UNetV5(nn.Module):
    """
    编解码结构：
    - 编码：OneConv -> GradientAwareMFE -> FRA -> MaxPool
    - 解码：UpConv -> 拼接skip -> DoubleConv
    - 输出：1x1卷积得到重建图（训练用）
    - 推理融合：取解码端最后一层多通道特征（x9），用于ESFF决策图
    """
    def __init__(self):
        super().__init__()
        # 编码
        self.left_conv_1 = OneConv(1, 16)
        self.left_MFEblock_1 = GradientAwareMFE(16, [2, 4, 8])
        self.left_att_1 = FRA(16)
        self.down_1 = nn.MaxPool2d(2, 2)

        self.left_conv_2 = OneConv(16, 32)
        self.left_MFEblock_2 = GradientAwareMFE(32, [2, 4, 8])
        self.left_att_2 = FRA(32)
        self.down_2 = nn.MaxPool2d(2, 2)

        self.left_conv_3 = OneConv(32, 64)
        self.left_MFEblock_3 = GradientAwareMFE(64, [2, 4, 8])
        self.left_att_3 = FRA(64)
        self.down_3 = nn.MaxPool2d(2, 2)

        self.left_conv_4 = OneConv(64, 128)
        self.left_MFEblock_4 = GradientAwareMFE(128, [2, 4, 8])
        self.left_att_4 = FRA(128)
        self.down_4 = nn.MaxPool2d(2, 2)

        # 中间
        self.center_conv = DoubleConv(128, 256)

        # 解码（标准U-Net拼接）
        self.up_1 = up_conv(256, 128)
        self.right_conv_1 = DoubleConv(128 + 128, 128)

        self.up_2 = up_conv(128, 64)
        self.right_conv_2 = DoubleConv(64 + 64, 64)

        self.up_3 = up_conv(64, 32)
        self.right_conv_3 = DoubleConv(32 + 32, 32)

        self.up_4 = up_conv(32, 16)
        self.right_conv_4 = DoubleConv(16 + 16, 16)

        # 输出层（训练重建用）
        self.output = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)

    def _forward_single(self, img):
        """
        单路前向：返回最后的多通道解码特征 x9 以及输出重建图 out（训练用）
        """
        # 编码
        x1 = self.left_conv_1(img)
        x1 = self.left_MFEblock_1(x1)
        x1 = self.left_att_1(x1)
        x1_down = self.down_1(x1)

        x2 = self.left_conv_2(x1_down)
        x2 = self.left_MFEblock_2(x2)
        x2 = self.left_att_2(x2)
        x2_down = self.down_2(x2)

        x3 = self.left_conv_3(x2_down)
        x3 = self.left_MFEblock_3(x3)
        x3 = self.left_att_3(x3)
        x3_down = self.down_3(x3)

        x4 = self.left_conv_4(x3_down)
        x4 = self.left_MFEblock_4(x4)
        x4 = self.left_att_4(x4)
        x4_down = self.down_4(x4)

        # 中间
        x5 = self.center_conv(x4_down)

        # 解码
        x6_up = self.up_1(x5, x4)
        x6 = self.right_conv_1(torch.cat([x6_up, x4], dim=1))

        x7_up = self.up_2(x6, x3)
        x7 = self.right_conv_2(torch.cat([x7_up, x3], dim=1))

        x8_up = self.up_3(x7, x2)
        x8 = self.right_conv_3(torch.cat([x8_up, x2], dim=1))

        x9_up = self.up_4(x8, x1)
        x9 = self.right_conv_4(torch.cat([x9_up, x1], dim=1))  # 多通道解码特征

        out = self.output(x9)  # 训练重建用的一通道输出
        return x9, out

    def forward(self, phase, img1, img2=None, kernel_radius=5):
        """
        phase:
          - 'train': 单输入重建训练，返回 out（1通道）
          - 'fuse' : 双输入推理，返回 numpy 的二值决策图（HxW, uint8 0/1）
        """
        if phase == 'train':
            _, out = self._forward_single(img1)
            return out

        elif phase == 'fuse':
            with torch.no_grad():
                feat1, _ = self._forward_single(img1)  # [B,C,H,W]
                feat2, _ = self._forward_single(img2)  # [B,C,H,W]
                dm_np = self.enhanced_fusion_channel_sf(feat1, feat2, kernel_radius=kernel_radius)
            return dm_np

        else:
            raise ValueError(f"Unknown phase: {phase}")

    # -------------------- ESFF：四方向梯度 + 局部上下文 --------------------
    @staticmethod
    def enhanced_fusion_channel_sf(f1, f2, kernel_radius=5):
        """
        使用Sobel H/V + 两组对角（共四方向）计算多方向梯度能量，
        再用盒滤聚合上下文，跨通道求和得到空间频率SF，比较得到二值决策图。
        输入：f1,f2: [B,C,H,W] 多通道解码特征
        返回：numpy uint8 二值图 [H,W] 0/1
        """
        device = f1.device
        b, c, h, w = f1.shape

        # 四方向核（Sobel与对角近似）
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32, device=device)
        ky = kx.t()
        kd1 = torch.tensor([[-2, -1, 0],
                            [-1,  0, 1],
                            [ 0,  1, 2]], dtype=torch.float32, device=device)  # 对角 \
        kd2 = torch.tensor([[ 0,  1, 2],
                            [-1,  0, 1],
                            [-2, -1, 0]], dtype=torch.float32, device=device)  # 对角 /
        kernels = [k.reshape(1, 1, 3, 3).repeat(c, 1, 1, 1) for k in [kx, ky, kd1, kd2]]

        def spatial_freq(x):
            g = 0
            for k in kernels:
                gd = F.conv2d(x, k, padding=1, groups=c)
                g += gd.pow(2)  # 累加四方向梯度能量
            # 局部上下文聚合（盒滤）
            ksz = kernel_radius * 2 + 1
            box = torch.ones(c, 1, ksz, ksz, device=device) / float(ksz * ksz)
            g = F.conv2d(g, box, padding=kernel_radius, groups=c)  # [B,C,H,W]
            return g.sum(dim=1)  # 跨通道求和 -> [B,H,W]

        f1_sf = spatial_freq(f1)
        f2_sf = spatial_freq(f2)
        dm = (f1_sf > f2_sf).to(torch.uint8)  # [B,H,W] 0/1

        # 返回单张的二值numpy图
        return dm.squeeze(0).cpu().numpy().astype(np.uint8)


# ------------------------------ 融合器（推理入口） ------------------------------
class UnetV5_Fuse:
    """
    推理融合入口：
      - 灰度输入（通过MSFE-Net特征）生成决策图
      - 决策图形态学优化 + 导向滤波
      - 在YCrCb上仅融合Y通道，色度按掩码硬选源
    """
    def __init__(self):
        # 设备
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 模型
        self.model = UNetV5()
        self.model_path = os.path.join(os.getcwd(), "nets", "parameters", "UnetV5_modify.pkl")
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            print(f"[Warn] 未找到权重文件：{self.model_path}，将使用随机初始化参数。")
        self.model.to(self.device)
        self.model.eval()

        # 归一化（与你训练时一致）
        self.mean_value = 0.4500517361627943
        self.std_value = 0.26465333914691797
        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([self.mean_value], [self.std_value])
        ])

        # 决策图与导向滤波参数（可按需调整）
        self.kernel_radius = 5   # ESFF局部上下文半径
        self.area_ratio = 0.01   # 连通域面积阈值占比
        self.ks = 5              # 形态学结构元素大小
        self.gf_radius = 4       # 导向滤波半径
        self.eps = 0.1           # 导向滤波epsilon

        # 是否保存可视化的决策图
        self.save_dm = False
        self.dm_save_dir = os.path.join(os.getcwd(), "nets", "dmnp")
        if self.save_dm and not os.path.exists(self.dm_save_dir):
            os.makedirs(self.dm_save_dir, exist_ok=True)

    def fuse(self, img1, img2):
        """
        输入：img1, img2: numpy(H,W)灰度或(H,W,3)RGB，uint8
        输出：fused: numpy，同尺寸RGB或灰度
        """
        # 删除alpha通道
        if img1.ndim == 3 and img1.shape[-1] == 4:
            img1 = img1[:, :, :3]
        if img2.ndim == 3 and img2.shape[-1] == 4:
            img2 = img2[:, :, :3]

        is_color = (img1.ndim == 3 and img1.shape[2] == 3)

        # 转灰度供模型提取特征
        if is_color:
            img1_gray = rgb2gray(img1).astype(np.float32)
            img2_gray = rgb2gray(img2).astype(np.float32)
        else:
            img1_gray = img1.astype(np.float32) / 255.0 if img1.dtype != np.float32 else img1
            img2_gray = img2.astype(np.float32) / 255.0 if img2.dtype != np.float32 else img2

        # 构建张量
        img1_tensor = self.data_transforms(PIL.Image.fromarray(img1_gray)).unsqueeze(0).to(self.device)
        img2_tensor = self.data_transforms(PIL.Image.fromarray(img2_gray)).unsqueeze(0).to(self.device)

        # 生成初始决策图（0/1）
        dm = self.model.forward("fuse", img1_tensor, img2_tensor, kernel_radius=self.kernel_radius)

        # 决策图形态学优化（开运算去噪、去除小连通域、填小孔、闭运算平滑）
        dm = self.enhanced_decision_map_processor(dm, area_ratio=self.area_ratio, ks=self.ks)

        # 轮廓平滑（保留主要区域）
        dm = self.fixed_contour_based_smoothing(dm, area_ratio=self.area_ratio, min_dimension=3)

        # 可视化保存（可选）
        if self.save_dm:
            uid = str(uuid.uuid4())
            cv2.imwrite(os.path.join(self.dm_save_dir, f"{uid}-decision_map.png"), (dm * 255).astype(np.uint8))

        # Guided Filter 优化决策图
        dm = dm.astype(np.float32)
        if is_color:
            # 灰度引导（亮度近似）
            I_guide = (0.299 * img1[:, :, 0].astype(np.float32) +
                       0.587 * img1[:, :, 1].astype(np.float32) +
                       0.114 * img1[:, :, 2].astype(np.float32)) / 255.0
        else:
            I_guide = (img1.astype(np.float32) + img2.astype(np.float32)) / 2.0
            if I_guide.max() > 1.0:
                I_guide = I_guide / 255.0

        dm_guided = self.guided_filter(I_guide, dm, self.gf_radius, eps=self.eps)
        dm_guided = np.clip(dm_guided, 0.0, 1.0)

        # 在YCrCb空间融合（只融合Y），Cr/Cb按像素硬选源
        if is_color:
            fused = self._fuse_color_ycrcb(img1, img2, dm_guided)
        else:
            # 灰度直接加权
            img1f = img1.astype(np.float32)
            img2f = img2.astype(np.float32)
            if img1f.max() > 1.0:
                fused = img1f * dm_guided + img2f * (1.0 - dm_guided)
            else:
                fused = (img1f * dm_guided + img2f * (1.0 - dm_guided)) * 255.0
            fused = np.clip(fused, 0, 255).astype(np.uint8)

        return fused

    # ------------------------------ 工具：决策图后处理 ------------------------------
    @staticmethod
    def _elliptical_kernel(ks):
        ks = max(1, int(ks))
        if ks % 2 == 0:
            ks += 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

    @staticmethod
    def _remove_small_components(mask, min_area):
        """
        去除小前景连通域
        mask: uint8 0/1
        """
        mask_u8 = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        out = np.zeros_like(mask_u8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 1
        return out.astype(np.uint8)

    @staticmethod
    def _fill_small_holes(mask, min_area):
        """
        填充小孔：对 ~mask 做连通域，小面积且不接触边界的区域填充
        """
        inv = (mask == 0).astype(np.uint8)
        h, w = inv.shape
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        out = mask.copy().astype(np.uint8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, ww, hh, _ = stats[i]
            touches_border = (x == 0 or y == 0 or x + ww == w or y + hh == h)
            if (not touches_border) and (area < min_area):
                out[labels == i] = 1
        return out

    def enhanced_decision_map_processor(self, dm, area_ratio=0.01, ks=5):
        """
        决策图增强处理：
          - 开运算去噪
          - 去除小连通域
          - 填充小孔
          - 闭运算平滑
        输入 dm: numpy 0/1
        """
        if dm.dtype != np.uint8:
            dm = (dm > 0).astype(np.uint8)
        h, w = dm.shape[:2]
        min_area = max(1, int(area_ratio * h * w))

        se = self._elliptical_kernel(ks)

        # 开运算去噪
        dm_open = cv2.morphologyEx(dm, cv2.MORPH_OPEN, se)

        # 去除小连通域
        dm_large = self._remove_small_components(dm_open, min_area=min_area)

        # 填小孔
        dm_filled = self._fill_small_holes(dm_large, min_area=min_area)

        # 闭运算平滑边界
        dm_close = cv2.morphologyEx(dm_filled, cv2.MORPH_CLOSE, se)

        # 安全检查
        if np.all(dm_close == 0) or np.all(dm_close == 1):
            # 回退到简单的开闭
            dm_close = cv2.morphologyEx(dm, cv2.MORPH_OPEN, se)
            dm_close = cv2.morphologyEx(dm_close, cv2.MORPH_CLOSE, se)

        return dm_close.astype(np.uint8)

    def fixed_contour_based_smoothing(self, dm, area_ratio=0.01, min_dimension=3):
        """
        基于轮廓的平滑：保留面积足够或宽高足够的轮廓，填充其内部。
        避免产生全黑/全白的退化结果。
        """
        dm_u8 = (dm > 0).astype(np.uint8) * 255
        h, w = dm_u8.shape[:2]
        min_area = max(1, int(area_ratio * h * w))

        contours, _ = cv2.findContours(dm_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (dm > 0).astype(np.uint8)

        filtered = np.zeros_like(dm_u8)
        keep = []
        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if ((ww >= min_dimension or hh >= min_dimension) and area >= min_area):
                keep.append(cnt)

        if not keep:
            return (dm > 0).astype(np.uint8)

        cv2.drawContours(filtered, keep, -1, 255, cv2.FILLED)
        out = (filtered > 0).astype(np.uint8)
        return out

    # ------------------------------ 工具：导向滤波 ------------------------------
    @staticmethod
    def box_filter(img, r):
        """
        盒滤波（使用OpenCV实现）
        img: HxW 或 HxWx1/3，float32
        """
        if img.ndim == 2:
            return cv2.boxFilter(img, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
        else:
            out = np.zeros_like(img, dtype=np.float32)
            for c in range(img.shape[2]):
                out[:, :, c] = cv2.boxFilter(img[:, :, c], ddepth=-1, ksize=(2 * r + 1, 2 * r + 1),
                                             borderType=cv2.BORDER_REFLECT)
            return out

    def guided_filter(self, I, p, r, eps=0.1):
        """
        导向滤波（He et al.）
        I: 引导图，HxW 或 HxWx3，float32 [0,1]
        p: 待滤波图（决策图），HxW 或 HxWx1，float32 [0,1]
        """
        I = I.astype(np.float32)
        if I.max() > 1.0:
            I = I / 255.0
        if p.ndim == 3 and p.shape[2] == 1:
            p = p[:, :, 0]
        p = p.astype(np.float32)

        h, w = p.shape[:2]
        N = self.box_filter(np.ones((h, w), dtype=np.float32), r)

        if I.ndim == 2:
            mean_I = self.box_filter(I, r)
            mean_p = self.box_filter(p, r)
            corr_I = self.box_filter(I * I, r)
            corr_Ip = self.box_filter(I * p, r)
            var_I = corr_I - mean_I * mean_I
            cov_Ip = corr_Ip - mean_I * mean_p
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            mean_a = self.box_filter(a, r) / N
            mean_b = self.box_filter(b, r) / N
            q = mean_a * I + mean_b
            return np.clip(q, 0.0, 1.0).astype(np.float32)
        else:
            # 使用灰度引导
            I_gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
            return self.guided_filter(I_gray, p, r, eps)

    # ------------------------------ 颜色融合（YCrCb） ------------------------------
    @staticmethod
    def _fuse_color_ycrcb(img1, img2, dm_guided):
        """
        颜色融合：在YCrCb上仅融合Y通道；Cr/Cb按像素硬选源（mask>0.5选img1的色度）
        """
        if img1.dtype != np.uint8:
            img1 = np.clip(img1, 0, 255).astype(np.uint8)
        if img2.dtype != np.uint8:
            img2 = np.clip(img2, 0, 255).astype(np.uint8)

        ycc1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
        ycc2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
        Y1, Cr1, Cb1 = cv2.split(ycc1)
        Y2, Cr2, Cb2 = cv2.split(ycc2)

        # Y通道用连续权重
        Yf = (Y1.astype(np.float32) * dm_guided + Y2.astype(np.float32) * (1.0 - dm_guided)).astype(np.uint8)

        # 色度用硬选源（二值mask）
        mask = (dm_guided > 0.5).astype(np.uint8)
        Crf = (Cr1 * mask + Cr2 * (1 - mask)).astype(np.uint8)
        Cbf = (Cb1 * mask + Cb2 * (1 - mask)).astype(np.uint8)

        fused_ycc = cv2.merge([Yf, Crf, Cbf])
        fused_rgb = cv2.cvtColor(fused_ycc, cv2.COLOR_YCrCb2RGB)
        return fused_rgb
