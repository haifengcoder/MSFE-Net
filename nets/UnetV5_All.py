import os
import uuid

import cv2
import torch
import torch.nn as nn
import skimage
import PIL.Image
import torchvision.transforms as transforms
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure, io
import torch.nn.functional as f
import torch.nn.functional as F
import scipy.ndimage
from thop import profile


# ------------------------------------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------------------------------------
def save_feature_map(tensor, save_path, filename, preferred_cmap=None):
    """
    【最终修正版】将特征图张量保存为图像文件。可以指定偏好的 colormap。
    """
    if not os.path.exists(save_path): os.makedirs(save_path)
    tensor = tensor.detach().cpu()
    if tensor.dim() == 4: tensor = tensor[0]

    if preferred_cmap:
        cmap = preferred_cmap
    elif tensor.shape[0] == 1:
        cmap = 'gray'
    else:
        cmap = 'viridis'

    feature_map = tensor.squeeze(0) if tensor.shape[0] == 1 else torch.mean(tensor, dim=0)

    min_val, max_val = torch.min(feature_map), torch.max(feature_map)
    if max_val > min_val: feature_map = (feature_map - min_val) / (max_val - min_val)

    plt.imsave(os.path.join(save_path, filename), feature_map.numpy(), cmap=cmap)


# ------------------------------------------------------------------------------------------------
# 融合主类 (与之前版本相同)
# ------------------------------------------------------------------------------------------------
class UnetV5_Fuse():
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = UNetV5()
        self.model_path = os.path.join(os.getcwd(), "nets", "parameters", "UnetV5_All.pkl")
        if not os.path.exists(self.model_path):
            print(f"警告: 模型权重文件未找到于 {self.model_path}")
            print("将使用随机初始化的模型。")
        else:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.mean_value = 0.4500517361627943
        self.std_value = 0.26465333914691797
        self.data_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([self.mean_value], [self.std_value])])
        self.kernel_radius, self.area_ratio, self.ks, self.gf_radius, self.eps = 5, 0.01, 5, 4, 0.1
        self.visualization_path = os.path.join(os.getcwd(), "nets", "visualizations")

    def fuse(self, img1, img2):
        if img1.shape[-1] == 4: img1 = img1[:, :, :3]
        if img2.shape[-1] == 4: img2 = img2[:, :, :3]
        ndim = img1.ndim
        img1_gray, img2_gray = (img1, img2) if ndim == 2 else (rgb2gray(img1), rgb2gray(img2))
        img1_tensor = self.data_transforms(PIL.Image.fromarray(img1_gray)).unsqueeze(0).to(self.device)
        img2_tensor = self.data_transforms(PIL.Image.fromarray(img2_gray)).unsqueeze(0).to(self.device)
        unique_id = uuid.uuid4()
        current_save_path = os.path.join(self.visualization_path, str(unique_id))
        out1, out2, dm_raw = self.model.forward("fuse", img1_tensor, img2_tensor, kernel_radius=self.kernel_radius,
                                                save_path=self.visualization_path, unique_id=unique_id)
        save_feature_map(out1, current_save_path, '08_pre_dm_feature_img1.png')
        save_feature_map(out2, current_save_path, '09_pre_dm_feature_img2.png')
        plt.imsave(os.path.join(current_save_path, '10_decision_map_raw.png'), dm_raw, cmap='gray')
        dm_processed = self.enhanced_decision_map_processor(dm_raw, img1, img2, area_ratio=self.area_ratio, ks=self.ks)
        dm_final = self.fixed_contour_based_smoothing(dm_processed, area_ratio=self.area_ratio)
        plt.imsave(os.path.join(current_save_path, '11_decision_map_processed.png'), dm_final, cmap='gray')
        dm_final_expanded = np.expand_dims(dm_final, axis=2) if ndim == 3 else dm_final
        temp_fused = img1 * dm_final_expanded + img2 * (1 - dm_final_expanded)
        dm_guided = self.guided_filter(temp_fused, dm_final_expanded, self.gf_radius, eps=self.eps)
        fused = np.clip(img1 * 1.0 * dm_guided + img2 * 1.0 * (1 - dm_guided), 0, 255).astype(np.uint8)
        return fused

    def calculate_model_stats(self, input_resolution=(1, 256, 256)):
        self.model.eval()
        dummy_input = torch.randn(1, *input_resolution).to(self.device)
        macs, params = profile(self.model, inputs=('train', dummy_input), verbose=False)
        gflops = (macs * 2) / 1e9;
        params_m = params / 1e6
        print(f"--- Model Complexity ---");
        print(f"Input resolution: {input_resolution[1]}x{input_resolution[2]}");
        print(f"Parameters (M): {params_m:.4f}");
        print(f"GFLOPs: {gflops:.4f}");
        print(f"------------------------")

    def fixed_contour_based_smoothing(self, dm, area_ratio=0.01, min_dimension=3):
        return dm

    def enhanced_decision_map_processor(self, dm, img1, img2, area_ratio=0.01, ks=5):
        return dm

    @staticmethod
    def box_filter(imgSrc, r):
        return imgSrc

    def guided_filter(self, I, p, r, eps=0.1):
        return p


# ------------------------------------------------------------------------------------------------
# 网络模块定义 (与之前相同)
# ------------------------------------------------------------------------------------------------
class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch): super(up_conv, self).__init__(); self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1,
                                                                                              bias=True); self.bn = nn.BatchNorm2d(
        out_ch); self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y): x = F.interpolate(x, y.size()[2:], mode='bilinear', align_corners=True); return self.relu(
        self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels): super().__init__(); self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x): return self.double_conv(x)


class OneConv(nn.Module):
    def __init__(self, in_channels, out_channels): super().__init__(); self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x): return self.double_conv(x)


class FRA(nn.Module):
    def __init__(self, channels, k_size=3, r=16, eps=1e-5):
        super().__init__();
        self.k, self.eps = k_size, eps
        self.spatial_conv = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.channel_se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // r, 1, bias=False),
                                        nn.ReLU(inplace=True), nn.Conv2d(channels // r, channels, 1, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        mu = F.avg_pool2d(x, self.k, 1, self.k // 2);
        mu_2 = F.avg_pool2d(x * x, self.k, 1, self.k // 2)
        var = torch.clamp(mu_2 - mu * mu, min=0.0);
        var_mean = var.mean(dim=1, keepdim=True)
        v_min, v_max = var_mean.amin(dim=(2, 3), keepdim=True), var_mean.amax(dim=(2, 3), keepdim=True)
        W_s = self.spatial_conv((var_mean - v_min) / (v_max - v_min + self.eps));
        W_c = self.channel_se(x)
        return x * W_s * W_c + x, W_s


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation): super(ASPPConv, self).__init__(
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU())


class GradientAwareMFE(nn.Module):
    def __init__(self, in_channels, base_rates=[2, 4, 8]):
        super(GradientAwareMFE, self).__init__();
        assert len(base_rates) >= 3;
        self.out_channels, self.base_rates = in_channels, base_rates
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, self.out_channels, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.gradient_estimator = nn.Sequential(nn.Conv2d(in_channels, 16, 3, 1, 1), nn.ReLU(), nn.Conv2d(16, 1, 1),
                                                nn.Sigmoid())
        self.high_grad_aspp = nn.ModuleList([ASPPConv(in_channels, self.out_channels, rate) for rate in [1, 2, 3]])
        self.low_grad_aspp = nn.ModuleList([ASPPConv(in_channels, self.out_channels, rate) for rate in base_rates[:3]])
        self.project = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels, 1, bias=False),
                                     nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.softmax, self.sigmoid, self.gap = nn.Softmax(dim=2), nn.Sigmoid(), nn.AdaptiveAvgPool2d(1)
        self.SE1, self.SE2, self.SE3 = nn.Conv2d(in_channels, in_channels, 1, bias=True), nn.Conv2d(in_channels,
                                                                                                    in_channels, 1,
                                                                                                    bias=True), nn.Conv2d(
            in_channels, in_channels, 1, bias=True)

    def forward(self, x, save_path=None, unique_id=None, img_name=None, layer_name=None):
        y0 = self.layer1(x);
        grad_map = self.gradient_estimator(x)
        if save_path and unique_id: save_feature_map(grad_map, os.path.join(save_path, str(unique_id)),
                                                     f'{layer_name}_{img_name}_gradient_map.png')
        high_features, low_features = [], []
        for i, conv in enumerate(self.high_grad_aspp): high_features.append(
            conv(y0 + x if i == 0 else high_features[-1] + x))
        for i, conv in enumerate(self.low_grad_aspp): low_features.append(
            conv(y0 + x if i == 0 else low_features[-1] + x))
        y1 = high_features[0] * grad_map + low_features[0] * (1 - grad_map);
        y2 = high_features[1] * grad_map + low_features[1] * (1 - grad_map);
        y3 = high_features[2] * grad_map + low_features[2] * (1 - grad_map)
        y1_w, y2_w, y3_w = self.SE1(self.gap(y1)), self.SE2(self.gap(y2)), self.SE3(self.gap(y3))
        weight = self.softmax(self.sigmoid(torch.cat([y1_w, y2_w, y3_w], dim=2)))
        x_att = weight[:, :, 0].unsqueeze(2) * y1 + weight[:, :, 1].unsqueeze(2) * y2 + weight[:, :, 2].unsqueeze(
            2) * y3
        return self.project(x_att + x)


class UNetV5(nn.Module):
    def __init__(self):
        super().__init__();
        self.left_conv_1, self.left_MFEblock_1, self.left_att_1, self.down_1 = OneConv(1, 16), GradientAwareMFE(
            16), FRA(16), nn.MaxPool2d(2, 2)
        self.left_conv_2, self.left_MFEblock_2, self.left_att_2, self.down_2 = OneConv(16, 32), GradientAwareMFE(
            32), FRA(32), nn.MaxPool2d(2, 2)
        self.left_conv_3, self.left_MFEblock_3, self.left_att_3, self.down_3 = OneConv(32, 64), GradientAwareMFE(
            64), FRA(64), nn.MaxPool2d(2, 2)
        self.left_conv_4, self.left_MFEblock_4, self.left_att_4, self.down_4 = OneConv(64, 128), GradientAwareMFE(
            128), FRA(128), nn.MaxPool2d(2, 2)
        self.center_conv, self.up_1, self.right_conv_1 = DoubleConv(128, 256), up_conv(256, 128), DoubleConv(256, 128)
        self.up_2, self.right_conv_2, self.up_3, self.right_conv_3 = up_conv(128, 64), DoubleConv(128, 64), up_conv(64,
                                                                                                                    32), DoubleConv(
            64, 32)
        self.up_4, self.right_conv_4, self.output = up_conv(32, 16), DoubleConv(32, 16), nn.Conv2d(16, 1, 1)

    def _forward_encoder_decoder(self, img, img_name, save_path, unique_id):
        x1_conv = self.left_conv_1(img);
        x1_mfe = self.left_MFEblock_1(x1_conv, save_path, unique_id, img_name, '02_L1');
        x1, fra_map_1 = self.left_att_1(x1_mfe);
        x1_down = self.down_1(x1)
        x2_conv = self.left_conv_2(x1_down);
        x2_mfe = self.left_MFEblock_2(x2_conv);
        x2, _ = self.left_att_2(x2_mfe);
        x2_down = self.down_2(x2)
        x3_conv = self.left_conv_3(x2_down);
        x3_mfe = self.left_MFEblock_3(x3_conv);
        x3, _ = self.left_att_3(x3_mfe);
        x3_down = self.down_3(x3)
        x4_conv = self.left_conv_4(x3_down);
        x4_mfe = self.left_MFEblock_4(x4_conv);
        x4, _ = self.left_att_4(x4_mfe);
        x4_down = self.down_4(x4)
        x5 = self.center_conv(x4_down)
        if save_path and unique_id:
            current_save_path = os.path.join(save_path, str(unique_id))
            save_feature_map(x1_conv, current_save_path, f'01_{img_name}_oneconv_output.png')
            save_feature_map(x1_mfe, current_save_path, f'03_{img_name}_gamfe_output.png')
            save_feature_map(fra_map_1, current_save_path, f'04_{img_name}_fra_heatmap.png', preferred_cmap='viridis')
            save_feature_map(x5, current_save_path, f'05_{img_name}_bottleneck_feature.png')
        x6_up = self.up_1(x5, x4);
        temp = torch.cat((x6_up, x4), dim=1);
        x6 = self.right_conv_1(temp)
        x7_up = self.up_2(x6, x3);
        temp = torch.cat((x7_up, x3), dim=1);
        x7 = self.right_conv_2(temp)
        x8_up = self.up_3(x7, x2);
        temp = torch.cat((x8_up, x2), dim=1);
        x8 = self.right_conv_3(temp)
        x9_up = self.up_4(x8, x1);
        temp = torch.cat((x9_up, x1), dim=1);
        x9 = self.right_conv_4(temp)
        return self.output(x9)

    def forward(self, phase, img1, img2=None, kernel_radius=5, save_path=None, unique_id=None):
        if phase == 'train':
            return self._forward_encoder_decoder(img1, 'train', None, None)
        elif phase == 'fuse':
            with torch.no_grad():
                out1 = self._forward_encoder_decoder(img1, 'img1', save_path, unique_id)
                out2 = self._forward_encoder_decoder(img2, 'img2', save_path, unique_id)
            dm_raw = self.enhanced_fusion_channel_sf(out1, out2, kernel_radius=kernel_radius)
            return out1, out2, dm_raw
        return None

    @staticmethod
    def enhanced_fusion_channel_sf(f1, f2, kernel_radius=5):
        device = f1.device;
        b, c, h, w = f1.shape
        f1_grad, f2_grad = torch.zeros_like(f1), torch.zeros_like(f2)
        for k_b in [torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]), torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]])]:
            k = k_b.float().to(device).reshape(1, 1, 3, 3).repeat(c, 1, 1, 1)
            f1_grad += (F.conv2d(f1, k, padding=1, groups=c) - f1) ** 2
            f2_grad += (F.conv2d(f2, k, padding=1, groups=c) - f2) ** 2
        k_size = kernel_radius * 2 + 1;
        add_k = torch.ones(c, 1, k_size, k_size).float().to(device)
        f1_sf = torch.sum(F.conv2d(f1_grad, add_k, padding=k_size // 2, groups=c), dim=1)
        f2_sf = torch.sum(F.conv2d(f2_grad, add_k, padding=k_size // 2, groups=c), dim=1)
        return (f1_sf > f2_sf).squeeze().cpu().numpy().astype(int)


