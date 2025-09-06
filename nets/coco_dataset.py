# nets/coco_dataset.py
# Step4: 统一数据归一化（若未显式传入transform，则默认 ToTensor + Normalize(mean,std)）
# 与 Step3 训练脚本中使用的 MEAN/STD 保持一致

import os
import random
import cv2
import numpy as np
import PIL.Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

DEFAULT_MEAN = 0.4500517361627943
DEFAULT_STD  = 0.26465333914691797


class COCODataset(Dataset):
    def __init__(self,
                 input_dir,
                 crop_size=256,
                 transform=None,
                 need_crop=False,
                 need_augment=False,
                 mean: float = DEFAULT_MEAN,
                 std: float = DEFAULT_STD):
        """
        input_dir    : 包含单通道灰度图的文件夹
        crop_size    : 随机裁剪尺寸（need_crop=True 时生效）
        transform    : 若为None，则默认 ToTensor + Normalize(mean,std)
        need_crop    : 是否随机裁剪
        need_augment : 是否数据增强（水平翻转）
        mean/std     : 归一化参数（与训练脚本一致）
        """
        self._images_basename = sorted(os.listdir(input_dir))
        if '.ipynb_checkpoints' in self._images_basename:
            self._images_basename.remove('.ipynb_checkpoints')
        self._images_address = [os.path.join(input_dir, item) for item in self._images_basename]

        self._crop_size = crop_size
        self._need_crop = need_crop
        self._need_augment = need_augment

        if transform is None:
            self._transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([mean], [std])
            ])
        else:
            self._transform = transform

    def __len__(self):
        return len(self._images_address)

    def __getitem__(self, idx):
        # 读入灰度并归一化到[0,1]，然后统一resize到256x256
        image = cv2.imread(self._images_address[idx], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {self._images_address[idx]}")
        image = image.astype(np.float32) / 255.0
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        if self._need_crop:
            image = self._random_crop(image)

        # 转 PIL 再做变换（ToTensor + Normalize）
        image_pil = PIL.Image.fromarray(image.astype(np.float32))

        if self._need_augment:
            image_pil = self._rand_horizontal_flip(image_pil)

        image_tensor = self._transform(image_pil)
        return image_tensor  # [1,H,W]（已Normalize）

    def _random_crop(self, image_np):
        h, w = image_np.shape[:2]
        if h < self._crop_size or w < self._crop_size:
            return image_np  # 尺寸不够则不裁剪
        start_row = random.randint(0, h - self._crop_size)
        start_col = random.randint(0, w - self._crop_size)
        roi = image_np[start_row: start_row + self._crop_size,
                       start_col: start_col + self._crop_size]
        return roi

    def _rand_horizontal_flip(self, image_pil):
        if random.random() < 0.5:
            image_pil = F.hflip(image_pil)
        return image_pil
