import os
import time  # 计时用
from skimage import io

# 假设您的 UnetV5_Fuse 类保存在 nets/UnetV5_All.py 文件中
from nets.UnetV5_All import UnetV5_Fuse


# from nets.sesf_net import SESF_Fuse # 这个导入未被使用，可以移除或注释掉

def main(input_dir, output_dir):
    """
    Image Fusion
    :param input_dir: str, input dir with all images stores in one folder
    :param output_dir: str, output dir with all fused images
    :return:
    """
    # 1. 初始化模型
    print("--- Initializing Model ---")
    fuse = UnetV5_Fuse()
    print("Model initialized.")
    print("=" * 40)

    # 2. 【改动】计算并打印模型复杂度
    print("--- Calculating Model Complexity ---")
    # 此处使用一个典型的输入分辨率 (1, 256, 256) 来计算FLOPs
    # 参数量(Parameters)与输入尺寸无关，但FLOPs与之相关
    fuse.calculate_model_stats(input_resolution=(1, 256, 256))
    print("=" * 40)

    # 3. 开始批量融合与计时
    print("--- Starting Batch Fusion Process ---")
    images_name = sorted(list({item[:-6] for item in os.listdir(input_dir)}))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_time = 0.0  # 累计耗时
    num_images = len(images_name)

    if num_images == 0:
        print(f"在目录 '{input_dir}' 中未找到符合命名规范 (*_1.png, *_2.png) 的图像对。")
        return

    for i, image_name in enumerate(images_name):
        print(f"[{i + 1}/{num_images}] Fusing {image_name}")
        img1_path = os.path.join(input_dir, image_name + "_1.png")
        img2_path = os.path.join(input_dir, image_name + "_2.png")

        # 检查图像文件是否存在
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f"  - 警告: 图像对 '{image_name}' 不完整，已跳过。")
            continue

        img1 = io.imread(img1_path)
        img2 = io.imread(img2_path)

        start = time.time()  # 开始计时
        fused = fuse.fuse(img1, img2)
        elapsed = time.time() - start
        total_time += elapsed  # 累加

        print(f"  - Done in {elapsed:.4f} s")  # 单张耗时
        io.imsave(os.path.join(output_dir, f"fused_{image_name}.png"), fused)  # 【改动】保存文件名更具可读性

    print("=" * 40)
    print("--- Fusion Summary ---")
    print(f"Total images processed: {num_images}")
    if num_images > 0:
        print(f"Average time per image pair: {total_time / num_images:.4f} s")
    print(f"All fused images have been saved to: {output_dir}")
    print("=" * 40)


if __name__ == "__main__":
    # 请根据您的目录结构调整这里的路径
    input_dir = os.path.join(os.getcwd(), "data", "multi_focus")
    output_dir = os.path.join(os.getcwd(), "data", "result")

    # 确保输入目录存在
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在 -> {input_dir}")
        print("请创建该目录并将您的多聚焦图像对（例如 'desk_1.png' 和 'desk_2.png'）放入其中。")
    else:
        main(input_dir, output_dir)
