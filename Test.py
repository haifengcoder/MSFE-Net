import os
from skimage import io


# from nets.UnetV5_modify import UnetV5_Fuse
from nets.UnetV5_visualization import UnetV5_Fuse

def main(input_dir, output_dir):
    """
    Image Fusion
    :param input_dir: str, input dir with all images stores in one folder
    :param output_dir: str, output dir with all fused images
    :return:
    """
    fuse = UnetV5_Fuse()
    images_name = sorted(list({item[:-6] for item in os.listdir(input_dir)}))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for  i, image_name in enumerate(images_name):
        print("Fusing {}".format(image_name))
        img1 = io.imread(os.path.join(input_dir, image_name + "_1.png"))
        img2 = io.imread(os.path.join(input_dir, image_name + "_2.png"))
        fused = fuse.fuse(img1, img2)
        io.imsave(os.path.join(output_dir,  f"{(i + 1):02d}.png"), fused)


if __name__ == "__main__":
    input_dir = os.path.join(os.getcwd(), "data", "multi_focus")
    output_dir = os.path.join(os.getcwd(), "data", "result")
    main(input_dir, output_dir)
