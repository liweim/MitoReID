import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse


def get_mean_std(folder):
    R_channel = 0
    G_channel = 0
    B_channel = 0
    R_channel_square = 0
    G_channel_square = 0
    B_channel_square = 0
    pixels_num = 0

    paths = glob.glob(f"{folder}/*/*/*.jpg")
    for path in tqdm(paths):
        img = Image.open(path)
        img = np.asarray(img) / 255.0
        h, w, _ = img.shape
        pixels_num += h * w

        R_temp = img[:, :, 0]
        R_channel += np.sum(R_temp)
        R_channel_square += np.sum(np.power(R_temp, 2.0))
        G_temp = img[:, :, 1]
        G_channel += np.sum(G_temp)
        G_channel_square += np.sum(np.power(G_temp, 2.0))
        B_temp = img[:, :, 2]
        B_channel = B_channel + np.sum(B_temp)
        B_channel_square += np.sum(np.power(B_temp, 2.0))

    R_mean = R_channel / pixels_num
    G_mean = G_channel / pixels_num
    B_mean = B_channel / pixels_num

    """   
    S^2
    = sum((x-x')^2 )/N = sum(x^2+x'^2-2xx')/N
    = {sum(x^2) + sum(x'^2) - 2x'*sum(x) }/N
    = {sum(x^2) + N*(x'^2) - 2x'*(N*x') }/N
    = {sum(x^2) - N*(x'^2) }/N
    = sum(x^2)/N - x'^2
    """

    R_std = np.sqrt(R_channel_square / pixels_num - R_mean * R_mean)
    G_std = np.sqrt(G_channel_square / pixels_num - G_mean * G_mean)
    B_std = np.sqrt(B_channel_square / pixels_num - B_mean * B_mean)

    print("R_mean, G_mean, B_mean: ", R_mean, G_mean, B_mean)
    print("R_std, G_std, B_std: ", R_std, G_std, B_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    args = parser.parse_args()
    get_mean_std(args.folder)



