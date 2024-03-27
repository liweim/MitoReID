from cellpose import models
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
import glob
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from scipy.ndimage import find_objects
import shutil
import random
from threading import Thread
import argparse

INPUT = 2048
WIDTH = 512
HALF_WIDTH = int(WIDTH / 2)
PAD = WIDTH
SHIFT = (WIDTH, WIDTH)
BORDER = 100
DEBUG = False


def merge_image(folder, lib):
    save_folder = f'E:\dataset\mito\mito_raw\{lib}'
    for w1_path in tqdm(glob.glob(folder + '/*/*_w1.tif')):
        w1 = cv2.imread(w1_path, cv2.IMREAD_GRAYSCALE)
        w2_path = w1_path.replace('_w1', '_w2')
        w2 = cv2.imread(w2_path, cv2.IMREAD_GRAYSCALE)
        w3_path = w1_path.replace('_w1', '_w3')
        w3 = cv2.imread(w3_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.merge([w1, w2, w3])
        img_path = osp.split(w1_path)[1]
        id, t, c = img_path.split('_')
        save_path = f'{save_folder}/{id}'
        if not osp.exists(save_path):
            os.makedirs(save_path)
        save_path += f'/{t}.jpg'
        cv2.imwrite(save_path, img)


def get_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def scale(img_np):
    img_ = np.uint8((img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np)) * 255)
    return img_


def contrast(img, a):
    img = Image.fromarray(img)
    enh_con = ImageEnhance.Contrast(img)
    img_ = enh_con.enhance(a)
    return np.asarray(img_)


def extract_nucleus(img):
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel(3), iterations=2)
    blur = cv2.GaussianBlur(opening, (5, 5), 0)
    ret, sure_bg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dist_transform = cv2.distanceTransform(sure_bg, 1, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 10 * dist_transform.mean(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    markers += 1

    markers[unknown == 255] = 0

    watershed_markers = cv2.watershed(cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR), markers)
    sure_bg[watershed_markers == -1] = 0
    erode = cv2.erode(sure_bg, get_kernel(3))
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = []
    areas = []
    for contour in contours:
        cnt = np.squeeze(contour)
        if len(cnt.shape) == 2:
            cnts.append(contour)
            areas.append(cv2.contourArea(contour))

    median_area = np.median(areas)
    rets = [(cnt, area) for cnt, area in zip(cnts, areas) if area > median_area * 0.3]
    cnts, areas = zip(*rets)
    return cnts, areas


def extract_cell(paths, imgs, masks, save_folder=None):
    all_cells, all_centers = [], []
    for idx, img_path in enumerate(paths):
        cells, centers, areas, intensities = [], [], [], []
        w1 = imgs[idx][:, :, 2]
        maski = masks[idx]
        outlines = np.zeros(maski.shape, bool)
        slices = find_objects(maski.astype(int))
        for i, si in enumerate(slices):
            if si is not None:
                sr, sc = si
                mask = (maski[sr, sc] == (i + 1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnt = np.concatenate(contours[-2], axis=0) + np.array([sc.start, sr.start])

                x_min, y_min, w, h = cv2.boundingRect(cnt)
                x_max = x_min + w
                y_max = y_min + h
                if x_min == 0 or y_min == 0 or x_max == INPUT or y_max == INPUT:
                    continue
                cells.append(cnt)
                x, y = np.mean(np.squeeze(cnt), 0).astype(int).tolist()
                intensity = w1[y, x]
                intensities.append(intensity)
                areas.append(cv2.contourArea(cnt))
                centers.append(np.array([x, y]))

        inlier_index = areas > np.mean(areas) / 3
        if idx == 0:
            inlier_index *= (intensities < np.mean(intensities) * 3)
        cells = [cell for idx, cell in zip(inlier_index, cells) if idx == True]
        if DEBUG:
            print(len(cells))
        all_cells.append(cells)
        centers = np.asarray(centers)[inlier_index]
        all_centers.append(centers)

        if DEBUG and save_folder is not None:
            if not osp.exists(save_folder):
                os.makedirs(save_folder)
            for cnt in cells:
                pvc, pvr = cnt.squeeze().T
                outlines[pvr, pvc] = 1
            outX, outY = np.nonzero(outlines)
            imgout = imgs[idx].copy()
            imgout[outX, outY] = np.array([255, 0, 0])
            ret = Image.fromarray(imgout)
            save_path = f'{save_folder}/{osp.split(img_path)[1]}'
            ret.save(save_path)
    return all_cells, all_centers


def generate_sequence(all_centers, all_cells, id, imgs, save_folder):
    fps = 2
    first_centers = all_centers[0]
    new_imgs = [cv2.copyMakeBorder(img[:, :, ::-1], PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT) for img in imgs]

    for x, y in first_centers:
        num = len(os.listdir(save_folder))
        filename = "{}_{}".format(id, str(num))

        base_center = (x, y)
        matched_cells = []
        bboxs = []
        flag = 0
        for centers, cells in zip(all_centers, all_cells):
            if len(centers) == 0:
                print(filename, "no cells")
                flag = 1
                break
            dists = [np.linalg.norm(pt - base_center) for pt in centers]
            idx = np.argmin(dists)
            if dists[idx] > 150:
                if DEBUG:
                    print(filename, dists[idx], "too far away")
                flag = 1
                break
            cell = cells[idx]
            base_center = centers[idx]
            matched_cells.append(cell)
            x_min, y_min, w, h = cv2.boundingRect(cell)
            x_max = x_min + w
            y_max = y_min + h
            bboxs.append([x_min, y_min, x_max, y_max])
        if flag:
            continue

        rgb_path = osp.join(save_folder, filename)
        if not osp.exists(rgb_path):
            os.makedirs(rgb_path)

        bboxs = np.asarray(bboxs)
        x_min = np.min(bboxs[:, 0])
        y_min = np.min(bboxs[:, 1])
        x_max = np.max(bboxs[:, 2])
        y_max = np.max(bboxs[:, 3])
        width = int(min(max(x_max - x_min, y_max - y_min), WIDTH) / 2)
        for j, (ori_img, cell, bbox) in enumerate(zip(new_imgs, matched_cells, bboxs)):
            x_min, y_min, x_max, y_max = bbox
            center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            x_min = center[0] - width + WIDTH
            y_min = center[1] - width + WIDTH
            x_max = center[0] + width + WIDTH
            y_max = center[1] + width + WIDTH

            new_img = np.zeros(ori_img.shape)
            cv2.drawContours(new_img, [cell + SHIFT], -1, (1, 1, 1), -1)
            new_img = cv2.blur(new_img, (11, 11))
            new_img = np.uint8(new_img * ori_img)
            new_img = new_img[y_min:y_max, x_min:x_max]

            save_path = osp.join(rgb_path, str(j + 1) + ".jpg")
            if DEBUG:
                print(save_path)
            cv2.imwrite(save_path, new_img)

    if DEBUG:
        video_path = osp.join(save_folder, id + ".avi")
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'),
                                fps, (INPUT, INPUT), isColor=True)
        for img in imgs:
            video.write(img)
        video.release()


def check_blur(imgs):
    blurs = []
    bad_thres = 3
    imgs = np.asarray(imgs)
    for img in imgs:
        w2 = img[:, :, 1]
        bw = ((w2 - np.min(w2)) / np.max(w2) * 255).astype(np.uint8)
        blur = cv2.Laplacian(bw, cv2.CV_64F).var()
        blurs.append(blur)
    min_blur = np.min(blurs)
    max_blur = np.max(blurs)
    new_blurs = blurs / min_blur
    num_bad = sum(new_blurs > bad_thres)
    if num_bad > 2:
        return []
    elif num_bad > 0:
        if DEBUG:
            print(num_bad, min_blur, max_blur, max_blur / min_blur)
        indces = np.where(new_blurs > bad_thres)[0]
        for index in indces:
            if index == 0:
                imgs[index] = imgs[index + 1]
            else:
                imgs[index] = imgs[index - 1]
    return list(imgs)


def generate_data(model, folder, save_folder):
    id = osp.split(folder)[1]
    if not id.isdigit():
        if '-' in id and id.split('-')[0].isdigit():
            real_id = id.split('-')[0]
        else:
            real_id = 'dmso'
    else:
        real_id = id

    # if osp.exists(osp.join(save_folder, real_id)) and real_id != 'dmso':
    #     print(f'{real_id} exist')
    #     return

    paths, imgs = [], []
    for i in range(16):
        img_path = f'{folder}/{i + 1}.jpg'
        paths.append(img_path)
        img = np.array(Image.open(img_path))
        min_img = np.min(np.min(img, 0), 0)
        img = img - min_img
        imgs.append(img)
    imgs = check_blur(imgs)
    if len(imgs) == 0:
        return

    # cellpose_imgs = []
    # for img in imgs:
    #     w3, w2 = img[:, :, 0].copy(), img[:, :, 1].copy()
    #     cellpose_img = img.copy()
    #     cellpose_img[:, :, 1] = np.maximum(w2, w3)
    #     cellpose_img[:, :, 0] = np.minimum(w2, w3)
    #     cellpose_imgs.append(cellpose_img)

    diameter = 300
    channels = [[2, 3]] * len(imgs)
    masks, flows, styles, diams = model.eval(imgs, diameter=diameter, flow_threshold=None, channels=channels,
                                             batch_size=144)
    all_cells, all_centers = extract_cell(paths, imgs, masks, f'{save_folder}/seg/{id}')

    save_folder = osp.join(save_folder, real_id)
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
    generate_sequence(all_centers, all_cells, id, imgs, save_folder)


def rm_initial_dead(folder):
    for path in glob.glob(f'{folder}/*/1.jpg'):
        img = Image.open(path)
        w3, w2, w1 = img.split()
        # outlines = np.array(img)

        cnts, areas = extract_nucleus(np.array(w1))
        w1_cnt = cnts[np.argmax(areas)]
        # vc, vr = w1_cnt.squeeze().T
        # outlines[vr, vc] = (255, 0, 0)
        w1_area = cv2.contourArea(w1_cnt)

        w2_cnt = np.concatenate(cv2.findContours(np.array(w2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2], axis=0)
        # vc, vr = w2_cnt.squeeze().T
        # outlines[vr, vc] = (0, 255, 0)
        # Image.fromarray(outlines).show()
        w2_area = cv2.contourArea(w2_cnt)
        ratio = w2_area / w1_area
        print(path, ratio)


def generate_data_batch(model, folders, save_folder):
    thread_list = []
    for i, folder in enumerate(folders):
        thread = Thread(target=generate_data,
                        args=(model, folder, save_folder))
        thread.start()
        thread_list.append(thread)
    for thread in thread_list:
        thread.join()


def random_split_dataset():
    folder = f'/home/weiming.li/dataset/mito_center/l*/fda/*'
    for path in glob.glob(folder):
        for pp in glob.glob(f'{path}/*'):
            if random.random() < 0.2:
                if random.random() < 0.5:
                    new_path = pp.replace('fda', 'query')
                else:
                    new_path = pp.replace('fda', 'gallery')
            else:
                new_path = pp.replace('fda', 'train')
            print(pp, new_path)
            os.renames(pp, new_path)


if __name__ == "__main__":
    from PIL import Image, ImageFilter
    img = Image.open('E:\dataset\mito\mito_raw\l1/fda/1/1.jpg')
    blurImage = img.filter(ImageFilter.GaussianBlur(11))
    blurImage.show()
    blurImage.save('E:\dataset\mito\mito_center\draw/1_1_blur.jpg')
    exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='/home/weiming.li/dataset/mito_raw/l1/fda', type=str)
    parser.add_argument('--save_folder', default='/home/weiming.li/dataset/mito_center/l1/fda', type=str)
    args = parser.parse_args()

    n_thread = 8
    folder_list = glob.glob(args.folder + "/*")
    folder_list.sort()

    model = models.Cellpose(gpu=True, model_type='cyto')

    for i in tqdm(range(0, len(folder_list), n_thread)):
        folders = folder_list[i:i + n_thread]
        generate_data_batch(model, folders, args.save_folder)
