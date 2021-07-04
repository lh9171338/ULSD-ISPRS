import sys
sys.path.append('..')
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import util.camera as cam
import util.bezier as bez
from config.cfg import parse


def save_npz(prefix, image, lines, heatmap_size):
    image_size = (image.shape[1], image.shape[0])
    sx, sy = heatmap_size[0] / image_size[0], heatmap_size[1] / image_size[1]

    lines_mask = lines[:, 0, 1] > lines[:, -1, 1]
    lines[lines_mask] = lines[lines_mask, ::-1]
    lines[:, :, 0] = np.clip(lines[:, :, 0] * sx, 0, heatmap_size[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * sy, 0, heatmap_size[1] - 1e-4)
    juncs = np.concatenate((lines[:, 0], lines[:, -1]))
    juncs = np.round(juncs, 3)
    juncs = np.unique(juncs, axis=0)

    np.savez_compressed(
        f'{prefix}.npz',
        junc=juncs,
        line=lines
    )
    cv2.imwrite(f'{prefix}.png', image)


def json2npz(src_path, dst_path, cfg, plot=False):
    split = 'test'
    os.makedirs(dst_path, exist_ok=True)

    json_file = os.path.join(src_path, f'{split}.json')
    with open(json_file, 'r') as f:
        dataset = json.load(f)

    for data in tqdm(dataset, desc=split):
        filename = data['filename']
        type = data['type']
        lines = np.asarray(data['lines'])
        image = cv2.imread(os.path.join(src_path, 'image', filename))

        if type == 'pinhole':
            lines = np.asarray(bez.interp_line(lines, num=7))
        elif type == 'fisheye':
            coeff = {'K': np.asarray(data['K']), 'D': np.asarray(data['D'])}
            camera = cam.Fisheye(coeff)
            pts_list = camera.interp_line(lines, resolution=0.01)
            lines = bez.fit_line(pts_list, order=6)[0]
        elif type == 'spherical':
            image_size = (image.shape[1], image.shape[0])
            camera = cam.Spherical(image_size)
            lines = camera.truncate_line(lines)
            lines = camera.remove_line(lines, thresh=10.0)
            pts_list = camera.interp_line(lines, resolution=0.01)
            lines = bez.fit_line(pts_list, order=6)[0]
        else:
            exit()

        prefix = os.path.join(dst_path, filename.split('.')[0])
        save_npz(prefix, image, lines.copy(), cfg.heatmap_size)

        if plot:
            bez.insert_line(image, lines, color=[0, 0, 255])
            bez.insert_point(image, lines, color=[255, 0, 0], thickness=4)
            cv2.namedWindow('image', 0)
            cv2.imshow('image', image)
            cv2.waitKey()


if __name__ == "__main__":
    os.chdir('..')
    # Parameter
    cfg = parse()
    print(cfg)

    # Path
    src_path = cfg.raw_dataset_path
    dst_path = cfg.groundtruth_path
    os.makedirs(dst_path, exist_ok=True)

    json2npz(src_path, dst_path, cfg)
