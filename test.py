import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import time
import torch.utils.data as Data
from network.ulsd import ULSD
from network.dataset import Dataset
from config.cfg import parse
from metric.eval_mAPJ import eval_mAPJ
from metric.eval_sAP import eval_sAP
import util.bezier as bez


def save_lines(image, lines, filename, cfg, plot=False, fast=False):
    width, height = image.shape[1], image.shape[0]
    image_size = (width, height)
    heatmap_size = cfg.heatmap_size
    sx, sy = image_size[0] / heatmap_size[0], image_size[1] / heatmap_size[1]
    lines[:, :, 0] *= sx
    lines[:, :, 1] *= sy

    if fast:
        bez.insert_line(image, lines, color=[0, 255, 255], thickness=2)
        bez.insert_point(image, lines[:, [0, -1]], color=[255, 255, 0], thickness=6)
        cv2.imwrite(filename, image)
        if plot:
            cv2.namedWindow('image', 0)
            cv2.imshow('image', image)
            cv2.waitKey(1)
    else:
        fig = plt.figure()
        fig.set_size_inches(width / height, 1, forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.xlim([-0.5, width - 0.5])
        plt.ylim([height - 0.5, -0.5])
        plt.imshow(image[:, :, ::-1])
        pts_list = bez.interp_line(lines)
        for pts in pts_list:
            pts = pts - 0.5
            plt.plot(pts[:, 0], pts[:, 1], color="blue", linewidth=0.5)
            plt.scatter(pts[[0, -1], 0], pts[[0, -1], 1], color="#FF0000", s=1.5, edgecolors="none", zorder=5)

        plt.savefig(filename, dpi=height, bbox_inches=0)
        if plot:
            plt.show()
        plt.close()


def test(model, loader, cfg, device):
    # Test
    model.eval()

    index = 0
    start = time.time()
    for images in tqdm(loader, desc='test: '):
        images = images.to(device)
        jmaps, joffs, line_preds, line_scores = model(images)

        jmaps = jmaps.detach().cpu()
        joffs = joffs.detach().cpu()
        line_preds = [line_pred.detach().cpu() for line_pred in line_preds]
        line_scores = [line_score.detach().cpu() for line_score in line_scores]
        for i in range(len(images)):
            jmap = jmaps[i].numpy()
            joff = joffs[i].numpy()
            line_pred = line_preds[i].numpy()
            line_score = line_scores[i].numpy()
            src_filename = loader.dataset.file_list[index].split()[0]
            filename = os.path.split(src_filename)[1]
            image_filename = os.path.join(cfg.output_path, filename[:-4] + '.png')
            npz_filename = os.path.join(cfg.output_path, filename[:-4] + '.pnz')
            if cfg.evaluate:
                np.savez(npz_filename, jmap=jmap, joff=joff,
                                    line_pred=line_pred, line_score=line_score)
            if cfg.save_image:
                image = cv2.imread(src_filename)
                line_pred = line_pred[line_score > cfg.score_thresh]
                save_lines(image, line_pred, image_filename, cfg)

            index += 1

    end = time.time()

    if cfg.evaluate:
        fps = index / (end - start)
        print(f'FPS: {fps:.1f}')
        mAPJ, P, R = eval_mAPJ(cfg.groundtruth_path, cfg.output_path)
        print(f'mAPJ: {mAPJ:.1f} | P: {P:.1f} | R: {R:.1f}')
        msAP, P, R, sAP = eval_sAP(cfg.groundtruth_path, cfg.output_path)
        print(
            f'msAP: {msAP:.1f} | P: {P:.1f} | R: {R:.1f} | sAP5: {sAP[0]:.1f} | sAP10: {sAP[1]:.1f} | '
            f'sAP15: {sAP[2]:.1f}')


if __name__ == '__main__':
    # Parameter
    cfg = parse()
    os.makedirs(cfg.output_path, exist_ok=True)

    # Use GPU or CPU
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{cfg.gpu}' if use_gpu else 'cpu')
    print('use_gpu: ', use_gpu)

    # Load model
    model = ULSD(cfg).to(device)
    model_filename = os.path.join(cfg.model_path, cfg.model_name)
    checkpoint = torch.load(model_filename, map_location=device)
    if 'model' in checkpoint.keys():
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    # Load dataset
    dataset = Dataset(cfg.test_dataset_path, cfg, with_label=False)
    loader = Data.DataLoader(dataset=dataset, batch_size=cfg.test_batch_size,
                                 num_workers=cfg.num_workers, shuffle=False)

    # Test network
    test(model, loader, cfg, device)
