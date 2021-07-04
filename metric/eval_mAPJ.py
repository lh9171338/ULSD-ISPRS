import sys
sys.path.append('..')
import os
import glob
import numpy as np
import time
import torch
from config.cfg import parse
from metric.eval_metric import calc_mAPJ


def non_maximum_suppression(heatmap):
    max_heatmap = torch.nn.functional.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    mask = (heatmap == max_heatmap)
    weight = torch.ones_like(mask) * 0.6
    weight[mask] = 1.0
    heatmap = weight * heatmap
    return heatmap


def calc_junction(jmap, joff, thresh=1e-2, top_K=1000):
    jmap = torch.from_numpy(jmap)
    joff = torch.from_numpy(joff)

    jmap = non_maximum_suppression(jmap)

    h, w = jmap.shape[-2], jmap.shape[-1]
    score = jmap.flatten()
    joff = joff.reshape(2, -1).t()

    num = min(int((score >= thresh).sum().item()), top_K)
    indices = torch.argsort(score, descending=True)[:num]
    score = score[indices]
    y, x = indices // w, indices % w
    junc = torch.cat((x[:, None], y[:, None]), dim=1) + joff[indices] + 0.5

    junc[:, 0] = junc[:, 0].clamp(min=0, max=w - 1e-4)
    junc[:, 1] = junc[:, 1].clamp(min=0, max=h - 1e-4)

    junc = junc.numpy()
    score = score.numpy()

    return junc, score


def eval_mAPJ(gt_path, pred_path):
    gt_file_list = sorted(glob.glob(os.path.join(gt_path, '*.npz')))
    pred_file_list = sorted(glob.glob(os.path.join(pred_path, '*.npz')))

    junc_gts, junc_preds, junc_scores, im_ids = [], [], [], []
    for i, (gt_file, pred_file) in enumerate(zip(gt_file_list, pred_file_list)):
        with np.load(gt_file) as npz:
            junc_gt = npz['junc']
        junc_gts.append(junc_gt)

        with np.load(pred_file) as npz:
            result = {name: arr for name, arr in npz.items()}
            jmap = result['jmap']
            joff = result['joff']
            junc_pred, junc_score = calc_junction(jmap, joff)

        junc_preds.append(junc_pred)
        junc_scores.append(junc_score)
        im_ids.append(np.array([i] * junc_pred.shape[0], dtype=np.int32))

    junc_preds = np.concatenate(junc_preds)
    junc_scores = np.concatenate(junc_scores)
    im_ids = np.concatenate(im_ids)
    indices = np.argsort(-junc_scores)
    junc_preds = junc_preds[indices]
    im_ids = im_ids[indices]

    mAPJ, P, R = calc_mAPJ(junc_gts, junc_preds, im_ids, [0.5, 1.0, 2.0])
    return mAPJ, P, R


if __name__ == '__main__':
    # Parameter
    os.chdir('..')
    cfg = parse()

    # Path
    gt_path = cfg.groundtruth_path
    pred_path = cfg.output_path

    start = time.time()
    mAPJ, P, R = eval_mAPJ(gt_path, pred_path)
    print(f'mAPJ: {mAPJ:.1f} | P: {P:.1f} | R: {R:.1f}')
    end = time.time()
    print('Time: %f s' % (end - start))
