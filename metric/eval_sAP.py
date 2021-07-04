import sys
sys.path.append('..')
import os
import glob
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from config.cfg import parse
from metric.eval_metric import calc_msAP, calc_sAP, plot_pr_curve
import util.bezier as bez


def eval_sAP(gt_path, pred_path, save_path=None, order=None):
    gt_file_list = sorted(glob.glob(os.path.join(gt_path, '*.npz')))
    pred_file_list = sorted(glob.glob(os.path.join(pred_path, '*.npz')))

    line_gts, line_preds, line_scores, im_ids = [], [], [], []
    for i, (gt_file, pred_file) in enumerate(zip(gt_file_list, pred_file_list)):
        with np.load(gt_file) as npz:
            line_gt = npz['line']
        line_gts.append(line_gt)

        with np.load(pred_file) as npz:
            result = {name: arr for name, arr in npz.items()}
            line_pred = result['line_pred']
            line_score = result['line_score']

        line_preds.append(line_pred)
        line_scores.append(line_score)
        im_ids.append(np.array([i] * line_pred.shape[0], dtype=np.int32))

    line_preds = np.concatenate(line_preds)
    line_scores = np.concatenate(line_scores)
    im_ids = np.concatenate(im_ids)
    indices = np.argsort(-line_scores)
    line_scores = line_scores[indices]
    line_preds = line_preds[indices]
    im_ids = im_ids[indices]

    n_pts = line_gts[0].shape[1]
    line_preds = np.asarray(bez.interp_line(line_preds, num=n_pts))

    msAP, P, R, sAP = calc_msAP(line_gts, line_preds, im_ids, [5.0, 10.0, 15.0])

    if save_path is not None:
        sAP10, _, _, rcs, prs = calc_sAP(line_gts, line_preds, im_ids, 10.0)
        figure = plot_pr_curve(rcs, prs, title='sAP${^{10}}$', legend=['ULSD'],)
        figure.savefig(os.path.join(save_path, f'sAP10_{cfg.version}.pdf'), format='pdf', bbox_inches='tight')
        sio.savemat(os.path.join(save_path, f'sAP10_{cfg.version}.mat'), {'rcs': rcs, 'prs': prs, 'AP': sAP10})
        plt.show()

    return msAP, P, R, sAP


if __name__ == "__main__":
    # Parameter
    os.chdir('..')
    cfg = parse()
    os.makedirs(cfg.figure_path, exist_ok=True)

    # Path
    gt_path = cfg.groundtruth_path
    pred_path = cfg.output_path
    save_path = cfg.figure_path

    start = time.time()
    msAP, P, R, sAP = eval_sAP(gt_path, pred_path, save_path, cfg.order)
    print(f'msAP: {msAP:.1f} | P: {P:.1f} | R: {R:.1f} | sAP5: {sAP[0]:.1f} | sAP10: {sAP[1]:.1f} | sAP15: {sAP[2]:.1f}')
    end = time.time()
    print('Time: %f s' % (end - start))
