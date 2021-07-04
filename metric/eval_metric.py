import numpy as np
import matplotlib.pyplot as plt


def plot_pr_curve(rcs, prs, title, legend):
    plt.figure()
    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.yticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve for ' + title)
    plt.plot(rcs, prs, 'r-')
    plt.rc('legend', fontsize=10)
    plt.legend(legend)

    f_scores = np.linspace(0.2, 0.9, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color=[0, 0.5, 0], alpha=0.3)
        plt.annotate('f={0:0.1}'.format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4, fontsize=10)

    return plt.gcf()


def calc_AP(rcs, prs):
    recall = np.concatenate(([0.0], rcs, [1.0]))
    precision = np.concatenate(([0.0], prs, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    AP = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

    return AP


def calc_APJ(junc_gts, junc_preds, im_ids, distance):
    n_gt = sum(junc_gt.shape[0] for junc_gt in junc_gts)
    n_pred = junc_preds.shape[0]
    tp, fp = np.zeros(n_pred, dtype=np.float), np.zeros(n_pred, dtype=np.float)
    hits = [[False for _ in junc_gt] for junc_gt in junc_gts]

    for i in range(n_pred):
        im_id = im_ids[i]
        junc_gt = junc_gts[im_id]
        junc_pred = junc_preds[i]
        dists = np.linalg.norm((junc_pred[None, :] - junc_gt), axis=1)
        choice = np.argmin(dists)
        dist = np.min(dists)
        if dist < distance and not hits[im_id][choice]:
            tp[i] = 1
            hits[im_id][choice] = True
        else:
            fp[i] = 1

    tp = np.cumsum(tp) / n_gt
    fp = np.cumsum(fp) / n_gt
    rcs = tp
    prs = tp / np.maximum(tp + fp, 1e-9)
    AP = calc_AP(rcs, prs)
    P, R = prs[-1], rcs[-1]

    return AP * 100.0, P * 100.0, R * 100.0


def calc_mAPJ(junc_gts, junc_preds, im_ids, distances):
    results = [calc_APJ(junc_gts, junc_preds, im_ids, distance) for distance in distances]
    AP = np.mean([result[0] for result in results])
    P = np.mean([result[1] for result in results])
    R = np.mean([result[2] for result in results])

    return AP, P, R


def calc_sAP(line_gts, line_preds, im_ids, distance):
    n_gt = sum(line_gt.shape[0] for line_gt in line_gts)
    n_pred, n_pts = line_preds.shape[0], line_preds.shape[1]
    tp, fp = np.zeros(n_pred, dtype=np.float), np.zeros(n_pred, dtype=np.float)
    hits = [[False for _ in line_gt] for line_gt in line_gts]

    for i in range(n_pred):
        im_id = im_ids[i]
        line_gt = line_gts[im_id]
        line_pred = line_preds[i]
        dists1 = ((line_pred[None, :] - line_gt) ** 2).sum(-1).sum(-1)
        dists2 = ((line_pred[None, ::-1] - line_gt) ** 2).sum(-1).sum(-1)
        dists = np.minimum(dists1, dists2) * 2.0 / n_pts
        choice = np.argmin(dists)
        dist = np.min(dists)
        if dist < distance and not hits[im_id][choice]:
            tp[i] = 1
            hits[im_id][choice] = True
        else:
            fp[i] = 1

    tp = np.cumsum(tp) / n_gt
    fp = np.cumsum(fp) / n_gt
    rcs = tp
    prs = tp / np.maximum(tp + fp, 1e-9)
    AP = calc_AP(rcs, prs)
    P, R = prs[-1], rcs[-1]

    return AP * 100.0, P * 100.0, R * 100.0, rcs, prs


def calc_msAP(line_gts, line_preds, im_ids, distances):
    results = [calc_sAP(line_gts, line_preds, im_ids, distance)[:3] for distance in distances]
    AP = np.mean([result[0] for result in results])
    P = np.mean([result[1] for result in results])
    R = np.mean([result[2] for result in results])

    return AP, P, R, [result[0] for result in results]
