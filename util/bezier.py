import numpy as np
import cv2
from scipy.special import comb


def fit_line(pts_list, order):
    """
    Fit lines
    :param pts_list:
    :param order:
    :return: lines: the shape is [N, O, 2]
            t_list:
    """
    lines, t_list = [], []
    t0 = np.linspace(0, 1, order + 1)
    for pts in pts_list:
        if len(pts) < 2:
            continue
        dists = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
        dists = np.cumsum(dists)
        t = np.concatenate((np.zeros(1), dists / dists[-1]))
        indices = [np.argmin(abs(t - i)) for i in t0]
        line = pts[indices]
        lines.append(line)
        t_list.append(t)

    lines = np.asarray(lines)
    return lines, t_list


def interp_line(lines, t_list=None, num=None, resolution=1.0):
    """
    Line interpolation
    :param lines: the shape is [N, O, 2]
    :param t_list:
    :param num:
    :param resolution:
    :return: pts_list
    """
    order = lines.shape[1] - 1
    p = comb(order, np.arange(order + 1))
    k = np.arange(0, order + 1)
    t = np.linspace(0, 1, order + 1)[:, None]
    coeff_matrix = p * (t ** k) * ((1 - t) ** (order - k))
    inv_coeff_matrix = np.linalg.inv(coeff_matrix)

    pts_list = []
    if t_list is not None:
        for line, t in zip(lines, t_list):
            control_points = np.matmul(inv_coeff_matrix, line)
            t = t[:, None]
            coeff_matrix = p * (t ** k) * ((1 - t) ** (order - k))
            pts = np.matmul(coeff_matrix, control_points)
            pts_list.append(pts)
    else:
        for line in lines:
            control_points = np.matmul(inv_coeff_matrix, line)
            K = int(round(max(abs(line[-1] - line[0])) / resolution)) + 1 if num is None else num
            t = np.linspace(0, 1, K)[:, None]
            coeff_matrix = p * (t ** k) * ((1 - t) ** (order - k))
            pts = np.matmul(coeff_matrix, control_points)
            pts_list.append(pts)

    return pts_list


def insert_line(image, lines, color, thickness=0):
    """
    Insert lines into image
    :param image: the shape is [N, O, 2]
    :param lines:
    :param color:
    :param thickness:
    :return: image
    """
    pts_list = interp_line(lines, resolution=1.0)
    for pts in pts_list:
        pts = np.round(pts).astype(np.int32)
        cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)

    return image


def insert_point(image, lines, color, thickness=1):
    """
    Insert points into image
    :param image: the shape is [N, O, 2]
    :param lines:
    :param color:
    :param thickness:
    :return: image
    """
    for pts in lines:
        pts = np.round(pts).astype(np.int32)
        for pt in pts:
            pt = tuple(pt)
            cv2.line(image, pt, pt, color=color, thickness=thickness)

    return image
