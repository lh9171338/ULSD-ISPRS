import torch
import torch.nn as nn
import numpy as np
from scipy.special import comb


class LoI(nn.Module):
    def __init__(self, order, n_pts):
        super(LoI, self).__init__()

        self.n_pts = n_pts

        p = comb(order, np.arange(order + 1))
        k = np.arange(0, order + 1)
        t0 = np.linspace(0, 1, order + 1)[:, None]
        coeff0 = p * (t0 ** k) * ((1 - t0) ** (order - k))
        t = np.linspace(0, 1, self.n_pts)[:, None]
        coeff = p * (t ** k) * ((1 - t) ** (order - k))

        lambda_ = np.matmul(coeff, np.linalg.inv(coeff0))
        lambda_ = torch.from_numpy(lambda_).float()
        self.register_buffer('lambda_', lambda_)

    def forward(self, feature, loi_pred):
        """
        Forward LoI Pooling Module

        Here are notations.

        * math:`C`: channel size of the input feature.
        * math:`H`: height of the input feature.
        * math:`W`: witdh of the input feature.
        * math:`R`: number of the proposed lines.
        * math:`O`: order of the Bezier curve.
        * math:`P`: number of the sample points.

        :param feature: The feature extracted from images.
                Its shape is :math:`(C, H, W)`.
        :param loi_pred: The lines array containing coordinates of proposal lines.
                Its shape is :math:`(R, O + 1, 2)`.
        :return:
            line_feature: The feature of lines.
            Its shape is :math:`(R, C, P)`.

        """
        c, h, w = feature.shape

        pts = (self.lambda_[None, :, :, None] * loi_pred[:, None]).sum(2) - 0.5
        pts = pts.reshape(-1, 2)
        px, py = pts[:, 0].contiguous(), pts[:, 1].contiguous()
        px0 = px.floor().clamp(min=0, max=w - 1)
        py0 = py.floor().clamp(min=0, max=h - 1)
        px1 = (px0 + 1).clamp(min=0, max=w - 1)
        py1 = (py0 + 1).clamp(min=0, max=h - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        loi_feature = (feature[:, py0l, px0l] * (py1 - py) * (px1 - px) + \
                       feature[:, py1l, px0l] * (py - py0) * (px1 - px) + \
                       feature[:, py0l, px1l] * (py1 - py) * (px - px0) + \
                       feature[:, py1l, px1l] * (py - py0) * (px - px0)).reshape(c, -1, self.n_pts).permute(1, 0, 2)

        return loi_feature


class LoIHead(nn.Module):
    def __init__(self, order, num_feats, n_pts):
        super(LoIHead, self).__init__()

        self.num_feats = num_feats
        self.n_pts = n_pts

        self.loi = LoI(order=order, n_pts=n_pts)
        self.pooling = nn.MaxPool1d(4, stride=4)
        self.fc1 = nn.Sequential(
            nn.Conv2d(self.num_feats, self.num_feats // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_feats // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_feats // 2, self.num_feats // 2, kernel_size=3, padding=1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear((self.num_feats // 2) * (self.n_pts // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, features, loi_preds):
        """
        Forward LoI Head Module

        Here are notations.

        * math:`N`: batch size of the input feature.
        * math:`C`: channel size of the input feature.
        * math:`H`: height of the input feature.
        * math:`W`: witdh of the input feature.
        * math:`R`: number of the proposed lines.
        * math:`O`: order of the Bezier curve.
        * math:`P`: number of the sample points.

        :param features: The features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
        :param loi_preds: The list of proposed lines array.
                Its shape is :math:`(N, (R, O + 1, 2))`.
        :return:
            loi_scores: The list of proposed line scores array.
                Its shape is :math:`(N, (R))`.
        """
        features_no_grad = features.detach()
        features = self.fc1(features_no_grad)
        b = features.shape[0]
        loi_scores = []
        for i in range(b):
            feature, loi_pred = features[i], loi_preds[i]
            loi_feature = self.loi(feature, loi_pred)
            loi_feature = self.pooling(loi_feature)
            loi_feature = loi_feature.reshape(loi_feature.shape[0], -1)
            loi_score = self.fc2(loi_feature).squeeze(-1)
            loi_scores.append(loi_score)

        return loi_scores

