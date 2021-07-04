import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual


class Hourglass(nn.Module):
    def __init__(self, n, num_blocks, num_feats):
        super(Hourglass, self).__init__()
        self.n = n
        self.num_blocks = num_blocks
        self.num_feats = num_feats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.num_blocks):
            _up1_.append(Residual(self.num_feats, self.num_feats))
        self.low1 = nn.MaxPool2d(kernel_size=2, stride=2)
        for j in range(self.num_blocks):
            _low1_.append(Residual(self.num_feats, self.num_feats))

        if self.n > 1:
            self.low2 = Hourglass(n - 1, self.num_blocks, self.num_feats)
        else:
            for j in range(self.num_blocks):
                _low2_.append(Residual(self.num_feats, self.num_feats))
            self.low2_ = nn.ModuleList(_low2_)

        for j in range(self.num_blocks):
            _low3_.append(Residual(self.num_feats, self.num_feats))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)

        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = x
        for j in range(self.num_blocks):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.num_blocks):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.num_blocks):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.num_blocks):
            low3 = self.low3_[j](low3)
        up2 = self.up2(low3)

        return up1 + up2


class StackedHourglassNetwork(nn.Module):
    def __init__(self, depth=4, num_stacks=2, num_blocks=2, num_feats=256):
        super(StackedHourglassNetwork, self).__init__()
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_feats = num_feats
        self.depth = depth

        self.down = nn.Sequential(
            nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Residual(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Residual(128, 128),
            Residual(128, self.num_feats)
        )

        hourglasses, residuals, fcs, fcs_ = [], [], [], []
        for i in range(self.num_stacks):
            hourglasses.append(Hourglass(self.depth, self.num_blocks, self.num_feats))
            for _ in range(self.num_blocks):
                residuals.append(Residual(self.num_feats, self.num_feats))
            fcs.append(nn.Sequential(
                nn.Conv2d(self.num_feats, self.num_feats, bias=True, kernel_size=1, stride=1),
                nn.BatchNorm2d(self.num_feats),
                nn.ReLU(inplace=True)
            ))
            if i < self.num_stacks - 1:
                fcs_.append(nn.Conv2d(self.num_feats, self.num_feats, kernel_size=1))

        self.hourglasses = nn.ModuleList(hourglasses)
        self.residuals = nn.ModuleList(residuals)
        self.fcs = nn.ModuleList(fcs)
        self.fcs_ = nn.ModuleList(fcs_)

    def forward(self, x):
        x = self.down(x)

        for i in range(self.num_stacks):
            y = self.hourglasses[i](x)
            y = self.residuals[i](y)
            y = self.fcs[i](y)
            if i < self.num_stacks - 1:
                fc_ = self.fcs_[i](y)
                x = x + fc_
        feature = y
        return feature
