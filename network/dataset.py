import os
import glob
import numpy as np
import random
import torch
from PIL import Image
import torch.utils.data as Data
import torchvision.transforms.transforms as tfs
from torch.utils.data.dataloader import default_collate


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        mean = torch.as_tensor(self.mean, dtype=image.dtype, device=image.device)
        std = torch.as_tensor(self.std, dtype=image.dtype, device=image.device)
        image = image.mul(std[:, None, None]).add(mean[:, None, None])
        return image


class Dataset(Data.Dataset):
    def __init__(self, path, cfg, with_label=True, augment=False):
        if with_label:
            image_file_list = glob.glob(os.path.join(path, '*.png')) + \
                              glob.glob(os.path.join(path, '*.jpg'))
            label_file_list = glob.glob(os.path.join(path, '*.npz'))
            image_file_list.sort()
            label_file_list.sort()
            self.file_list = [image_file + ' ' + label_file
                              for image_file, label_file in zip(image_file_list, label_file_list)]
        else:
            image_file_list = glob.glob(os.path.join(path, '*.png')) + \
                              glob.glob(os.path.join(path, '*.jpg'))
            image_file_list.sort()
            self.file_list = [image_file for image_file in image_file_list]

        self.with_label = with_label
        self.augment = augment
        self.image_size = cfg.image_size
        self.n_stc_posl = cfg.n_stc_posl
        self.n_stc_negl = cfg.n_stc_negl
        self.mean = cfg.mean
        self.std = cfg.std
        self.DeNormalize = DeNormalize(self.mean, self.std)

    def __getitem__(self, index):
        if self.with_label:
            image_file, label_file = self.file_list[index].split()
            image = Image.open(image_file).convert('RGB')
            image = image.resize(self.image_size, Image.BILINEAR)
            with np.load(label_file) as npz:
                label = {name: npz[name] for name in ['jmap', 'joff', 'cmap', 'coff', 'eoff', 'lmap',
                                                      'junc', 'lpos', 'lneg']}
            image, map, meta = self.transforms(image, label)
            return image, map, meta
        else:
            image_file = self.file_list[index].split()[0]
            image = Image.open(image_file).convert('RGB')
            image = image.resize(self.image_size, Image.BILINEAR)
            image = self.transforms(image)
            return image

    def __len__(self):
        return len(self.file_list)

    def transforms(self, image, label=None):
        if self.with_label:
            image = tfs.ToTensor()(image)
            image = tfs.Normalize(self.mean, self.std)(image)
            jmap = torch.from_numpy(label['jmap']).float()
            joff = torch.from_numpy(label['joff']).float()
            cmap = torch.from_numpy(label['cmap']).float()
            coff = torch.from_numpy(label['coff']).float()
            eoff = torch.from_numpy(label['eoff']).float()
            lmap = torch.from_numpy(label['lmap']).float()
            line = torch.from_numpy(label['lpos']).float()
            lpos = np.random.permutation(label['lpos'])[: self.n_stc_posl]
            lneg = np.random.permutation(label['lneg'])[: self.n_stc_negl]
            if len(lneg) == 0:
                lneg = np.zeros((0, lpos.shape[1], 2))
            npos, nneg = len(lpos), len(lneg)
            lpre = np.concatenate((lpos, lneg))
            for i in range(len(lpre)):
                if random.random() > 0.5:
                    lpre[i] = lpre[i, ::-1]
            lpre = torch.from_numpy(lpre).float()
            lpre_label = torch.cat([torch.ones(npos), torch.zeros(nneg)]).float()

            map = {'jmap': jmap, 'joff': joff, 'cmap': cmap, 'coff': coff, 'eoff': eoff, 'lmap': lmap}
            meta = {'line': line, 'lpre': lpre, 'lpre_label': lpre_label}
            return image, map, meta
        else:
            image = tfs.ToTensor()(image)
            image = tfs.Normalize(self.mean, self.std)(image)
            return image

    @staticmethod
    def collate(batch):
        return (
            default_collate([b[0] for b in batch]),
            {name: default_collate([b[1][name] for b in batch]) for name in batch[0][1].keys()},
            {name: [b[2][name] for b in batch] for name in batch[0][2].keys()}
        )
