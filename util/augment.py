import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, lines):
        for transform in self.transforms:
            if transform is not None:
                image, lines = transform(image, lines)
        return image, lines


class Noop:
    def __init__(self):
        pass

    def __call__(self, image, lines):
        return image.copy(), lines.copy()


class HorizontalFlip:
    def __init__(self):
        pass

    def __call__(self, image, lines):
        lines = lines.copy()
        width, height = image.shape[1], image.shape[0]
        image = image[:, ::-1]
        lines[:, :, 0] = width - lines[:, :, 0]
        lines[:, :, 0] = np.clip(lines[:, :, 0], 0.0, width - 1e-4)
        return image.copy(), lines


class VerticalFlip:
    def __init__(self):
        pass

    def __call__(self, image, lines):
        lines =  lines.copy()
        width, height = image.shape[1], image.shape[0]
        image = image[::-1, :]
        lines[:, :, 1] = height - lines[:, :, 1]
        lines[:, :, 1] = np.clip(lines[:, :, 1], 0.0, height - 1e-4)
        return image.copy(), lines


class HorizontalMove:
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, image, lines):
        lines = lines.copy()
        width, height = image.shape[1], image.shape[0]
        deltax = int(round(width * self.degree / 360.0)) % width
        image = np.concatenate((image[:, deltax:], image[:, :deltax]), axis=1)
        lines[:, :, 0] = (lines[:, :, 0] - deltax) % width
        return image.copy(), lines
