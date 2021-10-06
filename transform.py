import cv2
import numpy as np
import cupy as cp
from data import SequentialAug, RandomSizedCropAug, SaturationJitterAug, GridDistortion, NormalizeAug
from data import RotateAug, GaussianNoiseAug, TranslationAug, BrightnessJitterAug, ContrastJitterAug


class RandomTransform:
    def __init__(self, config):
        self.aug_list = SequentialAug()

        grid_distortion = config['grid_distortion']
        brightness = config['brightness']
        contrast = config['contrast']
        saturation = config['saturation']
        noise_sigma = config['noise_sigma']
        shear_ratio = config['shear_ratio']
        rotate_angle = config['rotate_angle']
        random_resize_crop = config['random_resize_crop']
        min_ratio = config['min_ratio']
        max_ratio = config['max_ratio']
        min_area = config['min_area']
        max_area = config['max_area']

        if brightness > 0:
            self.aug_list.add(BrightnessJitterAug(brightness=brightness))
        if contrast > 0:
            self.aug_list.add(ContrastJitterAug(contrast=contrast))
        if saturation > 0:
            self.aug_list.add(SaturationJitterAug(saturation=saturation))
        if shear_ratio > 0:
            self.aug_list.add(TranslationAug(shear_ratio=shear_ratio))
        if rotate_angle > 0:
            self.aug_list.add(RotateAug(angle=rotate_angle))
        if random_resize_crop:
            self.aug_list.add(RandomSizedCropAug(area=(min_area, max_area), ratio=(min_ratio, max_ratio)))
        if noise_sigma > 0:
            self.aug_list.add(GaussianNoiseAug(sigma=noise_sigma))
        if grid_distortion > 0:
            self.aug_list.add(GridDistortion(distort_limit=grid_distortion))

    def __call__(self, image):
        images = self.aug_list(image)
        return images


def pad_if_needed(image):
    rows, cols = image.shape[:2]
    min_side = max(rows, cols)
    if min_side > rows:
        h_pad_top = (min_side - rows) // 2
        h_pad_bottom = min_side - rows - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0
    if min_side > cols:
        w_pad_left = (min_side - cols) // 2
        w_pad_right = min_side - cols - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0
    image = cv2.copyMakeBorder(image, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, cv2.BORDER_CONSTANT, value=0)
    return image


class TrainTransform:
    def __init__(self, config):
        self.transform = RandomTransform(config)
        self.config = config
        self.target_shape = tuple(self.config['target_shape'])
        self.norm = NormalizeAug(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=1)

    def __call__(self, image):
        image = pad_if_needed(image)
        image = cp.asarray(image)
        image = image.astype(cp.float32) / 255.0
        image = self.transform(image)
        if np.random.uniform() < 0.5:
            image = image[:, ::-1]

        inter_method = [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]
        inter_method = inter_method[int(np.random.randint(0, 3))]
        image = cp.asnumpy(image)
        if image.shape[:2] != self.target_shape:
            image = cv2.resize(image, self.target_shape, interpolation=inter_method)
        image = cp.array(image)
        image = self.norm(image)
        image = image.astype(cp.float32, copy=False)
        return image


class TestTransform:
    def __init__(self, config):
        self.config = config
        self.target_shape = tuple(self.config['target_shape'])
        self.norm = NormalizeAug(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=1)

    def __call__(self, image, flip=False, inverse=False):
        image = pad_if_needed(image)
        image = cp.asarray(image)
        image = image.astype(cp.float32) / 255.0
        image = cp.asnumpy(image)
        if image.shape[:2] != self.target_shape:
            image = cv2.resize(image, self.target_shape, interpolation=cv2.INTER_LINEAR)
        image = cp.array(image)
        image = self.norm(image)
        image = image.astype(cp.float32, copy=False)
        return image
