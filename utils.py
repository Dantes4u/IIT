import numpy as np
import cupyx.scipy.ndimage
import cupy as cp


class SequentialAug:
    def __init__(self):
        super(SequentialAug, self).__init__()
        self.aug_list = []

    def add(self, aug):
        self.aug_list.append(aug)

    def __call__(self, image):
        np.random.shuffle(self.aug_list)
        for aug in self.aug_list:
            image = aug(image)
        return image


class NormalizeAug:
    def __init__(self, mean, std, scale):
        self.mean = mean
        self.std = std
        self.scale = scale

    def __call__(self, image):
        return (image / self.scale - cp.asarray(self.mean)) / cp.clip(cp.asarray(self.std), 1e-6)


class HorizontalFlipAug:
    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        if np.random.random() < self.p:
            image = cp.flip(image, axis=1)
        return image


class RandomSizedCropAug:
    def __init__(self, area, ratio):
        self.area = area
        self.ratio = ratio

    def __call__(self, image):
        h, w = image.shape[:2]
        image_area = h * w
        for _ in range(10):
            target_area = np.random.uniform(self.area[0], self.area[1]) * image_area
            new_ratio = np.random.uniform(*self.ratio)
            new_w = int(round(np.sqrt(target_area * new_ratio)))
            new_h = int(round(np.sqrt(target_area / new_ratio)))
            if np.random.random() < 0.5:
                new_h, new_w = new_w, new_h
            if new_w < w and new_h < h:
                x0 = np.random.randint(0, w - new_w)
                y0 = np.random.randint(0, h - new_h)
                image = image[y0: y0 + new_h, x0: x0 + new_w]
                matrix = cp.eye(4)
                matrix[0][0] = w / new_w
                matrix[1][1] = h / new_h
                output_shape = (h, w, 3)
                image = cupyx.scipy.ndimage.affine_transform(image, matrix, output_shape=output_shape,
                                                             output=image.dtype, mode='opencv', order=1)
                break
        return image


class RotateAug:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image):
        image = cupyx.scipy.ndimage.rotate(image, -np.random.uniform(-self.angle, self.angle),
                                           output=image.dtype, reshape=False, mode='constant', order=1)
        return image


class GridDistortion:
    def __init__(self, num_steps=5, distort_limit=0.3):
        self.num_steps = num_steps
        self.distort_limit = (-distort_limit, distort_limit)

    def __call__(self, image):
        xsteps = [1 + np.random.uniform(self.distort_limit[0], self.distort_limit[1])
                  for _ in range(self.num_steps + 1)]
        ysteps = [1 + np.random.uniform(self.distort_limit[0], self.distort_limit[1])
                  for _ in range(self.num_steps + 1)]
        height, width = image.shape[:2]

        x_step = width // self.num_steps
        xx = cp.zeros(width, cp.float32)
        prev = 0
        for idx in range(self.num_steps + 1):
            x = idx * x_step
            start = int(x)
            end = int(x) + x_step
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + x_step * xsteps[idx]

            xx[start:end] = cp.linspace(prev, cur, end - start)
            prev = cur

        y_step = height // self.num_steps
        yy = cp.zeros(height, np.float32)
        prev = 0
        for idx in range(self.num_steps + 1):
            y = idx * y_step
            start = int(y)
            end = int(y) + y_step
            if end > height:
                end = height
                cur = height
            else:
                cur = prev + y_step * ysteps[idx]

            yy[start:end] = cp.linspace(prev, cur, end - start)
            prev = cur

        map_x, map_y = cp.meshgrid(xx, yy)
        
        map_x = cp.asarray(map_x.flatten())
        map_y = cp.asarray(map_y.flatten())
        cmap = cp.stack((map_y, map_x), 0)

        image_r = cupyx.scipy.ndimage.map_coordinates(image[:, :, 0], cmap, mode='constant', order=1).reshape(image.shape[:-1])
        image_g = cupyx.scipy.ndimage.map_coordinates(image[:, :, 1], cmap, mode='constant', order=1).reshape(image.shape[:-1])
        image_b = cupyx.scipy.ndimage.map_coordinates(image[:, :, 2], cmap, mode='constant', order=1).reshape(image.shape[:-1])
        image = cp.stack((image_r, image_g, image_b), 2)

        return image


class TranslationAug:
    def __init__(self, shear_ratio):
        self.shear_ratio = shear_ratio

    def __call__(self, image):
        matrix = cp.eye(4)
        matrix[0][3] = np.random.uniform(-self.shear_ratio, self.shear_ratio) * image.shape[0]
        matrix[1][3] = np.random.uniform(-self.shear_ratio, self.shear_ratio) * image.shape[1]
        image = cupyx.scipy.ndimage.affine_transform(image, matrix, output_shape=image.shape, output=image.dtype,
                                                     mode='opencv', order=1)
        return image


class GaussianNoiseAug:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, image):
        sigma = cp.random.uniform(0, self.sigma)
        gaussian_noise = cp.random.normal(loc=1, scale=sigma, size=image.shape)
        image = image * gaussian_noise
        return image


class BrightnessJitterAug:
    def __init__(self, brightness):
        self.brightness = brightness

    def __call__(self, image):
        alpha = 1.0 + cp.random.uniform(-self.brightness, self.brightness)
        return image * alpha


class ContrastJitterAug:
    def __init__(self, contrast):
        self.contrast = contrast

    def __call__(self, image):
        coef = cp.array([[[0.299, 0.587, 0.114]]])
        alpha = 1.0 + cp.random.uniform(-self.contrast, self.contrast)
        gray = image * coef
        gray = (3.0 * (1.0 - alpha) / image.size) * cp.sum(gray)
        image *= alpha
        image += gray
        return image


class SaturationJitterAug:
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, image):
        coef = cp.array([[[0.299, 0.587, 0.114]]])
        alpha = 1.0 + cp.random.uniform(-self.saturation, self.saturation)
        gray = image * coef
        gray = cp.sum(gray, axis=2, keepdims=True)
        gray *= (1.0 - alpha)
        image *= alpha
        image += gray
        return image


