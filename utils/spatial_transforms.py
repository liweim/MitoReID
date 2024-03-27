import random
import math
import numbers
import collections
import numpy as np
import torch
import scipy.ndimage
from PIL import Image, ImageFilter

try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) * self.xshift))
        y1 = int(round((h - th) * self.yshift))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        self.xshift = random.uniform(0.25, 0.75)
        self.yshift = random.uniform(0.25, 0.75)


class CornerCrop(object):

    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.))
            y1 = int(round((image_height - th) / 2.))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class RandomFlip(object):
    """Horizontally flip the given PIL.Image randomly."""

    def __init__(self, random_rate=0.5):
        self.random_rate = random_rate

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < self.random_rate:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.p < self.random_rate:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def randomize_parameters(self):
        self.p = random.random()


class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self,
                 scales,
                 size,
                 interpolation=Image.BILINEAR,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

        self.crop_positions = crop_positions

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self.crop_positions[random.randint(
            0,
            len(self.scales) - 1)]


class MultiScaleRandomCrop(object):

    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        x1 = self.tl_x * (image_width - crop_size)
        y1 = self.tl_y * (image_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        # self.scale = 1
        self.tl_x = random.random()
        self.tl_y = random.random()


class KeepOrigin():
    def __call__(self, img):
        return img

    def randomize_parameters(self):
        return


class SpatialElasticDisplacement(object):

    def __init__(self, sigma=3.0, alpha=1.0, order=3, cval=0, mode="constant", random_rate=0.2):
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.cval = cval
        self.mode = mode
        self.random_rate = random_rate

    def __call__(self, img):
        if self.p < self.random_rate:
            image = np.asarray(img)
            image_first_channel = np.squeeze(image[..., 0])
            indices_x, indices_y = self._generate_indices(image_first_channel.shape, alpha=self.alpha, sigma=self.sigma)
            ret_image = (self._map_coordinates(
                image,
                indices_x,
                indices_y,
                order=self.order,
                cval=self.cval,
                mode=self.mode))

            return Image.fromarray(ret_image)
        else:
            return img

    def _generate_indices(self, shape, alpha, sigma):
        assert (len(shape) == 2), "shape: Should be of size 2!"
        dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    def _map_coordinates(self, image, indices_x, indices_y, order=1, cval=0, mode="constant"):
        assert (len(image.shape) == 3), "image.shape: Should be of size 3!"
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in range(image.shape[2]):
            remapped_flat = scipy.ndimage.interpolation.map_coordinates(
                image[..., c],
                (indices_x, indices_y),
                order=order,
                cval=cval,
                mode=mode
            )
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result

    def randomize_parameters(self):
        self.p = random.random()


class RandomRotate(object):

    def __init__(self, min=0, max=90, random_rate=0.5):
        self.interpolation = Image.BILINEAR
        self.min = min
        self.max = max
        self.random_rate = random_rate

    def __call__(self, img):
        min_val = int(np.min(np.array(img)))
        ret_img = img.rotate(self.rotate_angle, resample=self.interpolation, fillcolor=min_val)

        return ret_img

    def randomize_parameters(self):
        if random.random() < self.random_rate:
            self.rotate_angle = random.randint(self.min, self.max)
        else:
            self.rotate_angle = 0


class RandomResize(object):

    def __init__(self, min=0.8, max=1.2, random_rate=0.5):
        self.interpolation = Image.BILINEAR
        self.min = min
        self.max = max
        self.random_rate = random_rate

    def __call__(self, img):
        im_size = img.size
        ret_img = img.resize((int(im_size[0] * self.resize_const),
                              int(im_size[1] * self.resize_const)))

        return ret_img

    def randomize_parameters(self):
        if random.random() < self.random_rate:
            self.resize_const = random.uniform(self.min, self.max)
        else:
            self.resize_const = 1


class RandomScale(object):

    def __init__(self, min=0.9, max=1.1, random_rate=0.5):
        self.interpolation = Image.BILINEAR
        self.min = min
        self.max = max
        self.random_rate = random_rate

    def __call__(self, img):
        im_size = img.size
        ret_img = img.resize((int(im_size[0] * self.resize_const), im_size[1]))

        return ret_img

    def randomize_parameters(self):
        if random.random() < self.random_rate:
            self.resize_const = random.uniform(self.min, self.max)
        else:
            self.resize_const = 1


class GaussianBlur(object):
    def __init__(self, random_rate=0.01):
        self.random_rate = random_rate

    def __call__(self, img):
        if random.random() < self.random_rate:
            blurred = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
            return blurred
        else:
            return img

    def randomize_parameters(self):
        self.radius = random.uniform(1, 3)


class SaltImage(object):
    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, img):
        if self.p < 0.10:
            img = np.asarray(img)
            img = img.astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 255, img)
            return Image.fromarray(img.astype(np.uint8))
        else:
            return img

    def randomize_parameters(self):
        self.p = random.random()
        self.ratio = random.randint(80, 120)


class Dropout(object):

    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, img):
        if self.p < 0.10:
            img = np.asarray(img)
            img = img.astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 0, img)
            return Image.fromarray(img.astype(np.uint8))
        else:
            return img

    def randomize_parameters(self):
        self.p = random.random()
        self.ratio = random.randint(30, 50)


class MultiplyValues():

    def __init__(self, variance=10, random_rate=0.3):
        self.variance = variance
        self.random_rate = random_rate

    def __call__(self, img):
        if self.p < self.random_rate:
            image = np.asarray(img).astype(int)
            blank_idx = np.where(image == 0)
            image[:, :, 0] = image[:, :, 0] + self.add_r
            image[:, :, 1] = image[:, :, 1] + self.add_g
            image[:, :, 2] = image[:, :, 2] + self.add_b
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image[blank_idx] = 0
            image = image.astype(np.uint8)
            return Image.fromarray(image)
        else:
            return img

    def randomize_parameters(self):
        self.p = random.random()
        self.add_r = random.randint(-self.variance, self.variance)
        self.add_g = random.randint(-self.variance, self.variance)
        self.add_b = random.randint(-self.variance, self.variance)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, size=128, sl=0.1, sh=0.4, r1=0.8):
        self.size = size
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.w = self.size
        self.h = self.size

    def __call__(self, img):
        if self.p < 0.1:
            img = np.array(img)
            x1 = random.randint(0, self.size - self.h)
            y1 = random.randint(0, self.size - self.w)
            img[x1:x1 + self.h, y1:y1 + self.w] = 0
            return Image.fromarray(img)
        else:
            return img

    def randomize_parameters(self):
        self.p = random.random()
        for attempt in range(10):
            target_area = random.uniform(self.sl, self.sh) * self.size ** 2
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.size and h < self.size:
                self.w = w
                self.h = h
                return
