# %%

import json
import os

import PIL
import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
from skimage import exposure
from torch.utils.data import Dataset

gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))
img2tensor = tf.ToTensor()


class pil_contrast_strech(object):

    def __init__(self, low=2, high=98):
        self.low, self.high = low, high

    def __call__(self, img):
        # Contrast stretching
        img = np.array(img)
        plow, phigh = np.percentile(img, (self.low, self.high))
        return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))


# applicable to data that load as grayscale images
default_transform_gray = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((256, 256)),
    # tf.CenterCrop(256),
    pil_contrast_strech(),
    tf.ToTensor(),
    gray2rgb
])

default_transform_rgb = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((256, 256)),
    # tf.CenterCrop(256),
    pil_contrast_strech(),
    tf.ToTensor()
])


def default_open_json(jsonname):
    with open(jsonname) as fl:
        blob = json.load(fl)
    return blob


def load_tiff(folder_name):
    img_paths = os.listdir(folder_name)
    vol = []
    for img_path in img_paths:
        img = Image.open(f'{folder_name}/{img_path}')
        vol.append(img2tensor(img)/256)
    return vol


def load_npz(fname):
    # NOTE: preferably use universal metadata formats
    #  so that we don't have to specify this
    vol = np.load(fname)['oct_volume']
    imgs = [img2tensor(mat / 256) for mat in vol]
    return imgs

class AmishTiffDataset(Dataset):
    def __init__(self, tiffs_folder, transform=default_transform_gray, data_format='tiff',
                 get_samples=lambda df: [fname for fname in df[df['num_slices'] >= 97]['filename'].values]):

        self.samples = [f'{tiffs_folder}/{folder}' for folder in os.listdir(tiffs_folder)]
        self.t = transform

        self.data_reader = dict(
            tiff=load_tiff,
            npz=load_npz
        )[data_format]

        # TODO: write a dynamic label reader
        # At the moment we're just predicting and don't need this
        self.label_reader = lambda sample: 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # print(f'Loading {sample}')

        # torch tensor (image) or list of tensors (volume)
        imgs = self.data_reader(sample)
        label = self.label_reader(sample)

        if self.t is not None:
            imgs = [self.t(im) for im in imgs]

        # t_imgs = torch.stack(imgs)

        # atm our models use volumes stacked vertically as imgs
        t_imgs = torch.cat([self.t(im) for im in imgs], dim=1)

        # print(t_imgs.shape)
        return t_imgs, label


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU ID
    print(f'Using GPU #{os.environ["CUDA_VISIBLE_DEVICES"]}')

    # tiff_example = load_tiff('/data1/Ophthalmology/OCT/Amish/tiffs/oct_5013-1_2014-01-13_R_cube')
    tiff_example = load_tiff('/data1/Ophthalmology/OCT/Amish/tiffs/oct_5000-103_2014-06-04_L_cube')
    # image with yellow area
    # tiff_example = load_tiff('/data1/Ophthalmology/OCT/Amish/tiffs/oct_5018-1_2014-04-22_R_cube')

    print(f'Example volume has {len(tiff_example)} slices (each loaded as a Tensor object)')
    first_slice = tiff_example[0]
    print(f'First slice has a shape of {first_slice.shape}')

    plt.imshow(first_slice[0])  # need to get rid of the first dim in order to show the slice
    plt.show()
    #
    # for i, eye_slice in enumerate(tiff_example):
    #     if 40 < i < 50:
    #         plt.imshow(eye_slice[0])  # need to get rid of the first dim in order to show the slice
    #         plt.show()
    #         print(i)

