# %%

import io
import json
import os
from zipfile import ZipFile, BadZipFile

import PIL
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tf
from PIL import Image
from skimage import exposure
from torch.utils.data import Dataset

gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))

totensor = tf.Compose([
    tf.ToTensor(),
])


def get_label(sample, labels, pathology):
    filename = sample.split('/')[-1]

    _, patient_id, exam_date, laterality, _ = filename.split('_')
    if laterality == 'NA':
        laterality = 'NULL'  # so the returned label will be empty
    label = labels[(labels.PAT_ID == patient_id) &
                   (labels.EXAM_DATE == exam_date) &
                   (labels.Laterality == laterality)][pathology]
    return label.values

def get_samples(metadata, labels, pathology):
    samples = []
    for sample in metadata.vol_name.values:

        if 'cube' not in sample:
            continue

        label = get_label(sample, labels, pathology)
        # if label.empty:
        if label.size == 0:
            continue

        label = label[0]
        if np.isnan(label):
            continue

        # print(sample)
        samples.append(sample)
        # print(label)

    return samples


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


def default_open_imageZip_inmem(zipname):
    ims = []
    try:
        with ZipFile(zipname) as blob:
            imglist = blob.namelist()

            for imname in imglist:
                imgdata = blob.read(imname)
                img = Image.open(io.BytesIO(imgdata))
                ims += [totensor(img)]
    except BadZipFile:
        print()
        print('Invalid zipfile:', zipname)
        return None

    return ims


def load_npz(vol_name, npz_path='/scratch/avram/Amish/npz'):
    # NOTE: preferably use universal metadata formats
    #  so that we don't have to specify this
    vol = np.load(f'{npz_path}/{vol_name}.npz')['oct_volume']
    imgs = [totensor(mat / 256) for mat in vol]
    return imgs


def load_tiff(vol_name, tiff_path='/scratch/avram/Amish/tiffs'):
    img_paths = os.listdir(f'{tiff_path}/{vol_name}')
    vol = []
    for img_name in img_paths:
        img = Image.open(f'{tiff_path}/{vol_name}/{img_name}')
        vol.append(totensor(img)/256)
    return vol


class AmishDataset(Dataset):
    def __init__(self, metafile_path, labels_path, pathology, transform=default_transform_gray, data_format='npz'):

        # metadata fields are:
        # vol_name,patientid,patientid_e2e,laterality,num_slices,vol_number,dob
        self.metadata = pd.read_csv(metafile_path)
        self.labels = pd.read_csv(labels_path)
        self.pathology = pathology
        self.samples = get_samples(self.metadata, self.labels, pathology)
        self.t = transform

        print(f'{data_format.upper()} dataset loaded')
        self.data_reader = dict(
            zip=default_open_imageZip_inmem,
            npz=load_npz,
            tiff=load_tiff
        )[data_format]

        self.label_reader = get_label


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # print(f'Loading {sample}')

        # torch tensor (image) or list of tensors (volume)
        imgs = self.data_reader(sample)
        label = int(self.label_reader(sample, self.labels, self.pathology))

        # atm our models use volumes stacked vertically as imgs
        t_imgs = torch.cat([self.t(im) for im in imgs], dim=1)

        # print(t_imgs.shape)
        # return t_imgs, totensor(label)
        return t_imgs, label


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU ID
    print(f'Using GPU #{os.environ["CUDA_VISIBLE_DEVICES"]}')

    # npzExample = load_npz('/data1/Ophthalmology/OCT/Amish/npz/oct_5013-1_2014-01-13_R_cube.npz')
    npz_example = load_npz('/data1/Ophthalmology/OCT/Amish/npz/oct_5000-103_2014-06-04_L_cube.npz')
    # image with yellow area
    # npzExample = load_npz('/data1/Ophthalmology/OCT/Amish/npz/oct_5018-1_2014-04-22_R_cube.npz')

    print(f'Example volume has {len(npz_example)} slices (each loaded as a Tensor object)')
    first_slice = npz_example[0]
    print(f'First slice has a shape of {first_slice.shape}')

    plt.imshow(first_slice[0])  # need to get rid of the first dim in order to show the slice
    plt.show()

    # for i, eye_slice in enumerate(npz_example):
    #     if 40 < i < 50:
    #         plt.imshow(eye_slice[0])  # need to get rid of the first dim in order to show the slice
    #         plt.show()
    #         print(i)

