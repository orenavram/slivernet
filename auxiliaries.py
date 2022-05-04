import io
import json
import logging
import os
from zipfile import ZipFile, BadZipFile
import PIL
import numpy as np
from PIL import Image
from skimage import exposure
from torchvision import transforms as tf

logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))
totensor = tf.Compose([
    tf.ToTensor(),
])


def get_labels(sample, labels, pathologies):
    _, patient_id, exam_date, laterality, _ = sample.split('_')
    if laterality == 'NA':
        laterality = 'NULL'  # so the returned label will be empty
    label = labels[(labels.PAT_ID == patient_id) &
                   # (labels.EXAM_DATE == exam_date) &
                   (labels.Laterality == laterality)][pathologies].astype(np.float32).values
    if label.shape[0] > 1:
        # e.g., array([[ 1.,  1.],  # earlier exam
        #              [ 0.,  1.]]) # later exam
        # multiple scans were taken (and they are sorted by date). take the earliest scan (and slice weird to keep ndim the same!))
        label = label[:1]
    return label.squeeze()


def get_samples(metadata, labels, pathologies):
    samples = []
    label_to_count = {p: {} for p in pathologies}
    for sample in metadata.vol_name.values:

        if 'cube' not in sample:
            logger.debug(f'{sample} is not a cube')
            continue

        sample_labels = get_labels(sample, labels, pathologies)

        if sample_labels.size == 0:
            logger.debug(f'{sample} does not have labels')
            continue

        if np.isnan(sample_labels).any():
            logger.debug(f'{sample} labels contain NA: {sample_labels}')
            continue

        logger.debug(sample)
        samples.append(sample)
        for p, label in zip(pathologies, sample_labels):
            label_to_count[p][label] = label_to_count[p].get(label, 0) + 1
        logger.debug(label)

    logger.info(f'Label counts is: {label_to_count}')
    print(f'Label counts is: {label_to_count}')
    return samples


class pil_contrast_strech(object):

    def __init__(self, low=2, high=98):
        self.low, self.high = low, high

    def __call__(self, img):
        # Contrast stretching
        img = np.array(img)
        plow, phigh = np.percentile(img, (self.low, self.high))
        return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))


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

