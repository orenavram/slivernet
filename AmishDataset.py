import torch
from torch.utils.data import Dataset
from auxiliaries import *


class AmishDataset(Dataset):
    def __init__(self, data, pathologies, transform=default_transform_gray, data_format='npz'):
        self.samples = data.index.tolist()
        self.labels = data[pathologies]
        self.t = transform
        self.data_reader = dict(
            zip=default_open_imageZip_inmem,
            npz=load_npz,
            tiff=load_tiff
        )[data_format]

        logger.info(f'{data_format.upper()} dataset loaded')
        logger.info(f'Predicting {pathologies}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        logger.debug(f'Loading {sample}')

        imgs = self.data_reader(sample)
        labels = self.labels.loc[sample].values

        # tile volume slices into one very long image
        t_imgs = torch.cat([self.t(im) for im in imgs], dim=1)

        logger.debug(t_imgs.shape)
        return t_imgs, labels


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

# def split_dataset2(self, pct=0.2, random_seed=None):
#     dataset_size = len(self.samples)
#     split = round(pct * dataset_size)
#     np.random.seed(random_seed)
#     indices = np.random.permutation(dataset_size)
#
#     train_indices, val_indices = indices[:split], indices[split:]
#     train_df = self.df.iloc[train_indices].reset_index()
#     val_df = self.df.iloc[val_indices].reset_index()
#
#     attr = self.get_attributes()
#     train_attr = attr.copy()
#     train_attr['df'] = train_df
#     train_attr['mode'] = 'train'
#     test_attr = attr.copy()
#     test_attr['df'] = val_df
#     test_attr['mode'] = 'test'
#
#     ds_tr = AmishDataset(**train_attr)
#     ds_tst = AmishDataset(**test_attr)
#
#     return ds_tr, ds_tst
