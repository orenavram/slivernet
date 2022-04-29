import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU ID

from SliverNet2 import SliverNet2
from fastai.vision.all import *
from nonadaptiveconcatpool2d import load_backbone

from AmishDataset import AmishDataset

# nest = NesT(
#     image_size=19 * 256,
#     patch_size=16,
#     dim=96,
#     heads=1,
#     num_hierarchies=3,  # number of hierarchies
#     block_repeats=(2, 2, 8),  # the number of transformer blocks at each heirarchy, starting from the bottom
#     num_classes=1
# )


print(f'Using GPU #{os.environ["CUDA_VISIBLE_DEVICES"]}')
ds = AmishDataset('/scratch/avram/Amish/oct_metafile.csv',
                  '/scratch/avram/Amish/labels.csv',
                  data_format='npz',
                  pathology='cRORA')
                  # ['iRORA', 'cRORA']); print(f'NPZ data')

batch_size = 16
num_workers = 64

print(f'Number of samples is {len(ds)}')
print(f'Batch size is {batch_size}')
print(f'Num of cpus is {num_workers}')

adl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, drop_last=True)
dls = DataLoaders(adl, adl)
dls.c = 2

multi_gpu = False

# Load backbone and create model
backbone = load_backbone("kermany")
model = SliverNet2(backbone, n_out=dls.c)
model.to(device='cuda')
if (multi_gpu):
    model = torch.nn.DataParallel(model)
# try:
#     _ = learner.destroy()
#     torch.cuda.empty_cache()
# except:
#     print('No Learner to destroy')

# Create learner
print('Creating a new Learner...')
learner = Learner(dls, model, model_dir=f'SliverNet2_test', loss_func=torch.nn.CrossEntropyLoss())#, criterion=torch.nn.CrossEntropyLoss())
learner = learner.to_fp16()

learner.metrics = [accuracy]

# Fit
learner.fit_one_cycle(3, lr_max=1e-3)
learner.save('model')

# learner = Learner(DataLoaders(adl, adl), SliverNet2("kermany"), loss_func=fastai.optimizer.BCEWithLogitsLossFlat())
# # print('Searching for learning rate...')
# # learner.lr_find()
# # learner.recorder.plot(suggestion=True)
# print(80 * '#' + '\nFitting one cycle')
# learner.fit_one_cycle(10, 5e-2)