import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"  # GPU ID

from torch.utils.data import Subset
from SliverNet2 import SliverNet2
from fastai.vision.all import *
from fastai.callback.wandb import *
from fastai.data.transforms import TrainTestSplitter
from nonadaptiveconcatpool2d import load_backbone
from AmishDataset import AmishDataset
import wandb
wandb.init()

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
dataset = AmishDataset('/scratch/avram/Amish/oct_metafile.csv',
                       '/scratch/avram/Amish/labels.csv',
                       # pathologies=['SO_PED_DPED'],
                       # pathologies=['iRORA'],
                       # pathologies=['cRORA'],
                       pathologies=['iRORA', 'cRORA', 'SO_PED_DPED'],
                       data_format='npz')

batch_size = 24
num_workers = 60

print(f'Num of cpus is {num_workers}')
print(f'Number of samples is {len(dataset)}')
print(f'Batch size is {batch_size}')

splitter = TrainTestSplitter(test_size=0.2, random_state=42)
train_indices, valid_indices = splitter(dataset)#, stratify=True)

train_dataset = Subset(dataset, train_indices)
valid_dataset = Subset(dataset, valid_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
print(f'# of train batches is {len(train_loader)}')

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
print(f'# of validation batches is {len(valid_loader)}')

dls = DataLoaders(train_loader, valid_loader)
dls.c = len(dataset.pathologies)


# Load backbone and create model
backbone = load_backbone("kermany")
model = SliverNet2(backbone, n_out=dls.c)
model.to(device='cuda')

multi_gpu = True
if multi_gpu:
    model = torch.nn.DataParallel(model)

# Create learner
print('Creating a new Learner...')
learner = Learner(dls, model, model_dir=f'SliverNet2_test',
                  cbs=WandbCallback(),
                  loss_func=torch.nn.BCEWithLogitsLoss())
learner = learner.to_fp16()

learner.metrics = [accuracy_multi, RocAucMulti(average=None), APScoreMulti(average=None), F1ScoreMulti(average=None)]#, APScoreMulti, F1ScoreMulti, RocAuc]

# Fit
learner.fit_one_cycle(5, lr_max=1e-3)
learner.save('model')

