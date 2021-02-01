from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from os import path
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import sys
from torch.utils.data import Dataset as TorchDataset
import torch
from os import path
from scipy.io import loadmat


from ista_unet import dataset_dir
from ista_unet import model_save_dir

def worker_init_fn(worker_id):
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)
    torch.backends.cudnn.deterministic = True
    
# Creat the noisy ellipsoid dataset
class ellipses_dataset(Dataset):
    #Download the ellipsoid dataset https://github.com/panakino/FBPConvNet
    def __init__(self, mode, transform, dataset_dir , train_size = 475, **kwargs):            
       
        super(ellipses_dataset, self).__init__()
        self.dataset_str = 'Ellipsoid'
        self.transform = transform
        self.mode = mode
        
        unnorm_data = loadmat(path.join(dataset_dir, 'preproc_x20_ellipse_fullfbp.mat') )

        noisy_data = unnorm_data['lab_d']
        clean_data = unnorm_data['lab_n']

        self.train_size = train_size
        self.test_size = 500 - train_size
        
        if self.mode == 'train':
            self.mode_noisy_data = noisy_data[:, :, :, :self.train_size]
            self.mode_clean_data = clean_data[:, :, :, :self.train_size]

        else:
            self.mode_noisy_data = noisy_data[:, :, :, self.test_size: ]
            self.mode_clean_data = clean_data[:, :, :, self.test_size: ]

    def __len__(self):
        if self.mode == 'train':
            num = self.train_size
        else:
            num = self.test_size
        return num

    def __repr__(self):
        return "ellipses_dataset(mode={})". \
            format(self.mode)

    def __getitem__(self, idx):
        
        clean = self.transform(self.mode_clean_data[:, :, 0, idx])
        noisy = self.transform(self.mode_noisy_data[:, :, 0, idx])

        if self.mode == 'train':

            if np.random.randint(2) == 0:
                clean = torch.flip(clean, dims = [1])
                noisy = torch.flip(noisy, dims = [1])

            if np.random.randint(2) == 0:
                clean = torch.flip(clean, dims = [2])
                noisy = torch.flip(noisy, dims = [2])
                
        noisy = noisy.type(torch.FloatTensor)
        clean = clean.type(torch.FloatTensor)
        return noisy, clean


def get_dataloaders_ellipses(batch_size=1, dataset_dir = path.join(dataset_dir, 'Ellipse'), distributed_bool = False, num_workers = 0, **kwargs):
    
    batch_sizes = {'train': batch_size, 'test':1}

    train_transforms = transforms.Compose([
        transforms.ToTensor()])

    test_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    data_transforms = {'train': train_transforms,
                       'test': test_transforms}
    image_datasets = {'train': ellipses_dataset(transform = data_transforms['train'], dataset_dir = dataset_dir, mode = 'train'),
                      'test': ellipses_dataset(transform =  data_transforms['test'], dataset_dir = dataset_dir, mode = 'test')}
    
    if distributed_bool == True:
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_sizes[x], worker_init_fn = worker_init_fn, pin_memory=True, num_workers=num_workers, sampler=DistributedSampler(image_datasets[x]) ) for x in ['train', 'test']}
    else:
        dataloaders = {x: DataLoader(image_datasets[x],  batch_size=batch_sizes[x], shuffle=(x == 'train'), worker_init_fn = worker_init_fn, pin_memory=True, num_workers=num_workers ) for x in ['train', 'test']}
    return dataloaders