from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from os import path
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import sys
from dival import get_standard_dataset
from dival.datasets.fbp_dataset import get_cached_fbp_dataset
from torch.utils.data import Dataset as TorchDataset
import torch
from os import path

from ista_unet import dataset_dir
from ista_unet import model_save_dir

def worker_init_fn(worker_id):
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)
    torch.backends.cudnn.deterministic = True

class RandomAccessTorchDataset(TorchDataset):
    def __init__(self, dataset, part, reshape=None):
        self.dataset = dataset
        self.part = part
        self.reshape = reshape or (
            (None,) * self.dataset.get_num_elements_per_sample())

    def __len__(self):
        return self.dataset.get_len(self.part)

    def __getitem__(self, idx):
        arrays = self.dataset.get_sample(idx, part=self.part)
        mult_elem = isinstance(arrays, tuple)
        if not mult_elem:
            arrays = (arrays,)
        tensors = []
        for arr, s in zip(arrays, self.reshape):
            t = torch.from_numpy(np.asarray(arr))
            if s is not None:
                t = t.view(*s)
            tensors.append(t)
        return tuple(tensors) if mult_elem else tensors[0]
    

def get_dataloaders_ct(batch_size=1, distributed_bool = False, num_workers = 0, IMPL = 'astra_cuda', cache_dir = path.join(dataset_dir, 'cache_lodopab'), include_validation = True, **kwargs):
    
    if include_validation:
        parts = ['train', 'validation', 'test']
        batch_sizes = {'train': batch_size,'validation': 1, 'test':1 }

    else:
        parts = ['train', 'test']
        batch_sizes = {'train': batch_size, 'test':1 }
                  
    CACHE_FILES = {part: (path.join(cache_dir, 'cache_lodopab_' + part + '_fbp.npy'), None) for part in parts }
    
    standard_dataset = get_standard_dataset('lodopab', impl=IMPL)
    ray_trafo = standard_dataset.get_ray_trafo(impl=IMPL)
    dataset = get_cached_fbp_dataset(standard_dataset, ray_trafo, CACHE_FILES)
        
   
    # create PyTorch datasets        
    datasets = {x: RandomAccessTorchDataset(dataset = dataset,
        part =  x, reshape=((1,) + dataset.space[0].shape,
                               (1,) + dataset.space[1].shape)) for x in parts}
    
    if distributed_bool == True:
        dataloaders = {x: DataLoader(datasets[x], batch_size=batch_sizes[x], num_workers = num_workers, worker_init_fn = worker_init_fn, pin_memory=True, sampler=DistributedSampler(datasets[x]) ) for x in parts}
    else:
        dataloaders = {x: DataLoader(datasets[x], batch_size=batch_sizes[x], shuffle=(x == 'train'), worker_init_fn = worker_init_fn, pin_memory=True, num_workers = num_workers ) for x in parts}

    return dataloaders
