import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import *
from ista_unet import  dataset_dir
import os.path as path
from torch.utils.data.distributed import DistributedSampler
import torch

def worker_init_fn(worker_id):
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)
    torch.backends.cudnn.deterministic = True
    
    

def random_crop(hr, lr, size):
    h, w = lr.shape[:-1] 
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    crop_lr = lr[y:y+size, x:x+size].copy()
    crop_hr = hr[y:y+size, x:x+size].copy()
    return crop_hr, crop_lr

def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()

class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDataset, self).__init__()
        self.scale = scale
        self.size = size
        h5f = h5py.File(path, "r")
        
        self.hr = [v[:] for v in h5f["HR"].values()]
        self.lr = [v[:] for v in h5f["X{}".format(scale)].values()]
        
        self.len_hr = len(self.hr)
        
        h5f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()])

    def __len__(self):
        return len(self.hr)
    
    def __getitem__(self, index):
        size = self.size
        hr = self.hr[index]
        lr = self.lr[index]
        crop_hr, crop_lr = random_crop(hr, lr, size)
        flip_hr, flip_lr = random_flip_and_rotate(crop_hr, crop_lr)

        return self.transform(flip_lr ), self.transform(flip_hr )
        
        
class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(TestDataset, self).__init__()

        self.name  = dirname.split("/")[-1]
        self.scale = scale
        
        if "DIV" in self.name:
            self.hr = glob.glob(os.path.join("{}_HR".format(dirname), "*.png"))
            self.lr = glob.glob(os.path.join("{}_LR_bicubic".format(dirname), 
                                             "X{}/*.png".format(scale)))
        else:
            all_files = glob.glob(os.path.join(dirname, "x{}/*.png".format(scale)))
            self.hr = [name for name in all_files if "HR" in name]
            self.lr = [name for name in all_files if "LR" in name]

        self.hr.sort()
        self.lr.sort()
        
        
        self.hr_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr = Image.open(self.hr[index])
        lr = Image.open(self.lr[index])
        
        lr_transform = transforms.Compose([
            Resize((hr.size[1], hr.size[0]), interpolation=Image.BICUBIC),
            transforms.ToTensor()])
        
        hr = hr.convert("RGB")
        lr = lr.convert("RGB")
        
        return lr_transform(lr), self.hr_transform(hr)

    def __len__(self):
        return len(self.hr)
    
    
def get_dataloaders_super_resolution(scale_factor, batch_size, distributed_bool, num_workers, train_set = 'SR_291.h5', test_set = 'SR_B100', crop_size=128):
    
    train_data_path = path.join(dataset_dir, 'SR_datasets', train_set )
    test_data_dir = path.join(dataset_dir, 'SR_datasets', test_set)
    
    batch_sizes = {'train': batch_size, 'test': 1}
    image_datasets = {'train': TrainDataset(path = train_data_path,  scale = scale_factor, size =  crop_size), 
                      'test': TestDataset(test_data_dir, scale_factor) }
    
    if distributed_bool == True:
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_sizes[x], num_workers = num_workers, worker_init_fn = worker_init_fn, pin_memory=True, sampler=DistributedSampler(image_datasets[x]) ) for x in ['train', 'test']}
    else:
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_sizes[x], shuffle=(x == 'train'), worker_init_fn = worker_init_fn, pin_memory=True, num_workers = num_workers ) for x in ['train', 'test']}
    return dataloaders
