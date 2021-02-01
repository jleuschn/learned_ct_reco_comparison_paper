import os
import pathlib
import pytorch_lightning as pl
# import fastmri
# from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule, UnetModule
from pathlib import Path
from ista_unet import dataset_dir

def get_dataloaders_fastmri(mask_type = 'random',
                            center_fractions  = [0.08],
                            accelerations = [4],
                            challenge = 'singlecoil',
                            batch_size = 8,
                            num_workers = 4,
                            distributed_bool = False,
                            dataset_dir = dataset_dir,
                            mri_dir = 'fastmri/knee/',
                            worker_init_fn = None,
                            include_test = False,
                            **kwargs):
    data_path = Path(os.path.join(dataset_dir, mri_dir))
    
    mask = create_mask_for_mask_type(mask_type_str = mask_type, 
                                     center_fractions = center_fractions, 
                                     accelerations = accelerations )
    

    # use random masks for train transform, fixed masks for val transform
    train_transform = UnetDataTransform(challenge, mask_func=mask, use_seed=False)
    val_transform = UnetDataTransform(challenge, mask_func=mask)
    test_transform = UnetDataTransform(challenge)

    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path= data_path,
        challenge= challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed_sampler = distributed_bool
    )



    if include_test:        
        dataloaders = {'train': data_module.train_dataloader() ,
                       'validation': data_module.val_dataloader(), 
                       'test': data_module.test_dataloader()}
    else:
        dataloaders = {'train': data_module.train_dataloader() ,
                       'validation': data_module.val_dataloader()}        

    return dataloaders