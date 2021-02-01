import numpy as np
import torch
import torch.utils.data as td
import torch.nn as nn
from math import log10
import sys, os
import pickle5 as pickle
from .models import ista_unet
from ista_unet import model_save_dir

import math


def YCbCr_mse_loss(out, target):
    
    diff = out - target
    diff[:, 0, :, :] = (65.738/256) * diff[:, 0, :, :]
    diff[:, 1, :, :] = (129.057/256) * diff[:, 1, :, :]
    diff[:, 2, :, :] = (25.064/256) * diff[:, 2, :, :]
    
    mse = torch.mean(diff**2)
    
    return mse


def matlab_psnr(sr_tensor_3d, hr_tensor_3d, scale, round_digit = 2):
    diff = sr_tensor_3d - hr_tensor_3d
    shave = scale

    diff[0,:,:] = diff[0,:,:] * 65.738/256
    diff[1,:,:] = diff[1,:,:] * 129.057/256
    diff[2,:,:] = diff[2,:,:] * 25.064/256

    sum_diff = torch.sum(diff, axis=0)

    valid = sum_diff[shave:-shave, shave:-shave]
    mse = torch.mean(valid**2)

    psnr = -10 * math.log10(mse)
    return round(psnr, round_digit)


def calculate_psnr(model, phase, loaders, device):

    cumu_psnr = 0
    total_samples = 0
    loader = loaders[phase]
    loader.worker_init_fn(worker_id = 0)

    # Iterate over data.
    
    if model == None:

        for (x, d) in loader:
            loss_per_sample = torch.sum((x - d).pow(2), axis = (1,2,3) ).cpu().numpy()
            spatial_mean_loss_per_sample = loss_per_sample / (d.shape[1] * d.shape[2] * d.shape[3] ) # channel_num * width * height
            cumu_psnr += -10* np.sum(np.log10(spatial_mean_loss_per_sample )) 
            total_samples += len(spatial_mean_loss_per_sample)

        psnr = cumu_psnr / total_samples
    
    else:    
        model.eval()   # Set model to evaluate mode
        model.to(device)
        total_batch = len(loader)
        iter_idx = 0
        for (x, d) in loader:
            iter_idx += 1
            x, d = x.to(device), d.to(device)
            with torch.no_grad():
                output = model(x)
                output = torch.clamp(output, 0, 1) # clamp the output between 0 and 1.

            loss_per_sample = torch.sum((output - d).pow(2), axis = (1,2,3) ).cpu().numpy()
            spatial_mean_loss_per_sample = loss_per_sample / (d.shape[1] * d.shape[2] * d.shape[3] ) # channel_num * width * height
            cumu_psnr += -10* np.sum(np.log10(spatial_mean_loss_per_sample ))  # average across samples
            total_samples += len(spatial_mean_loss_per_sample)
#             print("iter/total iter: %d/%d | current psnr %.3f" % (iter_idx, total_batch, cumu_psnr /  total_samples ))

        psnr = cumu_psnr / total_samples
        print(f'{phase} PSNR: {psnr}')
    return psnr




def SNRab(x, x_hat, db=True):
    # defined in page 7: https://openreview.net/pdf?id=HyGcghRct7
    # https://github.com/swing-research/deepmesh/blob/bbaaf26a08515d1abfde52cdaa75de698ccfddc1/utils/SNRab.py
    # better safe than sorry (matrix vs array)

    xx = x.flatten()
    yy = x_hat.flatten()
    
    u = xx.sum()
    v = (xx*yy).sum()
    w = (yy**2).sum()
    p = yy.sum()
    q = len(xx)**2
    
    a = (v*q - u*p)/(w*q - p*p)
    b = (w*u - v*p)/(w*q - p*p)
    
    SNRopt = np.sqrt((xx**2).sum() / ((xx - (a*yy + b))**2).sum())
    SNRraw = np.sqrt((xx**2).sum() / ((xx - yy)**2).sum())
    
    if db:
        SNRopt = 20*np.log10(SNRopt)
        SNRraw = 20*np.log10(SNRraw)

    return SNRopt, SNRraw, a, b

def calculate_snr(model, phase, loaders, device):
    model.eval()   # Set model to evaluate mode
    model.to(device)
    
    SNR_opt_sum = 0
    total_loss = 0
    loader = loaders[phase]
    loader.worker_init_fn(worker_id = 0)

    # Iterate over data.
    for iter_idx, (x, d) in enumerate(loader):
        x, d = x.to(device), d.to(device)
    
        with torch.no_grad():
            output = model(x)
                    
        SNR_opt = SNRab(d.cpu().detach().numpy(), output.cpu().detach().numpy(), db=True)[0]        
        SNR_opt_sum += SNR_opt # average across samples

    SNR_opt_sum /= iter_idx
    print(f'{phase} SNR: {SNR_opt_sum}')
    return SNR_opt_sum


def load_ista_unet_model(guid, dataset = 'bsds', model_save_dir = model_save_dir, return_config_dict = False):
    model_save_dir_guid = os.path.join(model_save_dir, dataset, guid) 

    with open(os.path.join(model_save_dir_guid, 'config_dict.pickle' ) , 'rb') as handle:
        config_dict = pickle.load(handle)
    
    model = ista_unet( ** config_dict)
    
    model_name = os.path.join(model_save_dir_guid, 'ista_unet.pt' )
    config_dict['saved_path'] = model_save_dir_guid
    
    model.load_state_dict(torch.load(model_name))
    
    model.eval();
    
    if return_config_dict:
        return model, config_dict
    else:    
        return model


def load_ista_unet_model_best_val(guid, dataset = 'bsds', model_save_dir = model_save_dir, return_config_dict = False):
    model_save_dir_guid = os.path.join(model_save_dir, dataset, guid) 

    pickle_filename = os.path.join(model_save_dir_guid, 'config_dict_best_val.pickle' )
    model_name = os.path.join(model_save_dir_guid, 'ista_unet_best_val.pt' )
    if not os.path.isfile(pickle_filename):
        print('best val model not found, falling back to latest model')
        pickle_filename = os.path.join(model_save_dir_guid, 'config_dict.pickle' )
        model_name = os.path.join(model_save_dir_guid, 'ista_unet.pt' )
    
    with open(pickle_filename , 'rb') as handle:
        config_dict = pickle.load(handle)
    
    model = ista_unet( ** config_dict)
    
    config_dict['saved_path'] = model_save_dir_guid
    
    model.load_state_dict(torch.load(model_name))
    
    model.eval();
    
    if return_config_dict:
        return model, config_dict
    else:    
        return model



def calculate_psnr_mri_exp(model, phase, loaders, device):
    model.eval()
    model.to(device)
    
    psnr = 0
    loader = loaders[phase]
    
#     iter_idx = 0
    
    with torch.set_grad_enabled(False):

        for zero_filled_rec, target, mask in (loader):

            zero_filled_rec, target, mask = zero_filled_rec.to(device), target.to(device), mask.to(device)

            s1, s2, s3, s4 = target.shape

            output = model(zero_filled_rec)
    #         output = mri_data_consistency_step(output, zero_filled_rec, mask)

            output = torch.norm(output, dim =1, keepdim = True)
            error = target - output

            error_power = (torch.norm(error, dim = (2, 3))[:, 0])**2/np.prod(error.shape[2:4]) 
            signal_peak = torch.max((target.abs().reshape(s1, s2*s3*s4))**2, dim = 1 ).values
            

            psnr += torch.mean( 10*torch.log10(signal_peak/error_power) )
#             iter_idx +=1

        psnr = psnr/(len(loader))

        print(f'{phase} PSNR: {psnr}')
    return psnr



# def calculate_psnr_mri_exp(model, phase, loaders, device):
#     model.eval()   # Set model to evaluate mode
#     model.to(device)
    
#     psnr = 0
#     total_loss = 0
#     loader = loaders[phase]
# #     loader.worker_init_fn(worker_id = 0)

#     # Iterate over data.
#     for iter_idx, (inp, target) in enumerate(loader):
#         inp, target = inp.to(device), target.to(device)
        
        

#         with torch.no_grad():
#             output = model(inp)
#             output = torch.norm(output, dim =1, keepdim = True)
            
#         loss_per_sample = torch.sum((output - target).pow(2), dim = (1, 2, 3) ).cpu().numpy()
#         spatial_mean_loss_per_sample = loss_per_sample / (np.prod(target.shape[1:4])) # channel_num * width * height
        
#         output_peak = (torch.max(output, dim = (1, 2, 3)).cpu().numpy())**2
        
#         psnr += -10* np.sum(np.log10(spatial_mean_loss_per_sample/output_peak )) / target.shape[0] # average across samples

#     psnr /= (iter_idx+1)
#     print(f'{phase} PSNR: {psnr}')
#     return psnr




# def load_homotopy_ista_unet_model(guid, dataset = 'bsds', model_save_dir = '/home/liu0003/Desktop/projects/Unet-sparsity/saved_model/'):
#     model_save_dir_guid = os.path.join(model_save_dir, guid) 

#     with open(os.path.join(model_save_dir_guid, 'config_dict.pickle' ) , 'rb') as handle:
#         config_dict = pickle.load(handle)
    
#     model = homotopy_ista_unet( ** config_dict)
    
#     model_name = os.path.join(model_save_dir_guid, 'homotopy_ista_unet.pt' )
    
#     model.load_state_dict(torch.load(model_name))
    
#     model.eval();
#     return model


# def load_model_datasets(guid, model_save_dir = '/home/liu0003/Desktop/projects/Unet-sparsity/saved_model/'):
#     model_save_dir_guid = os.path.join(model_save_dir, guid) 

#     with open(os.path.join(model_save_dir_guid, 'config_dict.pickle' ) , 'rb') as handle:
#         config_dict = pickle.load(handle)

#     config_dict['train_path'] ='/home/liu0003/Desktop/datasets/CBSD/CBSD432/'
#     config_dict['test_path'] ='/home/liu0003/Desktop/datasets/CBSD/CBSD68/'
    
#     config_dict['distributed_bool'] = False
#     loaders = get_dataloaders(** config_dict)    
#     return loaders
