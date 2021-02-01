import os, sys, uuid, torch
from os import path
from shutil import copyfile
import numpy as np
import random

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import torch.multiprocessing as mp
import argparse
import pickle 

from ista_unet.models import ista_unet
# from ista_unet.load_dival_datasets import get_dataloaders_ct
from ista_unet.load_apples_dataset import get_dataloaders_ct
from ista_unet.utils import seed_everything
from dival.measure import PSNR
from ista_unet import dataset_dir
from ista_unet import model_save_dir

parser = argparse.ArgumentParser(description='ISTA_unet training')


# Dataset settings
parser.add_argument('--BATCH_SIZE', type=int, default=1, help='mini-batch size for training.')
parser.add_argument('--IMAGE_CHANNEL_NUM', type=int, default=1, help='number of channels of an image.')


# Model settings
parser.add_argument("--model_name", type=str, default='ct', help="the name of the model to be saved.")
parser.add_argument('--KERNEL_SIZE', type=int, default=3, help='the kernel size used for 2D convolution.')
parser.add_argument("--HIDDEN_WIDTHS", type=int, nargs='+', default= [1024, 512, 256, 128, 64], help="")
parser.add_argument('--ISTA_NUM_STEPS', type=int, default=5, help='number of ISTA iterations.')
# parser.add_argument('--LASSO_LAMBDA_SCALAR', type=float, default=0.001, help='initialized LASSO parameter.')
parser.add_argument('--LASSO_LAMBDA_SCALAR', type=float, default=0.0001, help='initialized LASSO parameter.')
parser.add_argument('--RELU_OUT_BOOL', default=True, type=lambda x: (str(x).lower() == 'true') )
parser.add_argument('--UNCOUPLE_ADJOINT_BOOL', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--BILINEAR_UP_BOOL', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--BIAS_UP_BOOL', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--model_save_dir', type=str, default= model_save_dir, help='directory where trained models are saved.')
                
    
# Training settings
parser.add_argument('--NUM_EPOCH', type=int, default=20, help='number of training epochs.')
parser.add_argument('--LOSS_STR', type=str, default= 'nn.MSELoss()', help='the loss function from the torch.nn module.')
parser.add_argument('--LEARNING_RATE', type=float, default=1e-4, help='learning rate of gradient descent.')


# DistributedDataParallel settings
parser.add_argument('--num_workers', type=int, default=0, help='') # NOTE HERE!!!!! for some reason, num_workers > 0 does not work for now.
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0,1], help="")
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument('--noise_setting', default='gaussian_noise', type=str, help='')
parser.add_argument('--num_angles', default=50, type=int, help='')
parser.add_argument('--SCATTERING_SCALING_FACTOR', default=400., type=float, help='')
parser.add_argument('--resume_from_model_save_dir_guid', default=None, type=str, help='')


args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

SCATTERING_SCALING_FACTOR = args.SCATTERING_SCALING_FACTOR

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)


def main():
    set_random_seeds()
    args = parser.parse_args()


    ngpus_per_node = torch.cuda.device_count()

    args.world_size = ngpus_per_node * args.world_size

    if args.resume_from_model_save_dir_guid is not None:
        mp.spawn(main_worker_resume, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))
        
    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)


    print('==> Making model..')


    model_setting_dict = {'kernel_size' : args.KERNEL_SIZE,
                          'hidden_layer_width_list': args.HIDDEN_WIDTHS,
                          'n_classes': args.IMAGE_CHANNEL_NUM, 
                          'ista_num_steps': args.ISTA_NUM_STEPS,  
                          'lasso_lambda_scalar': args.LASSO_LAMBDA_SCALAR, 
                          'uncouple_adjoint_bool': args.UNCOUPLE_ADJOINT_BOOL,
                          'relu_out_bool': args.RELU_OUT_BOOL}

    model = ista_unet( ** model_setting_dict)    
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.BATCH_SIZE = int(args.BATCH_SIZE / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    print('==> Preparing data..')
    dataset_setting_dict = {'batch_size': args.BATCH_SIZE, 
                        'num_workers': args.num_workers,
                        'distributed_bool': True,
                        'noise_setting': args.noise_setting,
                        'num_angles': args.num_angles}
    

    loaders = get_dataloaders_ct(** dataset_setting_dict, include_validation = True )
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max= args.NUM_EPOCH,
                eta_min= 1e-5)


    fit_setting_dict = {'num_epochs': args.NUM_EPOCH,
                    'criterion': eval(args.LOSS_STR),
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                    'device': args.gpu,
                    'noise_setting': args.noise_setting}


    config_dict = {**model_setting_dict, **dataset_setting_dict, **fit_setting_dict}

    if args.gpu == 0: 
        guid = str(uuid.uuid4())
        guid_dir = os.path.join(args.model_save_dir, args.model_name, guid ) 
        os.makedirs(guid_dir) 
        config_dict['guid'] = guid
    else:
        guid_dir = None
    
    print(config_dict)

    trained_model = fit_model(model, loaders = loaders, config_dict=config_dict, guid_dir=guid_dir, ** fit_setting_dict)

    if args.gpu == 0: 
        torch.save(trained_model.module.state_dict(), os.path.join(guid_dir, 'ista_unet.pt'))

        psnrs = []
        trained_model.eval()
        trained_model.to(args.gpu)

        print('Evaluating model')
        with torch.no_grad():
            for obs, gt in loaders['validation']:
                if args.noise_setting == 'scattering':
                    obs = obs / SCATTERING_SCALING_FACTOR
                reco = trained_model(obs.to(args.gpu)).cpu()
                psnrs.append(PSNR(reco, gt))
        print('mean psnr: {:f}'.format(np.mean(psnrs)))


def main_worker_resume(gpu, ngpus_per_node, args):

    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))
        
    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)


    print('==> Making model..')


    model_setting_dict = {'kernel_size' : args.KERNEL_SIZE,
                          'hidden_layer_width_list': args.HIDDEN_WIDTHS,
                          'n_classes': args.IMAGE_CHANNEL_NUM, 
                          'ista_num_steps': args.ISTA_NUM_STEPS,  
                          'lasso_lambda_scalar': args.LASSO_LAMBDA_SCALAR, 
                          'uncouple_adjoint_bool': args.UNCOUPLE_ADJOINT_BOOL,
                          'relu_out_bool': args.RELU_OUT_BOOL}

    model = ista_unet( ** model_setting_dict)    
    model.load_state_dict(torch.load(os.path.join(args.resume_from_model_save_dir_guid, 'ista_unet.pt')))
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.BATCH_SIZE = int(args.BATCH_SIZE / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    print('==> Preparing data..')
    dataset_setting_dict = {'batch_size': args.BATCH_SIZE, 
                        'num_workers': args.num_workers,
                        'distributed_bool': True,
                        'noise_setting': args.noise_setting,
                        'num_angles': args.num_angles}

    with open(os.path.join(args.resume_from_model_save_dir_guid, 'config_dict.pickle' ) , 'rb') as handle:
        prev_config_dict = pickle.load(handle)

    loaders = get_dataloaders_ct(** dataset_setting_dict, include_validation = True )
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
    optimizer.load_state_dict(prev_config_dict['optimizer'].state_dict())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max= args.NUM_EPOCH,
                eta_min= 1e-5)
    scheduler.load_state_dict(prev_config_dict['scheduler'].state_dict())

    num_prev_epochs = scheduler.state_dict()['last_epoch']
    print('resuming from previous training that was terminated after {:d} epochs'.format(num_prev_epochs))

    fit_setting_dict = {'num_epochs': args.NUM_EPOCH-num_prev_epochs,
                    'criterion': eval(args.LOSS_STR),
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                    'device': args.gpu,
                    'noise_setting': args.noise_setting}


    config_dict = {**model_setting_dict, **dataset_setting_dict, **fit_setting_dict}
    config_dict['num_epochs'] = args.NUM_EPOCH  # overwrite value from fit_setting_dict with total number of epochs
    config_dict['guid_resumed_from'] = prev_config_dict['guid']

    if args.gpu == 0: 
        guid = str(uuid.uuid4())
        guid_dir = os.path.join(args.model_save_dir, args.model_name, guid ) 
        os.makedirs(guid_dir) 
        config_dict['guid'] = guid
    else:
        guid_dir = None
    
    print(config_dict)

    trained_model = fit_model(model, loaders = loaders, config_dict=config_dict, guid_dir=guid_dir, ** fit_setting_dict)

    if args.gpu == 0: 
        torch.save(trained_model.module.state_dict(), os.path.join(guid_dir, 'ista_unet.pt'))

        psnrs = []
        trained_model.eval()
        trained_model.to(args.gpu)

        print('Evaluating model')
        with torch.no_grad():
            for obs, gt in loaders['validation']:
                if args.noise_setting == 'scattering':
                    obs = obs / SCATTERING_SCALING_FACTOR
                reco = trained_model(obs.to(args.gpu)).cpu()
                psnrs.append(PSNR(reco, gt))
        print('mean psnr: {:f}'.format(np.mean(psnrs)))


def fit_model(model, optimizer, scheduler, num_epochs, criterion, loaders, device, config_dict, guid_dir, noise_setting, seed = 0):

    train_loader = loaders['train']    
    len_train_loader = len(train_loader) 
    seed_everything(seed = seed)

    print('start training')
    best_mean_psnr = -np.inf
    for epoch in tqdm(range(num_epochs) ) :

        model.train()
        loss = 0
        for i, (x, d) in enumerate(train_loader):
            x, d = x.cuda(device), d.cuda(device)
                    
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            if noise_setting == 'scattering':
                x = x / SCATTERING_SCALING_FACTOR
            outputs = model(x)
            # print(torch.mean(outputs), torch.mean(d))
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, d)
            # compute accumulated gradients
            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)            

            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss +=  float(train_loss) 
            
            if i % 100 == 0:
                print("iter : {}/{}, loss = {:g}".format(epoch * len_train_loader + i, len_train_loader * num_epochs, float(train_loss)))
         
        # compute the epoch training loss
        loss = float(loss) / len_train_loader   
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:g}".format(epoch + 1, num_epochs, loss))
        
        # update the step-size
        scheduler.step() 
                
        psnrs = []
        model.eval()
        print('Evaluating model')
        with torch.no_grad():
            for obs, gt in loaders['validation']:
                if noise_setting == 'scattering':
                    obs = obs / SCATTERING_SCALING_FACTOR
                reco = model(obs.to(device)).cpu()
                psnrs.append(PSNR(reco, gt))
        print('mean psnr: {:f}'.format(np.mean(psnrs)))

        if device == 0:
            with open(os.path.join(guid_dir, 'config_dict.pickle'), 'wb') as handle:
                pickle.dump(config_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            torch.save(model.module.state_dict(), os.path.join(guid_dir, 'ista_unet.pt'))
            
            if np.mean(psnrs) > best_mean_psnr:
                best_mean_psnr = np.mean(psnrs)
                copyfile(os.path.join(guid_dir, 'config_dict.pickle'),
                         os.path.join(guid_dir, 'config_dict_best_val.pickle'))
                copyfile(os.path.join(guid_dir, 'ista_unet.pt'),
                         os.path.join(guid_dir, 'ista_unet_best_val.pt'))
        
    return model

    
if __name__=='__main__':
    main()