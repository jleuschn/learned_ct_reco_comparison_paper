import os, sys, uuid, torch
from os import path
import numpy as np
import random
from collections import defaultdict 
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import pickle 

from ista_unet.models import ista_unet
from ista_unet.load_fastmri_dataset import get_dataloaders_fastmri
from ista_unet.utils import seed_everything
from fastmri.evaluate import psnr


from ista_unet import dataset_dir
from ista_unet import model_save_dir

parser = argparse.ArgumentParser(description='ISTA_unet training')


# Dataset settings
parser.add_argument('--BATCH_SIZE', type=int, default=4, help='mini-batch size for training.')
parser.add_argument('--IMAGE_CHANNEL_NUM', type=int, default=1, help='number of channels of an image.')


# Model settings
parser.add_argument("--model_name", type=str, default='fastmri', help="the name of the model to be saved.")
parser.add_argument('--KERNEL_SIZE', type=int, default=3, help='the kernel size used for 2D convolution.')
# parser.add_argument("--HIDDEN_WIDTHS", type=int, nargs='+', default= [512, 256, 128, 64, 32], help="")
parser.add_argument("--HIDDEN_WIDTHS", type=int, nargs='+', default= [1024, 512, 256, 128, 64], help="")
parser.add_argument('--ISTA_NUM_STEPS', type=int, default=5, help='number of ISTA iterations.')
parser.add_argument('--LASSO_LAMBDA_SCALAR', type=float, default=0.001, help='initialized LASSO parameter.')
parser.add_argument('--RELU_OUT_BOOL', default=False, type=lambda x: (str(x).lower() == 'true') )
parser.add_argument('--UNCOUPLE_ADJOINT_BOOL', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--BILINEAR_UP_BOOL', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--BIAS_UP_BOOL', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--model_save_dir', type=str, default= model_save_dir, help='directory where trained models are saved.')
    
# Training settings
parser.add_argument('--NUM_EPOCH', type=int, default=20, help='number of training epochs.')
parser.add_argument('--LOSS_STR', type=str, default= 'nn.MSELoss()', help='the loss function from the torch.nn module.')
# parser.add_argument('--LEARNING_RATE', type=float, default=1e-4, help='learning rate of gradient descent.')
parser.add_argument('--LEARNING_RATE', type=float, default=5e-5, help='learning rate of gradient descent.')

# DistributedDataParallel settings
parser.add_argument('--num_workers', type=int, default=0, help='') # NOTE HERE!!!!! for some reason, num_workers > 0 does not work for now.
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0,1], help="")
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')


args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

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
                        'distributed_bool': True}
    

    loaders = get_dataloaders_fastmri(** dataset_setting_dict, include_test = False )
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max= args.NUM_EPOCH,
                eta_min= 5e-6)


    fit_setting_dict = {'num_epochs': args.NUM_EPOCH,
                    'criterion': eval(args.LOSS_STR), 
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                    'device': args.gpu} 


    config_dict = {**model_setting_dict, **dataset_setting_dict, **fit_setting_dict}

    print(config_dict)
    
    trained_model = fit_model(model, loaders = loaders, ** fit_setting_dict)

    if args.gpu == 0: 
        guid = str(uuid.uuid4())
        guid_dir = os.path.join(args.model_save_dir, args.model_name, guid ) 
        os.makedirs(guid_dir) 

        config_dict['guid'] = guid
        with open(os.path.join(guid_dir, 'config_dict.pickle'), 'wb') as handle:
            pickle.dump(config_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        torch.save(trained_model.module.state_dict(), os.path.join(guid_dir, 'ista_unet.pt'))

        
                
def fit_model(model, optimizer, scheduler, num_epochs, criterion, loaders, device, seed = 0):

    train_loader = loaders['train']    
    len_train_loader = len(train_loader) 
    seed_everything(seed = seed)
    
#     if device == 0:
#         psnr_array = evaluate_fastmri_without_save(model, loaders, device, 'validation')
#         print('mean psnr: {:f}'.format(np.mean(psnr_array)))
    
    print('start training')
    for epoch in tqdm(range(num_epochs) ) :

        model.train()
        loss = 0
        for i, (x, d, _, _, _, _, _ ) in enumerate(train_loader):
            x = x.unsqueeze(1)
            d = d.unsqueeze(1)
            x, d = x.cuda(device), d.cuda(device)
                    
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(x)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, d)
            # compute accumulated gradients
            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)            

            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss +=  float(train_loss) 
            if train_loss > 100:
                print(train_loss)
            if i % 100 == 0:
                print("iter : {}/{}, loss = {:.6f}".format(epoch * len_train_loader + i, len_train_loader * num_epochs, float(train_loss)))
         
        # compute the epoch training loss
        loss = float(loss) / len_train_loader   
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))
        
        # update the step-size
        scheduler.step() 
        psnr_array = evaluate_fastmri_without_save(model, loaders, device, 'validation')
        print('mean psnr: {:f}'.format(np.mean(psnr_array)))
        
    return model


def evaluate_fastmri_without_save(model, loader, device, phase):

    outputs = defaultdict(list)
    targets = defaultdict(list)

    with torch.no_grad():
        with tqdm(loader[phase]) as pbar:
            for obs, gt,  mean, std, fname, slice_num, max_value  in pbar:
                obs = obs.unsqueeze(1)
                gt = gt.unsqueeze(1)

                mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3)

                reco = model(obs.to(device)).cpu()

                # undo the instance-normalized the output and target
                trans_obs = (obs * std + mean).detach().numpy().squeeze()
                trans_reco = (reco * std + mean).detach().numpy().squeeze()
                trans_target = (gt * std + mean).detach().numpy().squeeze()

                for fname_idx in range(len(fname)):
                    # collect slices into the volume it belongs to
                    outputs[fname[fname_idx] ].append((slice_num.numpy()[fname_idx], trans_reco[fname_idx] ))
                    targets[fname[fname_idx] ].append((slice_num.numpy()[fname_idx], trans_target[fname_idx] ))

    stacked_outputs = defaultdict(list)
    stacked_targets = defaultdict(list)

    for fname in outputs.keys():
                
        outputs_list = [out for _, out in sorted(outputs[fname], key=lambda x: x[0]) if out.shape == (320, 320) ]
        targets_list = [target for _, target in sorted(targets[fname], key=lambda x: x[0]) if target.shape == (320, 320) ]
        
        
        stacked_outputs[fname] = np.stack(outputs_list)
        stacked_targets[fname] = np.stack(targets_list)

    psnr_list = []
    for fname in outputs:
        target = stacked_targets[fname]
        recons = stacked_outputs[fname]

        psnr_list.append( psnr(target, recons ) ) 

    psnr_array = np.array( psnr_list  )
    
    return psnr_array

if __name__=='__main__':
    main()