import numpy as np
import torch.nn as nn
# import matplotlib.animation as anim
# from IPython.display import HTML
import matplotlib.pyplot as plt
import torch
import random
import os

def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    

def soft_thres(x, threshold):
    sparse_x = torch.sign(x) * torch.max(torch.abs(x) - threshold, torch.zeros_like(x) )
    return sparse_x
    
    
def crop_center_grayscale(img,frac = 0.2):
    y,x = img.shape
    cropx = int(x * frac)
    cropy = int(y * frac)
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def crop_center_rgb(img,frac = 0.2):
    y,x, _ = img.shape
    cropx = int(x * frac)
    cropy = int(y * frac)
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]



calculate_nonzero_fraction = lambda aa: round(np.count_nonzero(aa.data.cpu())/ np.size(aa.data.cpu().numpy()), 3)

def plot_activations(num_row, num_col, activation_3d_tensor, disp_unit = 3):

    if num_row * num_col > activation_3d_tensor.shape[0]:
        raise ValueError("num_row * num_col > num channels available")

    single_fig_shape = activation_3d_tensor[0, :, :].shape
    hw_ratio = single_fig_shape[0]/single_fig_shape[1]

    layermax = torch.max(activation_3d_tensor.flatten())
    layermin =  torch.min(activation_3d_tensor.flatten())
            
    _, axs = plt.subplots(num_row, num_col, figsize=(disp_unit * num_col, disp_unit * num_row * hw_ratio ))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(activation_3d_tensor[i, :, :], vmin = layermin, vmax = layermax)
        ax.axis('off')
        if i ==0 :
           nz_frac = calculate_nonzero_fraction(activation_3d_tensor)
           ax.set_title('Dim = ' + str( list(activation_3d_tensor[i, :, :].shape) ) + '\n nonzero frac =' + str(nz_frac))
    plt.subplots_adjust(wspace=0.01)

    return axs

def get_activations(net, net_input):
    activations = {}

    def hook_fn(m, i, o):
        activations[m] = o 

    def get_all_layers(net):
        for name, layer in net._modules.items():
        # if isinstance(layer, nn.Sequential):
        # if not isinstance(layer, nn.Conv2d):
            if not isinstance(layer, nn.ReLU):
                get_all_layers(layer)
            elif isinstance(layer, nn.ReLU):
               # it's a Conv2d layer. Register a hook
                layer.register_forward_hook(hook_fn)

    get_all_layers(net)
    
    with torch.no_grad():
         output_x = net(net_input)

    return activations


def crop_center_2d(img_2d,frac = 0.4):
    y,x = img_2d.shape
    cropx = int(x * frac)
    cropy = int(y * frac)
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img_2d[starty:starty+cropy,startx:startx+cropx]