import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torchvision
import numpy as np
import pytorch_lightning as pl
from dival.util.torch_utility import TorchRayTrafoParallel2DAdjointModule
from odl.tomo.analytic import fbp_filter_op
from odl.contrib.torch import OperatorModule
from util.external_libs.iunets.layers import InvertibleDownsampling2D


class CINN(pl.LightningModule):
    """
    PyTorch cINN architecture for low-dose CT reconstruction.
    
    Attributes
    ----------
    cinn : torch module list
        Building blocks of the conditional network.
    cond_net : torch module list
        Building blocks of the conditional network.

    Methods
    -------
    forward(c)
        Compute the forward pass.
        
    """

    def __init__(self, in_ch: int, img_size, operator,
                 conditional_args: dict = {'filter_type':'Hann',
                                          'frequency_scaling': 1.,},
                 optimizer_args: dict = {'lr': 0.0005,
                                        'weight_decay': 0},
                 clamping: float = 1.5, weight_mse: float = 1.0, **kwargs):
        """
        CINN constructor.

        Parameters
        ----------
        in_ch : int
            Number of input channels. This should be 1 for regular CT.
        img_size : tuple of int
            Size (h,w) of the input image.
        operator : Type
            Forward operator. This is the ray transform for CT.
        conditional_args : dict, optional
            Arguments for the conditional network.
            The default are options for the FBP conditioner: filter_type='hann'
            and frequency_scaling=1.
        optimizer_args : dict, optional
            Arguments for the optimizer.
            The defaults are lr=0.0005 and weight_decay=0 for the ADAM
            optimizer.
        clamping : float, optional
            The default is 1.5.        

        Returns
        -------
        None.

        """
        super().__init__()
        
        # all inputs to init() will be stored (if possible) in a .yml file 
        # alongside the model. You can access them via self.hparams.
        self.save_hyperparameters()
        
        # shorten some of the names or store values that can't be placed in
        # a .yml file
        self.in_ch = self.hparams.in_ch
        self.img_size = self.hparams.img_size
        self.downsample_levels = 5
        self.op = operator
                
        # choose the correct loss function
        self.criterion = CINNNLLLoss(distribution='normal')
        
        # set the list of downsamling layers
        self.ds_list = ['invertible'] * self.downsample_levels   
        
        # initialize the input padding layer
        pad_size = (self.img_size[0] - self.op.domain.shape[0],
                    self.img_size[1] - self.op.domain.shape[1]) 
        self.img_padding = torch.nn.ReflectionPad2d((0, pad_size[0],
                                                     0, pad_size[1]))
        
        # build the cINN
        self.cinn = self.build_inn()
        
        # set the conditioning network
        self.cond_net = CondNetFBP(ray_trafo=self.op, 
                                   img_size=self.img_size,
                                   downsample_levels=self.downsample_levels,
                                   **conditional_args)
        
        # initialize the values of the parameters
        self.init_params()
        
      
    def build_inn(self):
        """
        Connect the building blocks of the cINN.

        Returns
        -------
        FrEIA ReversibleGraphNet
            cINN model.

        """
        
        nodes = [Ff.InputNode(self.in_ch, self.img_size[0], self.img_size[1],
                              name='inp')]

        conditions = [Ff.ConditionNode(4*self.in_ch,
                                       int(1/2*self.img_size[0]),
                                       int(1/2*self.img_size[1]), 
                                       name='cond_1'),
                      Ff.ConditionNode(8*self.in_ch,
                                       int(1/4*self.img_size[0]),
                                       int(1/4*self.img_size[1]), 
                                       name='cond_2'), 
                      Ff.ConditionNode(16*self.in_ch,
                                       int(1/8*self.img_size[0]),
                                       int(1/8*self.img_size[1]),
                                       name='cond_3'),
                      Ff.ConditionNode(32*self.in_ch,
                                       int(1/16*self.img_size[0]),
                                       int(1/16*self.img_size[1]),
                                       name='cond_4'),
                      Ff.ConditionNode(64*self.in_ch,
                                       int(1/32*self.img_size[0]),
                                       int(1/32*self.img_size[1]),
                                       name='cond_5')]

        split_nodes = []
        
        _add_downsample(nodes, self.ds_list[0], in_ch=self.in_ch)

        # Condition level 0
        _add_conditioned_section(nodes, depth=4, in_ch=4*self.in_ch, 
                                 cond_level=0, conditions=conditions)

        _add_downsample(nodes, self.ds_list[1], in_ch=4*self.in_ch)

        # Condition level 1
        _add_conditioned_section(nodes, depth=4, in_ch=16*self.in_ch, 
                                 cond_level=1, conditions=conditions)

        _add_downsample(nodes, self.ds_list[2], in_ch=16*self.in_ch)
        
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections':[32*self.in_ch,
                                                        32*self.in_ch],
                              'dim':0}, name="split_1"))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {},
                                   name='flatten_split_1'))

        # Condition level 2
        _add_conditioned_section(nodes, depth=4, in_ch=32*self.in_ch, 
                                 cond_level=2, conditions=conditions)
        
        _add_downsample(nodes, self.ds_list[3], in_ch=32*self.in_ch)
        
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections':[64*self.in_ch,
                                                        64*self.in_ch],
                              'dim':0}, name="split_2"))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {},
                                   name='flatten_split_2'))
        
        # Condition level 3
        _add_conditioned_section(nodes, depth=4, in_ch=64*self.in_ch, 
                                 cond_level=3, conditions=conditions)
        
        _add_downsample(nodes, self.ds_list[4], in_ch=64*self.in_ch)

        nodes.append(Ff.Node(nodes[-1], Fm.Split1D, 
                             {'split_size_or_sections':[128*self.in_ch,
                                                        128*self.in_ch],
                              'dim':0}, name="split_3"))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {},
                                   name='flatten_split_3'))

        # Condition level 4
        _add_conditioned_section(nodes, depth=4, in_ch=128*self.in_ch, 
                                 cond_level=4, conditions=conditions) 

        nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {}, name='flatten'))       


        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}, name='concat_splits'))
  
        nodes.append(Ff.OutputNode(nodes[-1], name='out'))
        
        return Ff.ReversibleGraphNet(nodes + conditions + split_nodes,
                                     verbose=False)
    
    def init_params(self):
        """
        Initialize the parameters of the model.

        Returns
        -------
        None.

        """
        # approx xavier
        for p in self.cond_net.parameters():
            p.data = 0.02 * torch.randn_like(p) 
            
        for key, param in self.cinn.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = 0.02 * torch.randn(param.data.shape)
                if len(split) > 3 and split[3][-1] == '4':
                    param.data.fill_(0.)
    
    def forward(self, cinn_input, cond_input, rev:bool = True,
                cut_ouput:bool = True):
        """
        Inference part of the whole model. There are two directions of the
        cINN. These are controlled by rev:
            rev==True:  Create a reconstruction x for a random sample z
                        and the conditional measurement y (Z|Y) -> X.
            rev==False: Create a sample z from a reconstruction x
                        and the conditional measurement y (X|Y) -> Z .

        Parameters
        ----------
        cinn_input : torch tensor
            Input to the cINN model. Depends on rev:
                rev==True: Random vector z.
                rev== False: Reconstruction x.
        cond_input : torch tensor
            Input to the conditional network. This is the measurement y.
        rev : bool, optional
            Direction of the cINN flow. For True it is Z -> X to create a 
            single reconstruction. Otherwise X -> Z.
            The default is True.
        cut_ouput : bool, optional
            Cut the output of the network to the domain size of the operator.
            This is only relevant if rev==True.
            The default is True.

        Returns
        -------
        torch tensor or tuple of torch tensor
            rev==True:  x : Reconstruction
            rev==False: z : Sample from the target distribution
                        log_jac : log det of the Jacobian

        """
        # direction (Z|Y) -> X
        if rev:
            x = self.cinn(cinn_input, c=self.cond_net(cond_input), rev=rev)

            if cut_ouput:
                return x[:,:,:self.op.domain.shape[0],:self.op.domain.shape[1]]
            else:    
                return x
        # direction (X|Y) -> Z
        else:
            cinn_input = self.img_padding(cinn_input)
            z = self.cinn(cinn_input, c=self.cond_net(cond_input), rev=rev)
            log_jac = self.cinn.log_jacobian(run_forward=False)
            return z, log_jac

    def training_step(self, batch, batch_idx):
        """
        Pytorch Lightning training step. Should be independent of forward() 
        according to the documentation. The loss value is logged.

        Parameters
        ----------
        batch : tuple of tensor
            Batch of measurement y and ground truth reconstruction gt.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        result : TYPE
            Result of the training step.

        """
        y, gt = batch

        # pad gt image to the right size
        gt = self.img_padding(gt)

        # run the conditional network
        c = self.cond_net(y)
        
        # run the cINN from X -> Z with the gt data and conditioning
        zz = self.cinn(gt, c)
        log_jac = self.cinn.log_jacobian(run_forward=False)

        # evaluate the NLL loss
        loss = self.criterion(zz=zz, log_jac=log_jac)
        self.log('train_nll', loss)

        if self.hparams.weight_mse > 0.0:
            z = torch.randn((gt.shape[0], self.img_size[0]*self.img_size[1]),
                            device=self.device)
            x_rec = self.cinn(z, c, rev=True)
            
            l2_loss = self.hparams.weight_mse * torch.sum((x_rec - gt) ** 2) / gt.shape[0]
            
            self.log('train_mse_backward', l2_loss)
            
            loss = loss + l2_loss        

        self.log('train_loss', loss+ l2_loss)
        
        self.last_batch = batch

        return loss + l2_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Pytorch Lightning validation step. Should be independent of forward() 
        according to the documentation. The loss value is logged and the
        best model according to the loss (lowest) checkpointed.

        Parameters
        ----------
        batch : tuple of tensor
            Batch of measurement y and ground truth reconstruction gt.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        result : tensor
            Result of the validation step.

        """
        y, gt = batch
        
        # pad gt image to the right size
        gt = self.img_padding(gt)
        
        # run the conditional network
        c = self.cond_net(y)
        
        # run the cINN from X -> Z with the gt data and conditioning
        zz = self.cinn(gt, c)
        log_jac = self.cinn.log_jacobian(run_forward=False)
        
        # evaluate the NLL loss
        loss = self.criterion(zz=zz, log_jac=log_jac)
        self.log('val_nll_loss', loss)
        
        if self.hparams.weight_mse > 0.0:
            z = torch.randn((gt.shape[0], self.img_size[0]*self.img_size[1]),
                            device=self.device)
            x_rec = self.cinn(z, c, rev=True)
            
            l2_loss = self.hparams.weight_mse * torch.sum((x_rec - gt) ** 2) / gt.shape[0]
            
            self.log('val_mse_backward', l2_loss)
            
            loss = loss + l2_loss
        
        # checkpoint the model and log the loss
        self.log('val_loss', loss)

        return loss
    
    def training_epoch_end(self, result):
        y, gt = self.last_batch
        img_grid = torchvision.utils.make_grid(gt, normalize=True,
                                               scale_each=True)

        self.logger.experiment.add_image("ground truth",
                    img_grid, global_step=self.current_epoch)
        
        z = torch.randn((gt.shape[0], self.img_size[0]*self.img_size[1]),
                        device=self.device)

        with torch.no_grad():
            c = self.cond_net.forward(y)

            cond_level = 0
            for cond in c:
                cond = cond.view(-1, 1, cond.shape[-2], cond.shape[-1])
                cond_grid = torchvision.utils.make_grid(cond, normalize=True,
                                                        scale_each=True)

                self.logger.experiment.add_image("cond_level_" + str(cond_level),
                    cond_grid, global_step=self.current_epoch)
                cond_level += 1

            fbp = self.cond_net.fbp_layer(y)

            x = self.forward(z, y, rev=True, cut_ouput=True)
            
            reco_grid = torchvision.utils.make_grid(x, normalize=True,
                                                    scale_each=True)
            self.logger.experiment.add_image("reconstructions",
                    reco_grid, global_step=self.current_epoch)

            fbp_grid = torchvision.utils.make_grid(fbp, normalize=True, 
                                                   scale_each=True)
            self.logger.experiment.add_image("filtered_back_projection",
                    fbp_grid, global_step=self.current_epoch)

    def configure_optimizers(self):
        """
        Setup the optimizer. Currently, the ADAM optimizer is used.

        Returns
        -------
        optimizer : torch optimizer
            The Pytorch optimizer.

        """
        optimizer = torch.optim.Adam(self.parameters(),
                        lr=self.hparams.optimizer_args['lr'], 
                        weight_decay=self.hparams.optimizer_args['weight_decay'])
        
        sched_factor = 0.4 # new_lr = lr * factor
        sched_patience = 2 
        sched_trehsh = 0.005
        sched_cooldown = 1

        reduce_on_plateu = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, factor=sched_factor,
                            patience=sched_patience, threshold=sched_trehsh,
                            min_lr=0, eps=1e-08, cooldown=sched_cooldown,
                            verbose = False)

        schedulers = {
         'scheduler': reduce_on_plateu,
         'monitor': 'val_loss', 
         'interval': 'epoch',
         'frequency': 1}

        return [optimizer], [schedulers]


def random_orthog(n):
    """
    Create a random, orthogonal n x n matrix.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    torch float tensor
        Orthogonal matrix.

    """
    w = np.random.randn(n, n)
    w = w + w.T
    w, _, _ = np.linalg.svd(w)
    return torch.FloatTensor(w)


class CondNetFBP(nn.Module):
    """
    Conditional network H that sits on top of the invertible architecture. It 
    features a FBP operation at the beginning and continues with post-
    processing steps.
    
    Attributes
    ----------
    resolution_levels : torch module list
        Building blocks of the conditional network.

    Methods
    -------
    forward(c)
        Compute the forward pass.
        
    """
    
    def __init__(self, ray_trafo, img_size, downsample_levels:int = 5,
                 filter_type:str = 'Hann', frequency_scaling:float = 1.):
        """
        

        Parameters
        ----------
        ray_trafo : TYPE
            DESCRIPTION.
        img_size : TYPE
            DESCRIPTION.
        filter_type : TYPE, optional
            DESCRIPTION. The default is 'Hann'.
        frequency_scaling : TYPE, optional
            DESCRIPTION. The default is 1..

        Returns
        -------
        None.

        """
        super().__init__()

        # FBP and resizing layers
        self.fbp_layer = FBPModule(ray_trafo, filter_type=filter_type,
                 frequency_scaling=frequency_scaling)
        
        self.img_size = img_size
        self.dsl = downsample_levels

        pad_size = (img_size[0] - ray_trafo.domain.shape[0],
                    img_size[1] - ray_trafo.domain.shape[1]) 
        self.img_padding = torch.nn.ReflectionPad2d((0, pad_size[0],
                                0, pad_size[1]))
        
        self.shapes = [1, 4, 8, 16, 32, 64]

        levels = []

        for i in range(self.dsl):
            levels.append(self.create_subnetwork(ds_level=i))

        self.resolution_levels = nn.ModuleList(levels)

    def forward(self, c):
        """
        Computes the forward pass of the conditional network and returns the
        results of all building blocks in self.resolution_levels.

        Parameters
        ----------
        c : torch tensor
            Input to the conditional network (measurement).

        Returns
        -------
        List of torch tensors
            Results of each block of the conditional network.

        """
        
        c = self.fbp_layer(c)
        c = self.img_padding(c)
        
        outputs = []
        for m in self.resolution_levels:
            outputs.append(m(c))
        return outputs

    def create_subnetwork(self, ds_level:int):
        padding = [4, 2, 2, 2, 1, 1, 1]
        kernel = [9, 5, 5, 5, 3, 3]

        modules = []
        
        for i in range(ds_level+1):
            modules.append(nn.Conv2d(in_channels=self.shapes[i], 
                                    out_channels=self.shapes[i+1], 
                                    kernel_size=kernel[i], 
                                    padding=padding[i], 
                                    stride=2))
              
            modules.append(nn.LeakyReLU())

        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.shapes[ds_level+1],
                                kernel_size=1))

        return nn.Sequential(*modules)


class Flatten(nn.Module):
    """
    Torch module for flattening an input.

    Methods
    -------
    forward(x)
        Compute the forward pass.
        
    """
    
    def __init__(self, *args):
        super().__init__()
    
    def forward(self, x):
        """
        Will just leave the channel dimension and combine all 
        following dimensions.

        Parameters
        ----------
        x : torch tensor
            Input for the flattening.

        Returns
        -------
        torch tensor
            Flattened torch tensor.

        """
        return x.view(x.shape[0], -1)
            
    
class FBPModule(torch.nn.Module):
    """
    Torch module of the filtered back-projection (FBP).

    Methods
    -------
    forward(x)
        Compute the FBP.
        
    """
    
    def __init__(self, ray_trafo, filter_type='Hann',
                 frequency_scaling=1.):
        super().__init__()
        self.ray_trafo = ray_trafo
        filter_op = fbp_filter_op(self.ray_trafo,
                          filter_type=filter_type,
                          frequency_scaling=frequency_scaling)
        self.filter_mod = OperatorModule(filter_op)
        self.ray_trafo_adjoint_mod = (
            TorchRayTrafoParallel2DAdjointModule(self.ray_trafo))
        
    def forward(self, x):
        x = self.filter_mod(x)
        x = self.ray_trafo_adjoint_mod(x)
        return x


def _add_conditioned_section(nodes, depth, in_ch, cond_level, conditions,
                             clamping=1.5):
    """
    Add conditioned notes to the network.

    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    depth : int
        Number of layer units in this block.
    in_ch : int
        Number of input channels.
    cond_level : int
        Current conditioning level.
    conditions : TYPE
        List of FrEIA condition notes.

    Returns
    -------
    None.

    """

    for k in range(depth):
        
        nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {},
                             name="ActNorm_{}_{}".format(cond_level, k)))


        nodes.append(Ff.Node(nodes[-1].out0, NICECouplingBlock,
                            {'F_args':{'leaky_slope': 5e-2,
                                       'channels_hidden':in_ch*2,
                                       'kernel_size': 3 if k % 2 == 0 else 1}},
                            conditions = conditions[cond_level],
                            name="NICEBlock_{}_{}".format(cond_level, k)))

        nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, {'seed':k}, 
                                 name='PermuteRandom{}_{}'.format(cond_level,
                                                                  k)))
        
def _add_downsample(nodes, downsample:str, in_ch:int, clamping:float = 1.5):
    """
    Downsampling operations.

    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    downsample : str
        Downsampling method. Currently there is one option: 'invertible'
    in_ch : int
        Number of input channels.
    clamping : float, optional
        The default value is 1.5.

    Returns
    -------
    None.

    """
    
    if downsample == 'invertible':
        nodes.append(Ff.Node(nodes[-1].out0, InvertibleDownsampling,
                             {'stride':2, 'method':'cayley', 'init':'haar',
                              'learnable':True}, name='invertible')) 
    else:
        raise NotImplementedError
          
    for i in range(2):

        nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {},
                             name="DS_ActNorm_{}".format(i)))

        nodes.append(Ff.Node(nodes[-1].out0, NICECouplingBlock,
                            {'F_args':{'leaky_slope': 5e-2,
                                       'channels_hidden':in_ch*2,
                                       'kernel_size': 1}},
                            name="DS_NICECoupling_{}".format(i)))

        nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, {'seed':i}, 
                                 name='DS_PermuteRandom_{}'.format(i)))


class F_conv(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, out_channels, channels_hidden=None,
                 kernel_size=3, leaky_slope=0.1):
        super().__init__()

        if not channels_hidden:
            channels_hidden = out_channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.conv1 = nn.Conv2d(in_channels, channels_hidden,
                               kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(channels_hidden, channels_hidden,
                               kernel_size=kernel_size, padding=pad)
        self.conv3 = nn.Conv2d(channels_hidden, out_channels,
                               kernel_size=kernel_size, padding=pad)

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out, self.leaky_slope)

        out = self.conv2(out)
        out = F.leaky_relu(out, self.leaky_slope)

        out = self.conv3(out)

        return out


class NICECouplingBlock(nn.Module):
    """
    From: github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/coupling_layers.py
    added argument c=[] to .jacobian()
    and chanded assert in __init__ to same in GlowCouplingBlock. 
    
    Coupling Block following the NICE design.
    subnet_constructor: 
        function or class, with signature constructor(dims_in, dims_out).
        The result should be a torch nn.Module, that takes dims_in input
        channels, nd dims_out output channels. See tutorial for examples.
    """

    def __init__(self, dims_in, dims_c=[],F_args={}):
        super().__init__()

        channels = dims_in[0][0]
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.F = F_conv(self.split_len2 + condition_length, self.split_len1,
                        **F_args)

        self.G = F_conv(self.split_len1 + condition_length, self.split_len2,
                        **F_args)


    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            y1 = x1 + self.F(x2_c)
            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            y2 = x2 + self.G(y1_c)
        else:
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            y2 = x2 - self.G(x1_c)
            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            y1 = x1 - self.F(y2_c)

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


def subnet_conv3x3(in_ch, out_ch):
    """
    Sub-network with 3x3 2d-convolutions and leaky ReLU activation.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.

    Returns
    -------
    torch sequential model
        The sub-network.

    """
    return nn.Sequential(
                nn.Conv2d(in_ch, 128, 3, padding=1), 
                nn.LeakyReLU(), 
                nn.Conv2d(128, 128, 3, padding=1), 
                nn.LeakyReLU(),
                nn.Conv2d(128, out_ch, 3, padding=1))


def subnet_conv1x1(in_ch, out_ch):
    """
    Sub-network with 1x1 2d-convolutions and leaky ReLU activation.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.

    Returns
    -------
    torch sequential model
        The sub-network.

    """
    return nn.Sequential(
                nn.Conv2d(in_ch, 128, 1), 
                nn.LeakyReLU(), 
                nn.Conv2d(128, 128, 1), 
                nn.LeakyReLU(),
                nn.Conv2d(128, out_ch, 1))


def subnetUncond(in_ch, out_ch):
    """
    Sub-netwok with 1x1 2d-convolutions for unconditioned parts of the cINN.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.

    Returns
    -------
    torch sequential model
        The sub-network.

    """
    return nn.Sequential(
                nn.Conv2d(in_ch, 128, 1),
                nn.LeakyReLU(), 
                nn.Conv2d(128, 128, 1), 
                nn.LeakyReLU(),
                nn.Conv2d(128, out_ch, 1))


def subnet_fc(in_ch, out_ch):
    """
    Sub-network with fully connected layers and leaky ReLU activation.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.

    Returns
    -------
    torch sequential model
        The sub-network.

    """
    return nn.Sequential(nn.Linear(in_ch, 512), 
                         nn.LeakyReLU(), 
                         nn.Linear(512, 512),
                         nn.LeakyReLU(),
                         nn.Linear(512, out_ch))


class CINNNLLLoss(_Loss):
    def __init__(self,  distribution: str, size_average=None, reduce=None, 
                 reduction: str = 'mean') -> None:
        """
        Class for negative log-likelihood loss for cINN models.

        Parameters
        ----------
        distribution : str
            Target distribution for the model:
                'normal': Normal distribution
                'radial': Radial distribution (not yet implemented)
        size_average : TYPE, optional
            DESCRIPTION. 
            The default is None.
        reduce : TYPE, optional
            DESCRIPTION. 
            The default is None.
        reduction : str, optional
            DESCRIPTION. 
            The default is 'mean'.

        Returns
        -------
        None

        """
        super(CINNNLLLoss, self).__init__(size_average, reduce, reduction)
        self.distribution = distribution

    def forward(self, zz, log_jac):
        """
        

        Parameters
        ----------
        zz : TYPE
            DESCRIPTION.
        log_jac : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.distribution == 'normal':
            return cinn_normal_nll_loss(zz=zz, log_jac=log_jac,
                                        reduction=self.reduction)
        else:
            raise NotImplementedError

def cinn_normal_nll_loss(zz, log_jac, reduction='mean'):
    """
    Negative log-likelihood loss for a cINN model with a normal distribution
    as target.

    Parameters
    ----------
    zz : torch tensor
        Vector from the normal distribution.
    log_jac : torch tensor
        Log det of the Jacobian.

    Returns
    -------
    torch tensor
        NLL score for zz.

    """

    ndim_total = zz.shape[-1]
    c =  ndim_total / 2. * torch.log(torch.tensor(2.0*3.14159))
    ret = c / ndim_total + torch.mean(zz**2) / 2 - torch.mean(log_jac) / ndim_total

    ret = ret / torch.log(torch.tensor(2.)) + 8.
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


class InvertibleDownsampling(InvertibleDownsampling2D):
    """
    Wrapper class for the InvertibleDownsampling2D from the iUnet code. Adds
    functionality to make the original code work with the FrEIA framework.
    """
    
    def __init__(self, input_dims, stride=2, method='cayley', init='haar',
                 learnable=True, *args, **kwargs):
        super(InvertibleDownsampling, self).__init__(
            in_channels=input_dims[0][0],
            stride=stride,
            method=method,
            init=init,
            learnable=learnable,
            *args,
            **kwargs)
        
    def forward(self, x, rev=False):
        if not rev:
            out = super(InvertibleDownsampling, self).forward(x[0])
        else:
            out = super(InvertibleDownsampling, self).inverse(x[0])
            
        return [out]
    
    def jacobian(self, x, rev=False):
        """
        This should return the log det of the layer according to the FrEIA
        code. The name is a bit confusing. Since the operator is SO:
        det=1 and therefore log(det)=0.

        Parameters
        ----------
        x : torch tensor
            Network input.
        rev : bool, optional
            Reverse network direction. The default is False.

        Returns
        -------
        int
            Log det of the jacobian.

        """
        return 0

    def output_dims(self, input_dims):
        """
        Calculates the output dimension of the invertible downsampling.
        Currently, only a stride of 2 is supported

        Parameters
        ----------
        input_dims : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c*self.stride[0] * self.stride[1], w//self.stride[0], h//self.stride[1]
        self.elements = c*w*h
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]
