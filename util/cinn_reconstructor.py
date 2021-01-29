import os
from warnings import warn
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
import numpy as np
from dival.reconstructors import StandardLearnedReconstructor
import pytorch_lightning as pl

from util.cinn import CINN


class CINNReconstructor(StandardLearnedReconstructor):
    """
    Dival reconstructor class for the cINN network.
    """

    HYPER_PARAMS = deepcopy(StandardLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'epochs': {
            'default': 200,
            'retrain': True
        },
        'lr': {
            'default': 0.0005,
            'retrain': True
        },
        'weight_decay': {
            'default': 0,
            'retrain': True
        },
        'batch_size': {
            'default': 10,
            'retrain': True
        },
        'clamping': {
            'default': 1.5,
            'retrain': True
        },
        'filter_type': {
            'default': 'Hann',
            'retrain': True
        },
        'frequency_scaling': {
            'default': 1.0,
            'retrain': True
        },
        'samples_per_reco': {
            'default': 100,
            'retrain': False
        },
        'weight_mse' : {
            'default': 1.0,
            'retrain': True
        }
    })

    def __init__(self,
                 ray_trafo,
                 in_ch: int = 1,
                 img_size=None,
                 max_samples_per_run: int = 100,
                 trainer_args:dict = {'distributed_backend': 'ddp',
                                      'gpus': [0]},
                 **kwargs):
        """
        Parameters
        ----------
        ray_trafo : TYPE
            Ray transformation (forward operator).
        in_ch : int, optional
            Number of input channels.
            The default is 1.
        img_size : tuple of int, optional
            Internal size of the reconstructed image. This must be divisible by
            2**i for i=1,...,downsample_levels. By choosing None an optimal
            image size will be determined automatically.
            The default is None.
        max_samples_per_run : int, optional
            Max number of samples for a single run of the network. Adapt to the
            memory of your GPU. To reach the desired samples_per_reco,
            multiple runs of the network will automatically be performed.
            The default is 100.
        trainer_args : dict, optional
            Arguments for the Pytorch Trainer.
            The defaults are distributed_backend='ddp' and gpus=[0]
        Returns
        -------
        None.

        """
        
        super().__init__(ray_trafo,  **kwargs)
        
        self.in_ch = in_ch
        self.max_samples_per_run = max_samples_per_run
        self.downsample_levels = 5
        
        self.trainer_args = trainer_args
        
        self.trainer = pl.Trainer(max_epochs=self.epochs, **self.trainer_args)
        
        if img_size is None:
            self.img_size = self.calc_img_size()
        else:
            self.img_size = img_size
            
        self.model_initialized = False
        
    def calc_img_size(self):
        """
        Calculate the optimal image size for the desired downsampling level.
        The image size must be divisible by 2**i for i=1,...,downsample_levels
        
        The size will only be increased to avoid information loss.

        Returns
        -------
        img_size : tuple of int
            New image size.

        """
        img_size = self.op.domain.shape
        found_shape = False
        
        while not found_shape:
            if (sum([img_size[0] % 2**(i+1) for i in range(self.downsample_levels)]) + 
                sum([img_size[1] % 2**(i+1) for i in range(self.downsample_levels)]) 
                ) == 0:
               found_shape = True
            else:
               found_shape = False
               img_size = (img_size[0]+1, img_size[1]+1)
               
        return img_size
                
    def init_model(self):
        """
        Initialize the model.

        Returns
        -------
        None.

        """
        self.model = CINN(in_ch=self.in_ch,
                          img_size=self.img_size,
                          operator=self.op, 
                          conditional_args = {
                              'filter_type': self.filter_type,
                              'frequency_scaling': self.frequency_scaling},
                          optimizer_args = {
                              'lr': self.lr,
                              'weight_decay': self.weight_decay},
                          clamping=self.clamping,
                          weight_mse=self.weight_mse)

    def train(self, dataset):
        """
        The training logic uses Pytorch Lightning.

        Parameters
        ----------
        dataset : LightningDataModule
            Pytorch Lighnting data module with (measurements, gt).
        checkpoint_path : str, optional
            Path to a .ckpt file to continue training. Will be ignored if None.
            The default is None.

        Returns
        -------
        None.

        """
        # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
        if self.torch_manual_seed:
            pl.seed_everything(self.torch_manual_seed)
            
        # create PyTorch datasets
        dataset_train = dataset.create_torch_dataset(
            part='train', reshape=((1,) + dataset.space[0].shape,
                                   (1,) + dataset.space[1].shape))

        dataset_validation = dataset.create_torch_dataset(
            part='validation', reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape))
        
        # create PyTorch dataloaders
        train_loader = DataLoader(dataset_train, batch_size=self.batch_size,
            num_workers=self.num_data_loader_workers, shuffle=True,
            pin_memory=True)
        
        val_loader = DataLoader(dataset_validation, batch_size=self.batch_size,
                num_workers=self.num_data_loader_workers, shuffle=False,
                pin_memory=True)
        
        if not self.model_initialized:
            self.init_model()
            self.model_initialized = True
        
        # train pytorch lightning model
        self.trainer.fit(self.model, train_dataloader=train_loader,
                         val_dataloaders=val_loader)


    def _reconstruct(self, observation, return_torch:bool = False,
                     return_std:bool = False, cut_ouput:bool = True,
                     *args, **kwargs):
        """
        Create a reconstruction from the observation with the cINN
        Reconstructor. 
        
        The batch size must be 1!

        Parameters
        ----------
        observation : numpy array or torch tensor
            Observation data (measurement).
        return_torch : bool, optional
            The method will return an ODL element if False with just the
            image size. If True a torch tensor will be returned. In this case,
            singleton dimensions are NOT removed!
            The default is False.
        return_std : bool, optional
            Also return the standard deviation between the samples used for 
            the reconstruction. This will slow down the reconstruction! Since
            all intermediate results are required, the std is computed on
            the cpu to reduce GPU memory consumption.
            The default is False.
        cut_ouput : bool, optional
            Cut the output to the original size of the ground truth data.
            The default is True.

        Returns
        -------
        xmean : ODL element or torch tensor
            Reconstruction based on the mean of the samples.
        xstd : ODL element or torch tensor, optional
            Standard deviation between the individual samples. Only returned
            if return_std == True

        """
        
        # initialize xmean
        xmean = 0
        
        # create torch tensor if necessary and put in on the same device as
        # the cINN model
        if not torch.is_tensor(observation):
                observation = torch.from_numpy(
                    np.asarray(observation)[None, None]).to(self.model.device)
        
        if observation.shape[0] > 1:
            warn('Batch size greater than 1 is not supported in the' + 
                 'reconstruction process!')
        
        # run the reconstruction process over multiple samples and calculate 
        # mean and standard deviation
        with torch.no_grad():
            xgen_list = []
            
            # Limit the number of max samples for a single run of the network
            # to self.max_samples_per_run
            samples_per_run =  np.arange(0, self.samples_per_reco,
                                         self.max_samples_per_run)
            samples_per_run = list(np.append(
                samples_per_run, self.samples_per_reco)[1:] - samples_per_run)
                                
            for num_samples in samples_per_run:
                # Draw random samples from the random distribution
                z = torch.randn((num_samples,
                                 self.img_size[0]*self.img_size[1]),
                                device=self.model.device)
                    
                obs_rep = torch.repeat_interleave(observation,
                                                  num_samples,
                                                  dim = 0)
                
                xgen = self.model(cinn_input=z, cond_input=obs_rep,
                                  rev=True, cut_ouput=cut_ouput)
                xmean = xmean + torch.sum(xgen, dim=0, keepdim=True)
      
                if return_std:
                    xgen_list.append(xgen.cpu())

            del z, obs_rep, xgen
            
            xmean = xmean / self.samples_per_reco
                    
            if return_std:
                xgen = torch.cat(xgen_list, axis=0)
                xstd = torch.std(xgen, dim=0, keepdim=True).to(self.model.device)
            
            if not return_torch:
                xmean = np.squeeze(xmean.detach().cpu().numpy())
                xmean = self.reco_space.element(xmean)
                if return_std:
                    xstd = np.squeeze(xstd.detach().cpu().numpy())
                    xstd = self.reco_space.element(xstd)
        
            if return_std:
                return xmean, xstd
            else:
                return xmean


    def num_train_params(self):
        params_trainable = list(filter(lambda p: p.requires_grad,
                                        self.model.parameters()))

        print("Number of trainable params: ",
              sum(p.numel() for p in params_trainable))
        
    def load_params(self, path, strict:bool = False, **kwargs):
        """
        Load a model from the given path. To load a model along with its
        weights, biases and module_arguments use a checkpoint.

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.
        strict : bool, optional
            Strict loading of all parameters. Will raise an error if there
            are unknown weights in the file.
            The default is False

        Returns
        -------
        None.

        """
        path_parts = [path, 'default', 'version_0', 'checkpoints']
        path = os.path.join(*path_parts)

        for file in os.listdir(path):
            if file.endswith(".ckpt"):
                self.model = CINN.load_from_checkpoint(
                                os.path.join(path, file), strict=strict,
                                operator=self.op, **kwargs)
        
        self.model.to('cuda')
        self.model_initialized = True
        
        # Update the hyperparams of the reconstructor based on the hyper-
        # params of the model. Hyperparams for the optimizer and training
        # routine are ignored.
        hparams = self.model.hparams
        
        self.img_size = hparams.img_size
        
        # set regular hyperparams
        for attr in self.hyper_params.keys():
            if attr in hparams.keys():
                self.hyper_params[attr] = hparams[attr]
                
            elif attr in hparams.conditional_args:
                self.hyper_params[attr] = hparams.conditional_args[attr]
