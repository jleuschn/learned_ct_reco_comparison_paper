"""
Rotate-and-flip transforms for data augmentation with torch.
"""
import torch

def random_flip_rotate_transform_fbp(sample, dims=(1, 2)):
    fbp, gt = sample
    choice = torch.randint(8, (1,))[0]
    if choice % 4 == 1:
        fbp = torch.flip(fbp, dims[:1])
        gt = torch.flip(gt, dims[:1])
    elif choice % 4 == 2:
        fbp = torch.flip(fbp, dims[1:])
        gt = torch.flip(gt, dims[1:])
    elif choice % 4 == 3:
        fbp = torch.flip(fbp, dims)
        gt = torch.flip(gt, dims)
    if choice // 4 == 1:
        fbp = torch.transpose(fbp, dims[0], dims[1])
        gt = torch.transpose(gt, dims[0], dims[1])
    return fbp, gt

# def random_flip_rotate_transform_obs(sample):
#     """
#     Rotates an observation and ground truth pair.
#     Only works for an even number of angles.
#     """
#     obs, gt = sample
#     choice = torch.randint(8, (1,))[0]
#     if choice == 1:
#         obs = torch.flip(obs, (1,))
#         gt = torch.flip(gt, (1,))
#     elif choice == 2:
#         obs = torch.flip(obs, (1, 2))
#         gt = torch.flip(gt, (2,))
#     elif choice == 3:
#         obs = torch.flip(obs, (2,))
#         gt = torch.flip(gt, (1, 2))
#     elif choice == 4:
#         assert obs.shape[1] % 2 == 0
#         obs = torch.flip(obs, (1, 2))
#         obs0, obs1 = torch.split(obs, obs.shape[1]//2, 1)
#         obs = torch.cat((torch.flip(obs1, (2,)), obs0), 1)
#         gt = torch.transpose(gt, 1, 2)
#     elif choice == 5:
#         assert obs.shape[1] % 2 == 0
#         obs = torch.flip(obs, (2,))
#         obs0, obs1 = torch.split(obs, obs.shape[1]//2, 1)
#         obs = torch.cat((torch.flip(obs1, (2,)), obs0), 1)
#         gt = torch.flip(gt, (1,))
#         gt = torch.transpose(gt, 1, 2)
#     elif choice == 6:
#         assert obs.shape[1] % 2 == 0
#         obs0, obs1 = torch.split(obs, obs.shape[1]//2, 1)
#         obs = torch.cat((torch.flip(obs1, (2,)), obs0), 1)
#         gt = torch.flip(gt, (2,))
#         gt = torch.transpose(gt, 1, 2)
#     elif choice == 7:
#         assert obs.shape[1] % 2 == 0
#         obs = torch.flip(obs, (1,))
#         obs0, obs1 = torch.split(obs, obs.shape[1]//2, 1)
#         obs = torch.cat((torch.flip(obs1, (2,)), obs0), 1)
#         gt = torch.flip(gt, (1, 2))
#         gt = torch.transpose(gt, 1, 2)
#     return obs, gt

# if __name__ == '__main__':
#     from dival import get_standard_dataset, get_reference_reconstructor
#     from dival.measure import PSNR
#     from dival.util.plot import plot_images
#     import numpy as np
#     dataset = get_standard_dataset('lodopab')
#     r = get_reference_reconstructor('fbp', 'lodopab')
#     for i in range(100):
#         obs, gt = dataset.get_sample(0)
#         obs, gt = (torch.from_numpy(np.asarray(obs))[None],
#                     torch.from_numpy(np.asarray(gt))[None])
#         # plot_images([obs[0].cpu().numpy()])
#         obs, gt = random_flip_rotate_transform_obs((obs, gt))
#         # plot_images([obs[0].cpu().numpy()])
#         obs, gt = obs[0].cpu().numpy(), gt[0].cpu().numpy()
#         reco = r.reconstruct(obs)
#         # plot_images([reco, gt])
#         print(PSNR(reco, gt))
