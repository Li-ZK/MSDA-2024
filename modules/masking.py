import random
import warnings

import kornia
import math
import numpy as np
import torch
from einops import repeat
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.nn import functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def strong_transform(param, data):
    data = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data)
    data = gaussian_blur(blur=param['blur'], data=data)
    return data


def denorm(img, mean, std):
    return img.mul(std).add(mean)


def renorm(img, mean, std):
    return img.sub(mean).div(std)


def color_jitter(color_jitter, mean, std, data, s=.25, p=.2):
    # s is the strength of colorjitter
    if color_jitter > p:
        mean = torch.as_tensor(mean, device=data.device)
        mean = repeat(mean, 'C -> B C 1 1', B=data.shape[0], C=3)
        std = torch.as_tensor(std, device=data.device)
        std = repeat(std, 'C -> B C 1 1', B=data.shape[0], C=3)
        if isinstance(s, dict):
            seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
        else:
            seq = nn.Sequential(
                kornia.augmentation.ColorJitter(
                    brightness=s, contrast=s, saturation=s, hue=s))
        data = denorm(data, mean, std)
        data = seq(data)
        data = renorm(data, mean, std)
    return data


def gaussian_blur(blur, data):
    if blur > 0.5:
        sigma = np.random.uniform(0.15, 1.15)
        kernel_size_y = int(
            np.floor(
                np.ceil(0.1 * data.shape[2]) - 0.5 +
                np.ceil(0.1 * data.shape[2]) % 2))
        kernel_size_x = int(
            np.floor(
                np.ceil(0.1 * data.shape[3]) - 0.5 +
                np.ceil(0.1 * data.shape[3]) % 2))
        kernel_size = (kernel_size_y, kernel_size_x)
        seq = nn.Sequential(
            kornia.filters.GaussianBlur2d(
                kernel_size=kernel_size, sigma=(sigma, sigma)))
        data = seq(data)
    return data


class Masking(nn.Module):
    def __init__(self, block_size, ratio, color_jitter_s, color_jitter_p, blur, mean, std,
                 spectral_block_size, spatial_block_size,
                 spectral_mask, spatial_mask):
        super(Masking, self).__init__()
        # block size
        self.block_size = block_size
        self.spectral_block_size = spectral_block_size
        self.spatial_block_size = spatial_block_size
        self.spectral_mask = spectral_mask
        self.spatial_mask = spatial_mask

        self.ratio = ratio

        self.augmentation_params = None
        if (color_jitter_p > 0 and color_jitter_s > 0) or blur:
            print('[Masking] Use color augmentation.')
            self.augmentation_params = {
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': color_jitter_s,
                'color_jitter_p': color_jitter_p,
                'blur': random.uniform(0, 1) if blur else 0,
                'mean': mean,
                'std': std
            }

    @torch.no_grad()
    def mask_spectral(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        # if self.augmentation_params is not None:
        #     img = strong_transform(self.augmentation_params, data=img.clone())

        mshape = B, round(_ / self.spectral_block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = F.interpolate(input_mask.unsqueeze(1), _).squeeze(1)
        # input_mask = resize(input_mask, size=(H, W))
        # alter mask ok
        # input_mask = input_mask.repeat(1, self.spectral_block_size, H, W)
        masked_img = img * input_mask.unsqueeze(-1).unsqueeze(-1)

        return masked_img

    @torch.no_grad()
    def mask_spatial(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        # if self.augmentation_params is not None:
        #     img = strong_transform(self.augmentation_params, data=img.clone())

        mshape = B, 1, round(H / self.spatial_block_size), round(W / self.spatial_block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        masked_img = img * input_mask

        return masked_img

    @torch.no_grad()
    def mask_co(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        # if self.augmentation_params is not None:
        #     img = strong_transform(self.augmentation_params, data=img.clone())

        spatial_mshape = B, 1, round(H / self.spatial_block_size), round(W / self.spatial_block_size)
        spatial_input_mask = torch.rand(spatial_mshape, device=img.device)
        spatial_input_mask = (spatial_input_mask > self.ratio).float()
        spatial_input_mask = resize(spatial_input_mask, size=(H, W))
        # masked_img = img * input_mask

        # if self.augmentation_params is not None:
        #     img = strong_transform(self.augmentation_params, data=img.clone())

        spectral_mshape = B, round(_ / self.spectral_block_size)
        spectral_input_mask = torch.rand(spectral_mshape, device=img.device)
        spectral_input_mask = (spectral_input_mask > self.ratio).float()
        spectral_input_mask = F.interpolate(spectral_input_mask.unsqueeze(1), _).squeeze(1)

        spatial_input_mask = spatial_input_mask.expand(B, _, H, W)
        spectral_input_mask = spectral_input_mask.unsqueeze(-1).unsqueeze(-1).expand(B, _, H, W)
        input_mask = spectral_input_mask + spatial_input_mask
        input_mask = (input_mask > 0).float()
        # input_mask = resize(input_mask, size=(H, W))
        # alter mask ok
        # input_mask = input_mask.repeat(1, self.spectral_block_size, H, W)
        masked_img = img * input_mask

        return masked_img

    @torch.no_grad()
    def forward(self, img: Tensor, use_spectral=None, use_spatial=None):
        B, C, H, W = img.shape
        assert H % 2 == 1 and W % 2 == 1 and H == W, 'cant set middle point'
        middle = img[:, :, math.floor(H / 2), math.floor(W / 2)].clone().detach()
        # middle *= 0
        if use_spatial is None:
            use_spatial = self.spatial_mask
        if use_spectral is None:
            use_spectral = self.spectral_mask

        img = img.clone()
        if use_spatial and use_spectral:
            img = self.mask_co(img)
        elif use_spectral and not use_spatial:
            img = self.mask_spectral(img)
        elif use_spatial and not use_spectral:
            img = self.mask_spatial(img)
        img[:, :, math.floor(H / 2), math.floor(W / 2)] = middle
        return img

class Masking_new(Masking):
    def __init__(self,**kwargs):
        super(Masking_new, self).__init__(**kwargs)

    @torch.no_grad()
    def mask_co(self, img: Tensor):
        img = img.clone()
        B, C, H, W = img.shape

        # if self.augmentation_params is not None:
        #     img = strong_transform(self.augmentation_params, data=img.clone())
        H_size = math.ceil(H / self.spatial_block_size)
        W_size = math.ceil(W / self.spatial_block_size)
        C_size = math.ceil(C / self.spectral_block_size)
        shape = B, C_size, H_size, W_size

        # spatial_mshape = B, 1, round(H / self.spatial_block_size), round(W / self.spatial_block_size)
        input_mask = torch.rand(shape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = repeat(input_mask, 'B C H W -> B (C c_s) (H h_s) (W w_s)', c_s=self.spectral_block_size,
                            h_s=self.spatial_block_size, w_s=self.spatial_block_size)
        input_mask = input_mask[:B, :C, :H, :W]
        masked_img = img * input_mask

        return masked_img


if __name__ == '__main__':
    x = torch.rand((64, 270, 11, 11))
    m = Masking(64, 1.0, 0.2, 0.2, True, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 4, 4, False, True)
    spectral_x = m(x, use_spectral=True, use_spatial=False)
    spatial_x = m(x, use_spectral=False, use_spatial=True)
    co_x = m(x, use_spectral=True, use_spatial=True)
    print('xx')
