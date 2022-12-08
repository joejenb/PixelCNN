from Residual import ResidualStack

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''class MaskedConv(nn.Module):

    def __init__(self, c_in, c_out, mask, **kwargs):
        super().__init__()
        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation*(kernel_size[i]-1)//2 for i in range(2)])

        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)
        self.register_buffer('mask', mask[None,None])

    def forward(self, x):
        self.conv.weight.data *= self.mask # Ensures zero's at masked positions
        return self.conv(x)
class VerticalConv(MaskedConv):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size//2+1:,:] = 0

        if mask_center:
            mask[kernel_size//2,:] = 0

        super().__init__(c_in, c_out, mask, **kwargs)

class HorizontalConv(MaskedConv):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        mask = torch.ones(1,kernel_size)
        mask[0,kernel_size//2+1:] = 0

        if mask_center:
            mask[0,kernel_size//2] = 0

        super().__init__(c_in, c_out, mask, **kwargs)

class GatedMaskedConv(nn.Module):

    def __init__(self, c_in, **kwargs):
        super().__init__()
        self.conv_vert = VerticalConv(c_in, c_out=2*c_in, **kwargs)
        self.conv_horiz = HorizontalConv(c_in, c_out=2*c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(2*c_in, 2*c_in, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out

class PixelCNN(nn.Module):
    def __init__(self, config, device):
        super(PixelCNN, self).__init__()

        self.device = device
        self.num_hiddens = config.num_hiddens
        self.num_channels = config.num_channels
        self.num_categories = config.num_categories
        self.representation_dim = config.representation_dim

        self.conv_vstack = VerticalConv(self.num_channels, self.num_hiddens, mask_center=True)
        self.conv_hstack = HorizontalConv(self.num_channels, self.num_hiddens, mask_center=True)

        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(self.num_hiddens),
            GatedMaskedConv(self.num_hiddens, dilation=2),
            GatedMaskedConv(self.num_hiddens),
            GatedMaskedConv(self.num_hiddens, dilation=4),
            GatedMaskedConv(self.num_hiddens),
            GatedMaskedConv(self.num_hiddens, dilation=2),
            GatedMaskedConv(self.num_hiddens)
        ])

        self.conv_out = nn.Conv2d(self.num_hiddens, self.num_channels * self.num_categories, kernel_size=1, padding=0)
    
    def sample(self):
        x_sample = torch.Tensor(1, 1, self.representation_dim, self.representation_dim).long().to(self.device) - 1
        #x_sample.fill_(0)

        for h in range(self.representation_dim):
            for w in range(self.representation_dim):
                for c in range(self.num_channels):

                    pred = self.forward(x_sample[:,:,:h+1,:])
                    probs = F.softmax(pred[:,:,c,h,w], dim=-1)
                    x_sample[:,c,h,w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)

        
        return x_sample

    def interpolate(self, x, y):
        xy_inter = (x + y) / 2
        xy_inter = self.denoise(xy_inter)        

        return xy_inter

    def denoise(self, x):
        x_new = x

        for h in range(self.representation_dim):
            for w in range(self.representation_dim):
                for c in range(self.num_channels):

                    pred = self.forward(x[:,:,:h+1,:])
                    probs = F.softmax(pred[:,:,c,h,w], dim=-1)
                    x_new[:,c,h,w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)

        return x_new

    def forward(self, x):
        x = x * 1.

        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)

        out = self.conv_out(F.elu(h_stack))

        out = out.reshape(out.shape[0], self.num_categories, self.num_channels, out.shape[-2], self.representation_dim)
        return out
'''
class MaskedConvolution(nn.Module):
    
    def __init__(self, c_in, c_out, mask, **kwargs):
        """
        Implements a convolution with mask applied on its weights.
        Inputs:
            c_in - Number of input channels
            c_out - Number of output channels
            mask - Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                   the convolution should be masked, and 1s otherwise.
            kwargs - Additional arguments for the convolution
        """
        super().__init__()
        # For simplicity: calculate padding automatically 
        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation*(kernel_size[i]-1)//2 for i in range(2)])
        # Actual convolution
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)
        
        # Mask as buffer => it is no parameter but still a tensor of the module 
        # (must be moved with the devices)
        self.register_buffer('mask', mask[None,None])
        
    def forward(self, x):
        self.conv.weight.data *= self.mask # Ensures zero's at masked positions
        return self.conv(x)

class VerticalStackConvolution(MaskedConvolution):
    
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size//2+1:,:] = 0
        
        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size//2,:] = 0
        
        super().__init__(c_in, c_out, mask, **kwargs)
        
class HorizontalStackConvolution(MaskedConvolution):
    
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1,kernel_size)
        mask[0,kernel_size//2+1:] = 0
        
        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0,kernel_size//2] = 0
        
        super().__init__(c_in, c_out, mask, **kwargs)



class GatedMaskedConv(nn.Module):
    
    def __init__(self, c_in, **kwargs):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(2*c_in, 2*c_in, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)
    
    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)
        
        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack
        
        return v_stack_out, h_stack_out

class PixelCNN(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()

        self.device = device
        self.num_hiddens = config.num_hiddens
        self.num_channels = config.num_channels
        self.num_categories = config.num_categories
        self.representation_dim = config.representation_dim

        self.conv_vstack = VerticalStackConvolution(self.num_channels, self.num_hiddens, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(self.num_channels, self.num_hiddens, mask_center=True)

        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(self.num_hiddens),
            GatedMaskedConv(self.num_hiddens, dilation=2),
            GatedMaskedConv(self.num_hiddens),
            GatedMaskedConv(self.num_hiddens, dilation=4),
            GatedMaskedConv(self.num_hiddens),
            GatedMaskedConv(self.num_hiddens, dilation=2),
            GatedMaskedConv(self.num_hiddens)
        ])

        self.conv_out = nn.Conv2d(self.num_hiddens, self.num_channels * self.num_categories, kernel_size=1, padding=0)
   
    def sample(self, img_shape, img=None):
        # Create empty image
        if img is None:
            img = torch.zeros(img_shape, dtype=torch.long).to(self.device) - 1
        # Generation loop
        for h in range(img_shape[2]):
            for w in range(img_shape[3]):
                for c in range(img_shape[1]):
                    # Skip if not to be filled (-1)
                    if (img[:,c,h,w] != -1).all().item():
                        continue
                    # For efficiency, we only have to input the upper part of the image
                    # as all other parts will be skipped by the masked convolutions anyways
                    pred = self.forward(img[:,:,:h+1,:]) 
                    probs = F.softmax(pred[:,:,c,h,w], dim=-1)
                    img[:,c,h,w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        return img
        
    def interpolate(self, x, y):
        xy_inter = (x + y) / 2
        xy_inter = self.denoise(xy_inter)        

        return xy_inter

    def denoise(self, x):
        x_new = x

        for h in range(self.representation_dim):
            for w in range(self.representation_dim):
                for c in range(self.num_channels):

                    pred = self.forward(x[:,:,:h+1,:])
                    probs = F.softmax(pred[:,:,c,h,w], dim=-1)
                    x_new[:,c,h,w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)

        return x_new

    def forward(self, x):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor with integer values between 0 and 255.
        """

        x = (x.float() / (self.num_categories - 1)) * 2 - 1 
        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))
        
        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        out = out.reshape(out.shape[0], 256, out.shape[1]//256, out.shape[2], out.shape[3])
        return out
    