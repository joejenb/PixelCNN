from Residual import ResidualStack

from VectorQuantiser import VectorQuantiserEMA

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv(nn.Module):

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

        self._device = device
        self._num_hiddens = config.num_hiddens
        self._num_channels = config.num_channels
        self._num_categories = config.num_categories
        self._representation_dim = config.representation_dim

        self.conv_vstack = VerticalConv(self._num_channels, self._num_hiddens, mask_center=True)
        self.conv_hstack = HorizontalConv(self._num_channels, self._num_hiddens, mask_center=True)

        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(self._num_hiddens),
            GatedMaskedConv(self._num_hiddens, dilation=2),
            GatedMaskedConv(self._num_hiddens),
            GatedMaskedConv(self._num_hiddens, dilation=4),
            GatedMaskedConv(self._num_hiddens),
            GatedMaskedConv(self._num_hiddens, dilation=2),
            GatedMaskedConv(self._num_hiddens)
        ])

        self.conv_out = nn.Conv2d(self._num_hiddens, self._num_channels * self._num_categories, kernel_size=1, padding=0)
    
    def sample(self):
        x_sample = torch.Tensor(1, 1, self._representation_dim, self._representation_dim).to(self._device)
        x_sample.fill_(0)

        for h in range(self._representation_dim):
            for w in range(self._representation_dim):
                for c in range(self._num_channels):

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

        for h in range(self._representation_dim):
            for w in range(self._representation_dim):
                for c in range(self._num_channels):

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

        out = out.reshape(out.shape[0], self._num_categories, self._num_channels, self._representation_dim, self._representation_dim)
        return out
