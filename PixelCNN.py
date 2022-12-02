from Residual import ResidualStack

from VectorQuantiser import VectorQuantiserEMA

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCNN(nn.Conv2d):
	def __init__(self, mask_type, *args, **kwargs):
		self.mask_type = mask_type
		assert mask_type in ['A', 'B'], "Unknown Mask Type"
		super(MaskedCNN, self).__init__(*args, **kwargs)
		self.register_buffer('mask', self.weight.data.clone())

		_, depth, height, width = self.weight.size()
		self.mask.fill_(1)
		if mask_type =='A':
			self.mask[:,:,height//2,width//2:] = 0
			self.mask[:,:,height//2+1:,:] = 0
		else:
			self.mask[:,:,height//2,width//2+1:] = 0
			self.mask[:,:,height//2+1:,:] = 0

	def forward(self, x):
		self.weight.data*=self.mask
		return super(MaskedCNN, self).forward(x)


class Gated_Act(nn.Module):
    def __init__(self):
        super(Gated_Act, self).__init__()

    def forward(self, x):
        return torch.tanh(x) * torch.sigmoid(x)


class PixelBlock(nn.Module):
    def __init__(self, mask_type="B", in_channels=128, out_channels=128, kernel_size=7):
        super(PixelBlock, self).__init__()
        self.f_pass = nn.Sequential(
            MaskedCNN(mask_type, in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            Gated_Act()
        )

    def forward(self,x):
        return self.f_pass(x) 


class PixelCNN(nn.Module):
    def __init__(self, config, device):
        super(PixelCNN, self).__init__()

        self._device = device
        self._num_filters = 2 * config.num_filters
        self._num_categories = config.num_categories
        self._representation_dim = config.representation_dim

        self.main = nn.Sequential(
            PixelBlock('A', 1, self._num_filters),
            PixelBlock('B', self._num_filters, self._num_filters),
            PixelBlock('B', self._num_filters, self._num_filters),
            PixelBlock('B', self._num_filters, self._num_filters),
            PixelBlock('B', self._num_filters, self._num_filters),
            PixelBlock('B', self._num_filters, self._num_filters),
            PixelBlock('B', self._num_filters, self._num_filters),
            PixelBlock('B', self._num_filters, self._num_filters),
            nn.Conv2d(self._num_filters, self._num_categories, 1)
        )
    
    def sample(self):
        x_sample = torch.Tensor(1, 1, self._representation_dim, self._representation_dim).to(self._device)
        x_sample.fill_(0)

        for row in range(self._representation_dim):
            for column in range(self._representation_dim):
                logits = self.main(x_sample)
                probabilities = F.softmax(logits[:, :, row, column], dim=-1).data
                x_sample[:, :, row, column] = torch.multinomial(probabilities, 1).int()
        
        return x_sample

    def interpolate(self, x, y):
        xy_inter = (x + y) / 2
        xy_inter = self.denoise(xy_inter)        

        return xy_inter

    def denoise(self, x):
        for row in range(self._representation_dim):
            for column in range(self._representation_dim):
                x[:, :, row, column] = 0
                logits = self.main(x)
                probabilities = F.softmax(logits[:, :, row, column], dim=-1).data
                x[:, :, row, column] = torch.multinomial(probabilities, 1).int()
        return x

    def forward(self, x):
        return self.main(x)
