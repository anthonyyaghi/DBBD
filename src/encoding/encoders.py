from torch import nn
from torchsparse import SparseTensor
from torchsparse.backbones import unet


class SparseResUNet(nn.Module):

    def __init__(self, in_channels: int):
        super(SparseResUNet, self).__init__()
        self.encoder = unet.SparseResUNet42(in_channels=in_channels)

    def forward(self, tensor: SparseTensor) -> SparseTensor:
        return self.encoder(tensor)
