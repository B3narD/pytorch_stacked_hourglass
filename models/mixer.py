import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class MlpMixer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels_dim)
        self.norm2 = nn.LayerNorm(channels_dim)
        self.mlp_tokens = MlpMixer(tokens_dim, hidden_dim=16)
        self.mlp_channels = MlpMixer(channels_dim, hidden_dim=16)

    def forward(self, x):
        # (batch, patch_num)
        y = self.norm1(x).transpose(1, 2)
        y = self.mlp_tokens(y).transpose(1, 2)
        x = x + y
        y = self.norm2(x)
        y = self.mlp_channels(y)
        return x + y

class MlpMixerHourglass(nn.Module):
    def __init__(self, tokens_dim, channels_dim):
        super(MlpMixerHourglass, self).__init__()
        self.up1 = MixerBlock(tokens_dim, channels_dim)
        # Lower branch
        #self.pool1 = nn.MaxPool2d(2, 2)
        self.pool1 = nn.Conv1d(channels_dim, channels_dim, kernel_size=2, stride=2, padding=0)
        self.low1 = MixerBlock(tokens_dim // 2, channels_dim)
        self.low2 = MixerBlock(tokens_dim // 2, channels_dim)
        self.up2 = nn.ConvTranspose1d(channels_dim, channels_dim, kernel_size=2, stride=2, padding=0)
        self.up3 = MixerBlock(tokens_dim, channels_dim)

    def forward(self, x):
        up1  = self.up1(x)
        x = x.transpose(1, 2)
        pool1 = self.pool1(x).transpose(1, 2)
        low1 = self.low1(pool1)
        low2 = self.low2(low1).transpose(1, 2)
        up2 = self.up2(low2).transpose(1, 2)
        up3 = self.up3(up2)
        return up1 + up3

if __name__ == '__main__':
    net = MlpMixerHourglass(196, 768)
    # ind = torch.randn(64, 14, 768)
    #net = MixerBlock(196, 768)
    ind = torch.randn(64, 196, 768)
    output_data = net(ind)
    print(output_data.shape)





