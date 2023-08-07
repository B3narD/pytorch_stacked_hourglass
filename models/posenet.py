import numpy as np
import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool, Residual
from models.mixer import MlpMixerHourglass, MlpMixer, MixerBlock
from task.loss import HeatmapLoss

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        
        self.nstack = nstack
        # input: (batchSize, channels, H, W)
        # output: (batchSize, inputDim, H/4, W/4)
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our posenet
        x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, E)
        return x

class PatchToHeatmap(nn.Module):
    def __init__(self, channels_dim, num_keypoints, heatmap_size):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size[0]
        self.linear = nn.Linear(channels_dim, num_keypoints * heatmap_size[0] * heatmap_size[1])

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], x.shape[1], self.num_keypoints, self.heatmap_size, self.heatmap_size)
        return x

class PatchToGlobal(nn.Module):
    def __init__(self, scale_factor, num_keypoints):
        super().__init__()
        self.scale_factor = scale_factor
        # Initialize the transposed convolution layer
        self.conv_transpose = nn.ConvTranspose2d(in_channels=num_keypoints, out_channels=num_keypoints, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, patch_heatmaps):
        # patch_heatmaps: tensor of shape (batchsize, num_patches, num_keypoints, heatmap_height, heatmap_width)
        # transpose the dimensions to: (batchsize, num_keypoints, heatmap_height, heatmap_width, num_patches)
        patch_heatmaps = patch_heatmaps.permute(0, 2, 3, 4, 1)
        global_heatmaps = []
        for i in range(patch_heatmaps.shape[-1]):
            global_heatmaps.append(self.conv_transpose(patch_heatmaps[..., i]))
        global_heatmap = torch.stack(global_heatmaps, dim=-1)
        global_heatmap = torch.sum(global_heatmap, dim=-1)  # Sum over the patches dimension
        return global_heatmap



class MLPPoseNet(nn.Module):
    def __init__(self, nstack, channels_dim, num_patches=196, heatmap_size=(8, 8), image_size=224):
        super(MLPPoseNet, self).__init__()

        self.nstack = nstack
        patch_size = int(np.sqrt(image_size * image_size / num_patches))
        self.pre = PatchEmbed(img_size=image_size, patch_size=patch_size, embed_dim=channels_dim)
        # (batchSize, patch, channels)
        self.hgs = nn.ModuleList([
            nn.Sequential(
                MlpMixerHourglass(num_patches, channels_dim),  # Changed to MlpMixerHourglass
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            MixerBlock(num_patches, channels_dim)
            for i in range(nstack)])

        self.outs = nn.ModuleList([nn.Sequential(PatchToHeatmap(channels_dim, 16, heatmap_size), PatchToGlobal(, 16)) for _ in range(nstack)])

        # self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList(
            [MixerBlock(num_patches, channels_dim) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList(
            [nn.Sequential(Conv(16, 16), PatchEmbed(heatmap_size[0]*heatmap_size[1], patch_size, channels_dim)) for i in range(nstack - 1)])
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our posenet
        # x of size 1,3,inpdim,inpdim
        #print(imgs.shape)
        x = self.pre(imgs)
        print('pre', x.shape)
        # (batchSize, patch, channels) -> (64, 196, 768)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            print('h',i,':',hg.shape)
            feature = self.features[i](hg)
            print(feature.shape)
            preds = self.outs[i](feature)
            print("preds:", preds.shape)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                mp = self.merge_preds[i](preds)
                print(mp.shape)
                mf = self.merge_features[i](feature)
                print(mf.shape)
                x = x + mp
                x = x + mf
        return torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:, i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss


if __name__ == '__main__':
    # net1 = PatchEmbed()
    #net2 = PatchToHeatmap(patch_dim=768, num_patches=196, heatmap_size=(8, 8))
    device = torch.device("cuda")
    net = MLPPoseNet(2, 768).to(device)
    #net = MLPPoseNet(6, 224, )
    inp = torch.randn(32, 3, 224, 224).to(device)
    #out = net2(inp2)
    out2 = net(inp)
    #out2 = net2(out)
    print(out2.shape)