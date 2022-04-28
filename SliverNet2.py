import torch
from FeatureCNN2 import FeatureCNN2
from nonadaptiveconcatpool2d import nonadaptiveconcatpool2d


class SliverNet2(torch.nn.Module):
    def __init__(self, backbone=None, n_out=2, ncov=0, add_layers=False):
        super().__init__()
        self.backbone = backbone
        # get_backbone(model_name, n_feats)  # change to load_backbone
        # self.model = torch.nn.Sequential(self.model, NonAdaptiveConcatPool2d(8))
        self.cov_model = FeatureCNN2(ncov=ncov, n_out=n_out, add_layers=add_layers)

    def forward(self,x,cov=None):
        # B x C x (n_slices x orig_W) x orig_W
        x = self.backbone(x) # get the feature maps
        # B x C x (n_slices x W) x W
        kernel_size = x.shape[-1]  # W
        x = nonadaptiveconcatpool2d(x, kernel_size) # pool the feature maps with kernel and stride W
        # B x C x n_slices x 1
        x = x.squeeze(dim=-1)
        # B x C x n_slices
        return self.cov_model(x,cov)  # 1d cnn, etc

    def feature_maps(self, x):
        # generate heatmaps on each image in the batch
        # make sure model is in eval mode
        # to visualize image i out of b
        # b, c, h,w = x.shape
        # ax.imshow(img[i][1],cmap='gray',alpha=1)
        # ax.imshow(hm[i], alpha=0.3, extent=(0,w,h,0),
        #               interpolation='bilinear', cmap='magma')

        # x: B x C x (n_slices x orig_W) x orig_W
        hm = self.backbone(x)
        # hm: B x C x (n_slices x W) x W
        return torch.mean(hm, 1)

    def max_slices(self, x, kernel_size=3):
        # in progress.
        # eventually we will find the slice(s) that give the highest signal
        # given kernel size k and number of layers 2
        # output i is effected by slices: i: i+2(k-1)
        # so, to be conservative we can mark slices [i-2, i+2(k-1)]
        # for k=3 this will provide 7 slice [i-2,i+4]
        x = self.backbone(x)
        kernel_size = x.shape[-1]  # W
        x = nonadaptiveconcatpool2d(x, kernel_size) # pool the feature maps with kernel and stride W
        # B x C x n_slices x 1
        x = x.squeeze(dim=-1)
        idx = self.cov_model.max_slice(x)
        idx = idx.view(-1, self.cov_model._conv_filters)

        return idx
