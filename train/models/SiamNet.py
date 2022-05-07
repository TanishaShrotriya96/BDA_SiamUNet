# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import OrderedDict
import torch
import torch.nn as nn
from models.non_local_layer import NONLocalBlock2D


class SiamNet(nn.Module):

    def __init__(self, in_channels=3, out_channels_s=2, out_channels_c=5):
        super(SiamNet, self).__init__()

        features = [1,2,4,8,16,32,64,128]

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.siamups = nn.ModuleList()
        self.siamconvs = nn.ModuleList()

        # UNet layers
        i=0
        for feature in features:
            name = "enc"+str(i)
            self.downs.append(
                SiamNet._block(in_channels, feature, name=name)
                )
            in_channels = feature
            i=i+1

        # Using a single convolution layer along with kernel size = 1
        self.bottleneck = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.nonlocal_ = NONLocalBlock2D(in_channels=8, inter_channels=2,
                                         sub_sample_factor=4, mode='embedded_gaussian')
        # Setting channels for each decoder level
        i=7
        for feature in reversed(features):
            name = "dec"+str(i)
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(SiamNet._block(feature * 2, feature, name=name))
            i=i-1

        self.conv_s = nn.Conv2d(in_channels=1, out_channels=out_channels_s, kernel_size=1)
        
        # Siamese classifier layers

        self.siam_bottleneck = nn.ConvTranspose2d(256,128, kernel_size=2, stride=2)
        self.siam_bottleneck_conv = SiamNet._block(256,256, name="siamBottleNeck")

        features = [4,8,16,32,64,128,256]
        i=7
        for feature in reversed(features):
            self.siamups.append(nn.ConvTranspose2d(feature, int(feature/4), kernel_size=2, stride=2))
            name = "conv"+str(i)
            self.siamups.append( SiamNet._block(int(feature/2), int(feature/2), name=name))
            i=i-1

        self.conv_c = nn.Conv2d(in_channels=2, out_channels=out_channels_c, kernel_size=1)


    def forward(self, x1, x2): 

        skip_connections_pre = []
        x = x1
        for down in self.downs:
            x = down(x)
            skip_connections_pre.append(x)
            x = self.pool(x)

        bottleneck_1 = self.bottleneck(x)
        x = bottleneck_1

        # reverse the skip connections, since we start from bottom layer as we upsample
        skip_connections_pre = skip_connections_pre[::-1]

        # we take a stepsize of 2, because skip connection is from the second conv layer in each encoder.
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # since skip connection only contains one convolution layer for each upsampling decoder unit
            skip_connection_pre = skip_connections_pre[idx//2]
            # increase image size before further upsampling
            concat_skip = torch.cat((skip_connection_pre, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        #final output for x1
        x1 = x

        skip_connections_post = []
        x = x2
        for down in self.downs:
            x = down(x)
            skip_connections_post.append(x)
            x = self.pool(x)

        bottleneck_2 = self.bottleneck(x)
        x = bottleneck_2

        # reverse the skip connections, since we start from bottom layer as we upsample
        skip_connections_post = skip_connections_post[::-1]

        # we take a stepsize of 2, because skip connection is from the second conv layer in each encoder.
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # since skip connection only contains one convolution layer for each upsampling decoder unit
            skip_connection_post = skip_connections_post[idx//2]
            # increase image size before further upsampling
            concat_skip = torch.cat((skip_connection_post, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        # final putput for x2
        x2 = x

        # Siamese
        siambottleneck = bottleneck_2 - bottleneck_1
        siambottleneck = self.siam_bottleneck(siambottleneck)
        skip_connection = skip_connections_post[0] - skip_connections_pre[0]
        x = self.siam_bottleneck_conv(torch.cat((skip_connection,siambottleneck),dim=1))

        # we take a stepsize of 2, because skip connection is from the second conv layer in each encoder.
        for idx in range(0, len(self.siamups), 2):
            x = self.siamups[idx](x)
            # since skip connection only contains one convolution layer for each upsampling decoder unit
            index = (idx//2)+1
            skip_connection = skip_connections_post[index] - skip_connections_pre[index]
            # add attention block
            if(index == 4):
                skip_connection = self.nonlocal_(skip_connection)
            # increase image size before further upsampling
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.siamups[idx+1](concat_skip)

        return self.conv_s(x1), self.conv_s(x2), self.conv_c(x)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )