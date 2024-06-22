# Copyright (C) 2023-2024:
#     Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the Apache License as published by
# the Free Software Foundation, either version 2.0 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the Apache License, Version 2.0 for more details.
#
# You should have received a copy of the Apache License, Version 2.0
# along with this program. If not, see https://www.apache.org/licenses/.


import torch
import torch.nn as nn



class Conv(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super(Conv, self).__init__()
        
        """
        A convolution layer, including multiple dilated convolutional operations, a 2D batch normalization, and an activation function
        
        Input: A feature with spatial dimension of H x W with the spectral dimeniosn of in_features

        Output: An extracted features with spatial dimension of H x W with the spectral dimeniosn of out_features


        """


        self.conv_3_di_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1, stride=stride, dilation=1),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(),
        )
        self.conv_3_di_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=2, stride=stride, dilation=2),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_features*2, out_channels=out_features, kernel_size=1, padding=0, stride=stride),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(),
        )

    def forward(self,x):

        x1 = self.conv_3_di_1(x)
        x2 = self.conv_3_di_2(x)
        x = torch.cat((x1,x2),dim=1)
        del x1,x2
        x = self.conv(x)

        return x


class DC4Flood(nn.Module):
    """
    The encoder-decoder struture of DC4Flood

    Input: A tensor of a 3D SAR image with spatial dimensions of H x W, and the polarimetric channels of D

    Output: A tensor of latent features with spatial dimensions of H x W, and the spectral dimension of 2

    """
    def __init__(self, num_features=2):
        super(DC4Flood, self).__init__()
        self.conv1 = Conv(2,32)
        self.conv2 = Conv(32,64)
        self.conv3 = Conv(64,128)
        self.conv4 = Conv(128, 256)
        self.conv5 = Conv(256, 512) 
        self.conv6 = Conv(512, 1024)
        self.conv7 = Conv(1024, num_features)

        self.deconv1 = Conv(num_features,1024)
        self.deconv2 = Conv(1024,512)
        self.deconv3 = Conv(512,256)
        self.deconv4 = Conv(256,128)
        self.deconv5 = Conv(128,64)
        self.deconv6 = Conv(64,32)
        self.deconv7 = Conv(32,2)

    def forward(self,x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        code = x
        
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)

        x = x.reshape((x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3],1))
        return x, code




dc4f = DC4Flood()
print(dc4f)




