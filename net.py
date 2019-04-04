import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class RGBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.input_conv.apply(weights_init('kaiming'))

    def forward(self, input):
        return self.input_conv(input)


class RGBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = RGBConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = RGBConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = RGBConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = RGBConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        h = self.conv(input)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))
        self.in_channels = in_channels

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        #print("mask size: ", mask.size())
        mask = mask[:, 1:2, :, :]
        #print("mask size: ", mask.size())
        mask = mask.repeat(1, self.in_channels, 1, 1)
        #print("mask size: ", mask.size())
        #print("input size: ", input.size())
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class PConvUNet(nn.Module):
    def __init__(self, layer_size=6, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size

        self.d_enc_1 = PCBActiv(4, 64, bn=False, sample='down-7')
        self.rgb_enc_1 = RGBActiv(3, 64, bn=False, sample='down-7')
        self.d_enc_2 = PCBActiv(128, 128, sample='down-5')
        self.rgb_enc_2 = RGBActiv(64, 128, bn=False, sample='down-5')
        self.d_enc_3 = PCBActiv(256, 256, sample='down-5')
        self.rgb_enc_3 = RGBActiv(128, 256, bn=False, sample='down-5')
        self.d_enc_4 = PCBActiv(512, 256, sample='down-3')
        self.rgb_enc_4 = RGBActiv(256, 256, bn=False, sample='down-3')

        self.d_enc_5 = PCBActiv(512, 256, sample='down-3')
        self.rgb_enc_5 = RGBActiv(256, 256, bn=False, sample='down-3')

        self.d_enc_6 = PCBActiv(512, 256, sample='down-3')
        self.rgb_enc_6 = RGBActiv(256, 256, bn=False, sample='down-3')
        # takes in [rgb_enc_6, d_enc_6]
        self.d_dec_6 = PCBActiv(256 + 256, 256, activ='leaky')
        # takes in [rgb_enc_6]
        self.rgb_dec_6 = RGBActiv(256, 256, activ='leaky')

        # takes in [rgb_enc_5, d_enc_5, rgb_dec_6, d_dec_6]
        self.d_dec_5 = PCBActiv(256 + 256 + 256 + 256, 256, activ='leaky')
        # takes in [rgb_enc_5, rgb_dec_6]
        self.rgb_dec_5 = RGBActiv(256 + 256, 256, activ='leaky')

        # takes in [rgb_enc_4, d_enc_4, rgb_dec_5, d_dec_5]
        self.d_dec_4 = PCBActiv(256 + 256 + 256 + 256, 256, activ='leaky')
        self.rgb_dec_4 = RGBActiv(256 + 256, 256, activ='leaky')

        # takes in [rgb_enc_3, d_enc_3, rgb_dec_4, d_dec_4]
        self.d_dec_3 = PCBActiv(256 + 256 + 256 + 256, 256, activ='leaky')
        self.rgb_dec_3 = RGBActiv(256 + 256, 256, activ='leaky')

        # takes in [rgb_enc_2, d_enc_2, rgb_dec_3, d_dec_3]
        self.d_dec_2 = PCBActiv(128 + 128 + 256 + 256, 128, activ='leaky')
        self.rgb_dec_2 = RGBActiv(128 + 256, 128, activ='leaky')

        # takes in [rgb_enc_1, d_enc_1, rgb_dec_2, d_dec_2]
        self.d_dec_1 = PCBActiv(64 + 64 + 128 + 128, 64, bn=False, activ=None, conv_bias=True)
        # technically just a dummy set of weights
        self.rgb_dec_1 = RGBActiv(64 + 128, 64, bn=False, activ=None, conv_bias=True)

        # takes in [rgb, masked_depth, rgb_dec_1, d_dec_1]
        self.d_dec_0 = PCBActiv(64 + 64 + 4, 1, bn=False, activ=None, conv_bias=True)

    def forward(self, rgb, masked_depth, input_mask):
        d_dict = {}  # for the output of the depth layers
        rgb_dict = {}  # for the output of the RGB layers
        mask_dict = {}  # for the mask outputs of the depth layers

        rgb_dict['e_0'], d_dict['e_0'], mask_dict['e_0'] = rgb, masked_depth, input_mask

        #print(rgb.size())
        #print(masked_depth.size())

        enc_key_prev = 'e_0'
        for i in range(1, self.layer_size + 1):
            enc_key = 'e_{:d}'.format(i)
            # first, run it through the rgb convolutional layer
            l_key = 'rgb_enc_{:d}'.format(i)

            #print("Giving layer {} input of size {}".format(l_key, rgb_dict[enc_key_prev].size()))
            rgb_dict[enc_key] = getattr(self, l_key)(rgb_dict[enc_key_prev])

            l_key = 'd_enc_{:d}'.format(i)
            #print("Giving layer {} input of size {}".format(l_key, torch.cat((d_dict[enc_key_prev], rgb_dict[enc_key_prev]), 1).size()))
            d_dict[enc_key], mask_dict[enc_key] = getattr(self, l_key)(
                torch.cat((d_dict[enc_key_prev], rgb_dict[enc_key_prev]), 1),
                mask_dict[enc_key_prev])
            enc_key_prev = enc_key

        enc_key = 'e_{:d}'.format(self.layer_size)
        h_rgb = getattr(self, 'rgb_dec_{:d}'.format(self.layer_size))(
            rgb_dict[enc_key]
        )
        h_depth, h_mask = getattr(self, 'd_dec_{:d}'.format(self.layer_size))(
            torch.cat((d_dict[enc_key], rgb_dict[enc_key]), 1),
            mask_dict[enc_key]
        )

        for i in range(self.layer_size - 1, 0, -1):
            enc_key = 'e_{:d}'.format(i)

            h_rgb = F.interpolate(h_rgb, scale_factor=2, mode=self.upsampling_mode)
            h_depth = F.interpolate(h_depth, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')

            l_key = 'd_dec_{:d}'.format(i)
            #print("Giving layer {} input of size {}".format(l_key, torch.cat((rgb_dict[enc_key], h_rgb, d_dict[enc_key], h_depth), 1).size()))
            h_depth, h_mask = getattr(self, l_key)(
                torch.cat((rgb_dict[enc_key],
                           h_rgb,
                           d_dict[enc_key],
                           h_depth), 1),
                torch.cat((h_mask, mask_dict[enc_key]), 1))

            l_key = 'rgb_dec_{:d}'.format(i)
            #print("Giving layer {} input of size {}".format(l_key, torch.cat((rgb_dict[enc_key], h_rgb), 1).size()))
            h_rgb = getattr(self, l_key)(
                torch.cat((rgb_dict[enc_key], h_rgb), 1))

        h_rgb = F.interpolate(h_rgb, scale_factor=2, mode=self.upsampling_mode)
        h_depth = F.interpolate(h_depth, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')

        h_depth, h_mask = self.d_dec_0(
            torch.cat((rgb,
                       h_rgb,
                       masked_depth,
                       h_depth), 1),
            h_mask
        )

        #print("done")
        return h_depth, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


if __name__ == '__main__':
    size = (1, 3, 5, 5)
    input = torch.ones(size)
    input_mask = torch.ones(size)
    input_mask[:, :, 2:, :][:, :, :, 2:] = 0

    conv = PartialConv(3, 3, 3, 1, 1)
    l1 = nn.L1Loss()
    input.requires_grad = True

    output, output_mask = conv(input, input_mask)
    loss = l1(output, torch.randn(1, 3, 5, 5))
    loss.backward()

    assert (torch.sum(input.grad != input.grad).item() == 0)
    assert (torch.sum(torch.isnan(conv.input_conv.weight.grad)).item() == 0)
    assert (torch.sum(torch.isnan(conv.input_conv.bias.grad)).item() == 0)

    # model = PConvUNet()
    # output, output_mask = model(input, input_mask)
