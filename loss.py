import torch
import torch.nn as nn
from pytorch_msssim import msssim, ssim


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

def normalize01(t):
    t = t - torch.min(t)
    t = t / torch.max(t)
    return t


class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)# / torch.sum((1-mask))
        loss_dict['valid'] = self.l1(mask * output, mask * gt)# / torch.sum(mask)

        loss_dict['tv'] = total_variation_loss(output_comp)

        #print(torch.max(normalize01(output)))
        #print(torch.min(normalize01(output)))

        msssim_loss = 1 - msssim(normalize01(output), normalize01(gt), normalize=True)
        #print(msssim_loss)
        loss_dict['ms-ssim'] = msssim_loss

        return loss_dict
