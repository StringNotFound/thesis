import torch
import torch.nn as nn
from pytorch_msssim import msssim, ssim


def L1_Charbonnier_loss(X, Y):
    """L1 Charbonnierloss."""
    eps = 1e-6

    diff = torch.add(X, -Y)
    diff = diff.pow(2.0) + eps
    error = torch.sqrt(diff)
    loss = torch.mean(error)
    return loss

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

        #with torch.no_grad():
            #sobelConv = nn.Conv2d(1, 2, 3, bias=False)
            #print(sobelConv.weight.data.size())
            #sobelFilter = torch.tensor([[[-1, 0, 1],
                                         #[-2, 0, 2],
                                         #[-1, 0, 1]],
                                        #[[-1, -2, -1],
                                         #[0, 0, 0],
                                         #[1, 2, 1]]], dtype=torch.float).to(torch.device('cuda'))
            #sobelFilter = torch.unsqueeze(sobelFilter, 1)
            #print(sobelFilter.size())
            #sobelConv.weight.data = sobelFilter
            #output_edges = sobelConv(output)
            #gt_edges = sobelConv(gt)

        #loss_dict['edges'] = self.l1(output_edges, gt_edges)

        #loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)# / torch.sum((1-mask))
        #loss_dict['valid'] = self.l1(mask * output, mask * gt)# / torch.sum(mask)

        loss_dict['l1'] = self.l1(output, gt)

        loss_dict['tv'] = total_variation_loss(output_comp)

        msssim_loss = 1 - msssim(normalize01(output), normalize01(gt), normalize=True)
        #print(msssim_loss)
        loss_dict['ms-ssim'] = msssim_loss

        loss_dict['charbonnier'] = L1_Charbonnier_loss(output, gt)

        return loss_dict
