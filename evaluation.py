import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np

from util.image import unnormalize


def evaluate(model, dataset, device, filename):
    masked_depth, mask, color_img, depth_gt = zip(*[dataset[i * 1000] for i in range(8)])

    # B X 4 X W X H
    color_img = torch.stack(color_img)
    mask = torch.stack(mask)
    masked_depth = torch.stack(masked_depth)
    depth_gt = torch.stack(depth_gt)

    #print("dimensions of input arguments to function 'evaluate':")
    #print(masked_depth.size())
    #print(mask.size())
    #print(color_img.size())
    #print(depth_gt.size())

    rgbd_input = torch.cat((color_img, masked_depth), 1)
    # B X 1 X W X H

    mask_input = mask.repeat(1, 4, 1, 1)

    with torch.no_grad():
        output, _ = model(rgbd_input.to(device), mask_input.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * depth_gt + (1 - mask) * output

    grid = make_grid(
        torch.cat((color_img, mask.repeat(1, 3, 1, 1), output.repeat(1, 3, 1, 1), output_comp.repeat(1, 3, 1, 1), depth_gt.repeat(1, 3, 1, 1)), dim=0),
        normalize=True, scale_each=True)
    save_image(grid, filename)
