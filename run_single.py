import argparse
import numpy as np
import torch

import opt
from evaluation import evaluate
from loss import DepthLoss
from net import PConvUNet
from util.io import load_ckpt
from util.io import save_ckpt
from PIL import Image
import torchvision.transforms.functional as F
import cv2


def loadTensors(color_path, depth_path, mask_path):
    size = 256

    color_img = Image.open(color_path)
    depth_img = Image.open(depth_path)
    mask = Image.open(mask_path)

    ## Deal with the depth image
    depth_img = np.asarray(depth_img)
    depth_img = depth_img / np.max(depth_img) * 255
    (h, w) = depth_img.shape
    scaling_factor = h / float(size)
    depth_img = cv2.resize(depth_img, dsize=(int(w / scaling_factor), int(h / scaling_factor)), interpolation=cv2.INTER_CUBIC)

    # center crop
    w = depth_img.shape[1]
    depth_img = depth_img[:, int(w / 2 - size / 2) : int(w / 2 + size / 2)]
    depth_img = depth_img[None, :, :]

    # normalize
    depth_img = (depth_img - np.mean(depth_img))
    if np.std(depth_img) != 0:
        depth_img = depth_img / np.std(depth_img)

    # cast to a tensor
    depth_img = torch.tensor(depth_img, dtype=torch.float)

    ## Deal with the color image
    color_img = np.asarray(color_img)
    (h, w, c) = color_img.shape
    scaling_factor = h / float(size)
    color_img = cv2.resize(color_img, dsize=(int(w / scaling_factor), int(h / scaling_factor)), interpolation=cv2.INTER_CUBIC)

    w = color_img.shape[1]
    color_img = color_img[:, int(w / 2 - size / 2) : int(w / 2 + size / 2), :]
    color_img = np.swapaxes(color_img, 0, 2)
    color_img = np.swapaxes(color_img, 1, 2)

    color_img = (color_img - np.mean(color_img))
    if np.std(color_img) != 0:
        color_img = color_img / np.std(color_img)

    color_img = torch.tensor(color_img, dtype=torch.float)

    ## Deal with the mask
    #mask_transform = transforms.Compose([transforms.Resize(size=(size, size)), transforms.ToTensor()])
    #mask = mask_transform(mask.convert('L'))
    mask = mask.convert('L')
    mask = F.resize(mask, size=(256, 256))
    mask = F.to_tensor(mask)
    mask[mask > 0] = 1
    # Black means masked out. For this mask dataset, we need to invert the masks
    mask = 1 - mask

    return depth_img * mask, mask, color_img, depth_img


def main():
    parser = argparse.ArgumentParser()
    # training options
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, default='./output.png')
    parser.add_argument('--output_gt', type=str, default='./output_gt.png')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--depth_image', type=str, required=True)
    parser.add_argument('--mask', type=str, required=True)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device('cpu')

    model = PConvUNet().to(device)
    model.eval()

    criterion = DepthLoss().to(device)
    load_ckpt(args.model, [('model', model)])

    masked_depth, mask, color_img, depth_gt = loadTensors(args.image, args.depth_image, args.mask)

    masked_depth = torch.unsqueeze(masked_depth, 0)
    mask = torch.unsqueeze(mask, 0)
    color_img = torch.unsqueeze(color_img, 0)
    depth_gt = torch.unsqueeze(depth_gt, 0)

    #rgbd_masked = torch.cat((color_img, masked_depth), 1)
    with torch.no_grad():
        output, _ = model(color_img, masked_depth, mask.repeat(1, 4, 1, 1))
    loss_dict = criterion(masked_depth, mask, output, depth_gt)

    loss = 0.0
    #loss_params = {'hole': 1, 'valid': 1, 'tv': 0, 'ms-ssim': 1}
    loss_params = {'l1': 0, 'tv': 0, 'ms-ssim': 0, 'charbonnier': 1}
    for key, coef in loss_params.items():
        value = coef * loss_dict[key]
        loss += value

    print("Loss: {}".format(loss))

    np_img = output.numpy()
    np_img = np_img - np.min(np_img)
    np_img = np_img / np.max(np_img) * 255
    np_img = np_img.astype('uint8')
    np_img = np.squeeze(np_img)
    output_img = Image.fromarray(np_img)
    output_img.save(args.output)

    np_img = depth_gt.numpy()
    np_img = np_img - np.min(np_img)
    np_img = np_img / np.max(np_img) * 255
    np_img = np_img.astype('uint8')
    np_img = np.squeeze(np_img)
    output_img = Image.fromarray(np_img)
    output_img.save(args.output_gt)


if __name__ == '__main__':
    main()
