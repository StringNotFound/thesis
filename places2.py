import random
import torch
from torchvision.transforms.functional import normalize
from PIL import Image
from glob import glob
import numpy as np
import cv2
import pickle


class Places2(torch.utils.data.Dataset):
    def __init__(self, data_root, size, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        if split == 'train':
            file_handler = open('train_color_paths.obj', 'rb')
            self.color_paths = pickle.load(file_handler)
            file_handler.close()
            #self.color_paths = glob(r'E:\train\**\col\*.png'.format(data_root), recursive=True)
        else:
            file_handler = open('test_color_paths.obj', 'rb')
            self.color_paths = pickle.load(file_handler)
            file_handler.close()
            #self.color_paths = glob(r'E:\test\**\col\*.png'.format(data_root), recursive=True)

        self.mask_paths = glob(r'.\masks\*.png')
        self.N_mask = len(self.mask_paths)
        self.size = size

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        #print(color_path)
        depth_path = color_path.replace('col', 'up_png')
        depth_path = depth_path.replace('_c', '_ud')
        #print(depth_path)
        color_img = Image.open(color_path)
        depth_img = Image.open(depth_path)
        #print(color_img.shape)
        #color_img = self.img_transform(color_img.convert('RGB'))

        ## Deal with the depth image
        depth_img = np.asarray(depth_img)
        if np.max(depth_img) == 0:
            print("Found bad depth image at {}".format(depth_path))
            return self.__getitem__((index + 1) % self.__len__())
        depth_img = depth_img / np.max(depth_img) * 255
        (h, w) = depth_img.shape
        scaling_factor = h / float(self.size)
        depth_img = cv2.resize(depth_img, dsize=(int(w / scaling_factor), int(h / scaling_factor)), interpolation=cv2.INTER_CUBIC)

        # center crop
        w = depth_img.shape[1]
        depth_img = depth_img[:, int(w / 2 - self.size / 2) : int(w / 2 + self.size / 2)]
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
        scaling_factor = h / float(self.size)
        color_img = cv2.resize(color_img, dsize=(int(w / scaling_factor), int(h / scaling_factor)), interpolation=cv2.INTER_CUBIC)

        #print(color_img.shape)
        w = color_img.shape[1]
        color_img = color_img[:, int(w / 2 - self.size / 2) : int(w / 2 + self.size / 2), :]
        #print(color_img.shape)
        color_img = np.swapaxes(color_img, 0, 2)
        color_img = np.swapaxes(color_img, 1, 2)
        #for i in range(3):
            #color_img[i, :, :] = (color_img[i, :, :] - np.mean(color_img[i, :, :])) / np.std(color_img[i, :, :])

        color_img = (color_img - np.mean(color_img))
        if np.std(color_img) != 0:
             color_img = color_img / np.std(color_img)

        color_img = torch.tensor(color_img, dtype=torch.float)
        #color_img = torch.nn.normalize(color_img)

        ## Deal with the mask
        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('L'))
        mask[mask > 0] = 1

        #print(mask.size())

        return depth_img * mask, mask, color_img, depth_img

    def __len__(self):
        return len(self.color_paths)
