from glob import glob
from PIL import Image
import numpy as np
import os
import pickle


def valid(path):
    depth_path = path.replace('col', 'up_png')
    depth_path = depth_path.replace('_c', '_ud')
    exists = os.path.isfile(depth_path)
    if not exists:
        return False

    return True


test_color_paths = glob(r'E:\test\**\col\*.png', recursive=True)
test_color_paths = [path for path in test_color_paths if valid(path)]

train_color_paths = glob(r'E:\train\**\col\*.png', recursive=True)
train_color_paths = [path for path in train_color_paths if valid(path)]

print("Found {} valid train files".format(len(train_color_paths)))
print("Found {} valid test files".format(len(test_color_paths)))

train_f = open('train_color_paths.obj', 'wb')
pickle.dump(train_color_paths, train_f)
train_f.close()

test_f = open('test_color_paths.obj', 'wb')
pickle.dump(test_color_paths, test_f)
test_f.close()
