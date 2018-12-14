from __future__ import print_function
import os
import glob
import scipy

import tensorflow as tf
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
%matplotlib inline

Label_dict = {
    "apple":1,
    "ball":2,
    "banana":3,
    "bell_pepper":4,
    "binder":5,
    "bowl":6,
    "calculator":7,
    "camera":8,
    "cap":9,
    "cell_phone":10,
    "cereal_box":11,
    "coffee_mug":12,
    "comb":13,
    "dry_battery":14,
    "flashlight":15,
    "food_bag":16,
    "food_box":17,
    "food_can":18,
    "food_cup":19,
    "food_jar":20,
    "garlic":21,
    "glue_stick":22,
    "greens":23,
    "hand_towel":24,
    "instant_noodles":25,
    "keyboard":26,
    "kleenex":27,
    "lemon":28,
    "lightbulb":29,
    "lime":30,
    "marker":31,
    "mushroom":32,
    "notebook":33,
    "onion":34,
    "orange":35,
    "peach":36,
    "pear":37,
    "pitcher":38,
    "plate":39,
    "pliers":40,
    "potato":41,
    "rubber_eraser":42,
    "scissors":43,
    "shampoo":44,
    "soda_can":45,
    "sponge":46,
    "stapler":47,
    "tomato":48,
    "toothbrush":49,
    "toothpaste":50,
    "water_bottle":51
    }

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class Dataset(object):
    def __init__(self, rgb_path, depth_path, num_imgs):
        self.RGB_path = rgb_path
        self.Depth_path = depth_path
        self.num_imgs = num_imgs

    def normalize_np_rgb_image(self, image):
        return (image / 255.0 - 0.5) / 0.5

    def normalize_np_depth_image(self, image):
        return (image / 65535.0 - 0.5) / 0.5

    def get_input_rgb(self, image_path):
        image = np.array(Image.open(image_path)).astype(np.float32)
        return self.normalize_np_rgb_image(image)

    def get_input_depth(self, image_path):
        image = np.array(Image.open(image_path)).astype(np.float32)
        return self.normalize_np_depth_image(image)

    def get_imagelist(self, data_path):
        imgs_path = os.path.join(data_path, '*.png')
        all_namelist = glob.glob(imgs_path, recursive=True)
        imgs_name = [f for f in os.listdir(data_path) if f.endswith('.png')]
        return all_namelist[:self.num_imgs], imgs_name

    def get_label(self, STR):
        str1 = STR.split('_',1)[0]
        str2 = STR.split('_',1)[1]
        if not RepresentsInt(str2)
            str1 = str1 + '_' + str2
        label = Label_dict[str1]
        return label

    def get_nextbatch(self, batch_size):
        assert (batch_size > 0),"Give a valid batch size"
        cur_idx = 0
        image_namelist, image_names = self.get_imagelist(self.RGB_path)
        while cur_idx + batch_size <= self.num_imgs:
            cur_namelist_rgb = image_namelist[cur_idx:cur_idx + batch_size]
            cur_namelist_depth = [os.path.join(Depth_path,depth_image) for depth_image in image_names]
            cur_batch_rgb = [self.get_input_rgb(image_path) for image_path in cur_namelist_rgb]
            cur_batch_rgb = np.array(cur_batch).astype(np.float32)
            cur_batch_depth = [self.get_input_depth(image_path) for image_path in cur_namelist_depth]
            cur_batch_depth = np.array(cur_batch).astype(np.float32)
            cur_idx += batch_size
            labels = [self.get_label() for name in image_names]
            yield cur_batch_rgb, cur_batch_depth, labels

    def get_nextbatch_RGBonly(self, batch_size):
        assert (batch_size > 0),"Give a valid batch size"
        cur_idx = 0
        image_namelist, image_names = self.get_imagelist(self.RGB_path)
        while cur_idx + batch_size <= self.num_imgs:
            cur_namelist_rgb = image_namelist[cur_idx:cur_idx + batch_size]
            cur_batch_rgb = [self.get_input_rgb(image_path) for image_path in cur_namelist_rgb]
            cur_batch_rgb = np.array(cur_batch).astype(np.float32)
            cur_idx += batch_size
            labels = [self.get_label() for name in image_names]
            yield cur_batch_rgb, labels

    def show_image(self, image, normalized=True):
        if not type(image).__module__ == np.__name__:
            image = image.numpy()
        if normalized:
            npimg = (image * 0.5) + 0.5
        npimg.astype(np.uint8)
        plt.imshow(npimg, interpolation='nearest')



# define rgb_path and depth_path and num_imgs
data_loader = Dataset(rgb_path, depth_path, num_imgs)

# data_loader can now be used in the same way as HW4
