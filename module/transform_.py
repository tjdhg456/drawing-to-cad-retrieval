from PIL import Image
import numpy as np
import torch
from glob import glob
import os
import random

class Padded(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img_array = np.array(img)

        if img_array.shape[0] > img_array.shape[1]:
            max_length = img_array.shape[0]
            imp_length = img_array.shape[1]

            padded = np.zeros([max_length, max_length])
            padded[:,:] = 255

            mid = (max_length - imp_length) // 2
            padded[:, mid:mid+imp_length] = img_array
        else:
            max_length = img_array.shape[1]
            imp_length = img_array.shape[0]

            padded = np.zeros([max_length, max_length])
            padded[:,:] = 255

            mid = (max_length - imp_length) // 2
            padded[mid:mid+imp_length, :] = img_array

        return Image.fromarray(padded)

class Binary(object):
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, img):
        img_arr = np.array(img)
        img_new = np.ones_like(img_arr) * 255.
        img_new[np.where(img_arr < self.criterion)] = 0.
        return Image.fromarray(img_new)

class RandomResize(object):
    def __init__(self, max_size, ratio=0):
        self.rand_size = int(np.random.randint(int(max_size/2), max_size, 1))
        self.ori_size = max_size
        self.ratio = ratio
        self.max_size = max_size

    def __call__(self, img):
        if self.ratio != 0:
            self.rand_size = int(self.max_size / self.ratio)

        new_img = img.resize([self.rand_size, self.rand_size])

        padded = np.ones([self.ori_size, self.ori_size]) * 255.
        mid = (self.ori_size - self.rand_size) // 2

        padded[mid:mid+self.rand_size, mid:mid+self.rand_size] = new_img

        return Image.fromarray(padded)

class MintoMax(object):
    def __init__(self):
        pass
    def __call__(self, img):
        if torch.max(img).item() > 1:
            img /= 255

        return img

class AddNoise(object):
    def __init__(self, aug_path, prob=0.5):
        self.aug_list = glob(os.path.join(aug_path, '*.png'))
        self.aug_list = [list_.replace('\\', '/') for list_ in self.aug_list]
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob, 1)[0] == 1:

            ## Crop the patch from the part image (for augmentation)
            random.shuffle(self.aug_list)
            aug_path = self.aug_list[0]

            aug_location = aug_path.split('/')[-1].split('_')[0]
            aug_img = Image.open(aug_path)

            max_size = np.max([aug_img.size[0], aug_img.size[1]])
            if max_size < 200:
                scale = 1
            elif max_size < 400:
                scale = 1.5
            else:
                scale = 2

            aug_img = aug_img.resize([int(aug_img.size[1]/scale), int(aug_img.size[0]/scale)])
            mask = np.array(aug_img)[:,:,3]
            aug_y_len, aug_x_len = mask.shape[0], mask.shape[1]

            ## BBox of original img
            column, row = np.where(np.array(img) != 255)
            x_min, x_max, y_min, y_max = np.min(row), np.max(row), np.min(column), np.max(column)

            ## Mask x Original Image
            img = np.array(img)
            if 'L' in aug_location:
                x_min = 50
                x_max = 120
            elif 'R' in aug_location:
                x_min = 300
                x_max = 500

            ix = 0
            while True:
                if ix > 5:
                    return Image.fromarray(img)
                else:
                    j = np.random.choice(range(x_min, x_max), 1)[0]
                    ix += 1
                    if (j + aug_x_len) < 510:
                        break
            ix = 0
            while True:
                if ix >5:
                    return Image.fromarray(img)
                else:
                    i = np.random.choice(range(y_min, y_max), 1)[0]
                    ix += 1
                    if (i + aug_y_len) < 510:
                        break

            ori_patch = img[i:(i+aug_y_len), j:(j+aug_x_len)]

            mask = np.array(aug_img)[:,:,3]
            masked = np.array(aug_img)[:,:,0] * (1-mask)

            index_mask = np.where(mask != 0)
            try:
                ori_patch[index_mask] = masked[index_mask]
                img[i:(i + aug_y_len), j:(j + aug_x_len)] = ori_patch
                return Image.fromarray(img)
            except:
                return Image.fromarray(img)

        else:
            return img