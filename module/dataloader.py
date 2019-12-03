import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm

class Loader_2D(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, data_list, furniture_list, transform=None, train=True, real=False):
        self.data_list = data_list
        self.furniture_list = list(set(furniture_list))

        self.train = train
        self.transform = transform
        self.real = real
        random_state = np.random.RandomState(29)

        # Image Dictionary for each furniture
        self.labels, self.labels_set, self.label_to_indices, self.data_dict = dict(), dict(), dict(), dict()
        self.triplets = []
        for data_ in self.data_list:
            if data_[0] not in self.data_dict.keys():
                self.data_dict[data_[0]] = [data_]
            else:
                self.data_dict[data_[0]].append(data_)

        # Make the labels and triplets for evaluation
        for furniture in self.furniture_list:
            self.labels[furniture] = np.asarray([data_[1] for data_ in self.data_dict[furniture]])
            self.labels_set[furniture] = set(self.labels[furniture])
            self.label_to_indices[furniture] = {label: np.where(self.labels[furniture] == label)[0]
                                         for label in self.labels_set[furniture]}

            if self.train == False:
                for dataset in self.data_dict[furniture]:
                    furniture = dataset[0]
                    triplets = [dataset[2], self.data_dict[furniture][random_state.choice(self.label_to_indices[furniture][dataset[1]])][2],
                             self.data_dict[furniture][random_state.choice(self.label_to_indices[furniture][np.random.choice(list(self.labels_set[furniture] - set(dataset[1])))])][2],
                                np.where(np.asarray(list(self.labels_set[furniture])) == dataset[1])[0]]

                    self.triplets.append(triplets)

    def __getitem__(self, index):
        if self.train:
            f_ix, p_ix, path_ix = self.data_list[index]

            positive_index = np.random.choice(self.label_to_indices[f_ix][p_ix], 1)[0]

            negative_label = np.random.choice(list(self.labels_set[f_ix] - set(self.labels[f_ix][positive_index])))
            negative_index = np.random.choice(self.label_to_indices[f_ix][negative_label])

            img1 = path_ix
            img2 = self.data_dict[f_ix][positive_index][2]
            img3 = self.data_dict[f_ix][negative_index][2]
            labels= torch.tensor(-1).long()

        else:
            img1 = self.triplets[index][0]
            img2 = self.triplets[index][1]
            img3 = self.triplets[index][2]
            labels = torch.tensor(self.triplets[index][3]).long()

        img1 = Image.open(img1).convert('L')
        img2 = Image.open(img2).convert('L')
        img3 = Image.open(img3).convert('L')

        if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)

        return img1, img2, img3, labels

    def __len__(self):
        return len(self.data_list)

## Dataset and DataLoader
class Loader_both(Dataset):
    """
       Train: For each sample (anchor) randomly chooses a positive and negative samples
       Test: Creates fixed triplets for testing
    """
    def __init__(self, data_list, cad_folder, furniture_list, transform=None, transform3D=None, train=True):
        self.data_list = data_list
        self.furniture_list = list(set(furniture_list))

        self.train = train
        self.transform = transform
        self.transform3D = transform3D
        random_state = np.random.RandomState(29)

        # Image Dictionary for each furniture
        self.labels, self.labels_set, self.label_to_indices, self.cad_dict = dict(), dict(), dict(), dict()
        self.triplets = []

        for furniture in self.furniture_list:
            part_imp = glob(os.path.join(cad_folder, furniture, '*'))
            part_dict = dict()
            for part in part_imp:
                part_name = part.split('/')[-1]
                img_list = glob(os.path.join(part, '*.png'))
                part_dict[part_name] = self.GatherImg(img_list)
            self.cad_dict[furniture] = part_dict

            # Make the labels and triplets for evaluation
            self.labels[furniture] = np.asarray([data_[1] for data_ in self.data_list if data_[0] == furniture])
            self.labels_set[furniture] = set(self.labels[furniture])

        if self.train == False:
            for dataset in self.data_list:
                furniture = dataset[0]

                triplets = [dataset[2], self.cad_dict[furniture][dataset[1]],
                            self.cad_dict[furniture][np.random.choice(list(self.labels_set[furniture] - set([dataset[1]])), 1)[0]],
                            np.where(np.asarray(list(self.labels_set[furniture])) == dataset[1])[0][0]]

                self.triplets.append(triplets)

    def GatherImg(self, img_list):
        img_gather = []
        for img in img_list:
            img_gather.append(torch.unsqueeze(self.transform3D(Image.open(img).convert('L')), dim=0))
        img_gather = torch.cat(img_gather, dim=0)
        return img_gather

    def __getitem__(self, index):
        if self.train:
            f_ix, p_ix, path_ix = self.data_list[index]

            positive_img_views = self.cad_dict[f_ix][p_ix]

            n_ix = np.random.choice(list(self.labels_set[f_ix] - set([p_ix])), 1)[0]
            negative_img_views = self.cad_dict[f_ix][n_ix]

            img1 = self.transform(Image.open(path_ix).convert('L'))
            img2 = positive_img_views
            img3 = negative_img_views
            labels = torch.tensor(-1).long()

        else:
            img1 = self.transform(Image.open(self.triplets[index][0]).convert('L'))
            img2 = self.triplets[index][1]
            img3 = self.triplets[index][2]
            labels = torch.tensor(self.triplets[index][3]).long()

        return img1, img2, img3, labels

    def __len__(self):
        return len(self.data_list)



















