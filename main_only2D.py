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
import trainer
import torch.optim as optim
import torch.nn.functional as F
import random
from util.earlystop import EarlyStopping
from module.triplet_selector import HardestNegativeTripletSelector
from module.loss import TripletLoss, OnlineTripletLoss, ExpTripletLoss
from module.dataloader import Loader_2D, Loader_both
from module import transform_
import argparse
from torch.optim.lr_scheduler import StepLR

## Option
args = argparse.ArgumentParser()
args.add_argument('--folder_name', default='new_binary_resize', type=str)
args.add_argument('--batch_size', default=32, type=int)
args.add_argument('--lr', default=1e-4, type=float)
args.add_argument('--momentum', default=0.9, type=float)
args.add_argument('--epoch_num', default=1000, type=int)
args.add_argument('--optimizer', default='adam', type=str)
args.add_argument('--embedding_size', default=256, type=int)
args.add_argument('--gpu', default='5,6', type=str)
args.add_argument('--te_num', default=30, type=int)
args.add_argument('--loss_name', default='exp_triplet', type=str)
arg = args.parse_args()

# Input Option
os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu
folder_name = arg.folder_name
batch_size = arg.batch_size

lr = arg.lr
epoch_num = arg.epoch_num
optimizer_name = arg.optimizer
embedding_size = arg.embedding_size
loss_name = arg.loss_name

num_image_te = arg.te_num # How many test images will be input per class
num_thresh = 1 # How many threshold values for choosing the results of retrieval

# aug = 'random_size'
# aug = 'noise'
aug = 'both'

epoch = 0
best_acc = 0

print('Model : loss - %s, lr - %f, emb size - %d, momentum - %f, optimizer %s'
      %(loss_name, lr, embedding_size, arg.momentum, optimizer_name))

## File Path
path_2d_tr = '/SSD1/furniture/furniture_data/new_train/'
path_2d_te = '/SSD1/furniture/furniture_data/new_test/'
# path_2d_tr = '/data_1/furniture/furniture_data/train/'
# path_2d_te = '/data_1/furniture/furniture_data/test/'

train_furniture = os.listdir(path_2d_tr)
test_furniture = os.listdir(path_2d_te)

train_list = []
test_list = []
for furniture in train_furniture:
    imp_list = glob(os.path.join(path_2d_tr, furniture, '*.png'))
    train_list += [(furniture, imp_image.split('/')[-1][:3], imp_image) for imp_image in imp_list]

for furniture in test_furniture:
    imp_list = glob(os.path.join(path_2d_te, furniture, '*.png'))
    test_list += [(furniture, imp_image.split('/')[-1][:3], imp_image) for imp_image in imp_list]

# Select the small portion of images
def select_num(data_list, num_image):
    furniture_list = [data[0] for data in data_list]
    furniture_set = set(furniture_list)

    selected = []
    for furniture in furniture_set:
        part_list = [data[1] for data in data_list if data[0] == furniture]
        part_set = set(part_list)
        for part in part_set:
            data_imp = [data_ for data_ in data_list if (data_[0] == furniture) and (data_[1] == part)]

            data_ix = np.random.choice(range(len(data_imp)), num_image, replace=False)
            data_imp = np.asarray(data_imp)
            data_selected = data_imp[data_ix].tolist()
            selected += data_selected
    return selected

if num_image_te is not None:
    test_list = select_num(test_list, num_image_te)

random.shuffle(train_list)

## Option
transform = transforms.Compose([
    transforms.Resize([512, 512]),
    transform_.Binary(criterion=150),
    transforms.ToTensor(),
    transform_.MintoMax()
])


transform_aug = transforms.Compose([
    transforms.Resize([512, 512]),
    transform_.RandomResize(max_size=512),
    transform_.AddNoise(aug_path='./aug_data', prob=0.5),
    transform_.Binary(criterion=150),
    transforms.ToTensor(),
    transform_.MintoMax()
])

## Dataset and DataLoader
train_dataset = Loader_2D(train_list, train_furniture, train=True, transform=transform_aug)
test_dataset = Loader_2D(test_list, test_furniture, train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=3)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1)

## Model
# 2D(outline)
model_2d = models.resnet34(pretrained=True)
model_2d.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
if embedding_size != 1000:
    model_2d.fc = nn.Linear(512, embedding_size)

model_2d = nn.DataParallel(model_2d).cuda()

## Tiplet Loss
# criterion
if loss_name == 'triplet':
    criterion = TripletLoss(margin=0.5)
elif loss_name == 'exp_triplet':
    criterion = ExpTripletLoss()

# optimizer
param_list = list(model_2d.parameters())

if optimizer_name == 'adam':
    optimizer = optim.Adam(param_list, lr=lr)

elif optimizer_name == 'SGD':
    optimizer = optim.SGD(param_list, lr=lr, momentum=arg.momentum)

early = EarlyStopping(patience=6)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

## Training and evaluation
os.makedirs('/SSD1/furniture/result/%s/' %folder_name, exist_ok=True)

for epoch in range(epoch_num):
    print('**** Training Start ****')
    trainer.pre_train(model_2d, epoch, criterion, optimizer, scheduler, train_loader, type='only', print_num=400)

    print('**** Testing Start ****')
    model_2d, test_loss = trainer.pre_test(model_2d, epoch, criterion, test_loader, type='only')

    if num_image_te is not None:
        del test_dataset, test_loader, test_list
        test_list = []
        for furniture in test_furniture:
            imp_list = glob(os.path.join(path_2d_te, furniture, '*.png'))
            test_list += [(furniture, imp_image.split('/')[-1][:3], imp_image) for imp_image in imp_list]

        test_list = select_num(test_list, num_image_te)
        test_dataset = Loader_2D(test_list, test_furniture, train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1)

    early(test_loss, model_2d)
    if early.early_stop == True:
        best_model = early.best_model
        best_loss = early.val_loss_min
        torch.save(best_model.module.state_dict(), '/SSD1/furniture/result/%s/emb%d_lr%.6f_%s_%.3f_%s.pt' \
                   %(folder_name, embedding_size, lr, optimizer_name, best_loss, aug))
        break

