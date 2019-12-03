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
from module.model import MVCNN
import argparse
from torch.optim.lr_scheduler import StepLR

## Option
args = argparse.ArgumentParser()
args.add_argument('--folder_name', default='2D_3D', type=str)
args.add_argument('--model_path', type=str, default='/SSD1/furniture/result/only2D/emb128_lr0.010000_adam_0.262.pt')
args.add_argument('--embedding_size', type=int, default=128)
args.add_argument('--batch_size', default=4, type=int)
args.add_argument('--lr', default=1e-4, type=float)
args.add_argument('--momentum', default=0.9, type=float)
args.add_argument('--epoch_num', default=1000, type=int)
args.add_argument('--optimizer', default='adam', type=str)
args.add_argument('--gpu', default='0,1', type=str)
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

epoch = 0
best_acc = 0

print('Model : loss - %s, lr - %f, emb size - %d, momentum - %f, optimizer %s'
      %(loss_name, lr, embedding_size, arg.momentum, optimizer_name))

## File Path
path_2d_tr = '/SSD1/furniture/furniture_data/train/'
path_2d_te = '/SSD1/furniture/furniture_data/test/'
cad_base_folder = '/SSD1/furniture/furniture_data/obj_depth'

train_furniture = os.listdir(path_2d_tr)[:2]
test_furniture = os.listdir(path_2d_te)[:2]

train_list = []
test_list = []
for furniture in train_furniture:
    imp_list = glob(os.path.join(path_2d_tr, furniture, 'surface/*.png'))
    train_list += [(furniture, imp_image.split('/')[-1][:3], imp_image) for imp_image in imp_list]

for furniture in test_furniture:
    imp_list = glob(os.path.join(path_2d_te, furniture, 'surface/*.png'))
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
    transforms.CenterCrop([1024,1024]),
    transforms.Resize([512, 512]),
    transforms.ToTensor(), ])

transform3D = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.ToTensor(),])

## Dataset and DataLoader
train_dataset = Loader_both(train_list, cad_base_folder, train_furniture, transform=transform, transform3D=transform3D, train=True)
test_dataset = Loader_both(test_list, cad_base_folder, test_furniture, transform=transform, transform3D=transform3D, train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)

## Model
model = MVCNN(arg.embedding_size, arg.model_path, infer=False)
model = nn.DataParallel(model).cuda()

## Tiplet Loss
# criterion
if loss_name == 'triplet':
    criterion = TripletLoss(margin=0.5)
elif loss_name == 'exp_triplet':
    criterion = ExpTripletLoss()

# optimizer
param_list = list(model.module.model_2D.parameters()) + list(model.module.model_3D_1.parameters()) + list(model.module.model_3D_2.parameters()) + list(model.module.model_3D_fc.parameters())
if optimizer_name == 'adam':
    optimizer = optim.Adam(param_list, lr=lr)

elif optimizer_name == 'SGD':
    optimizer = optim.SGD(param_list, lr=lr, momentum=arg.momentum)

early = EarlyStopping(patience=7)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

## Training and evaluation
os.makedirs('/SSD1/furniture/result/%s/' %folder_name, exist_ok=True)

for epoch in range(epoch_num):
    print('**** Training Start ****')
    model = trainer.pre_train(model, epoch, criterion, optimizer, scheduler, train_loader, type='both', print_num=100)

    print('**** Testing Start ****')
    model, test_loss = trainer.pre_test(model, epoch, criterion, test_loader, type='both')

    if num_image_te is not None:
        del test_dataset, test_loader, test_list
        test_list = []
        for furniture in test_furniture:
            imp_list = glob(os.path.join(path_2d_te, furniture, 'surface/*.png'))
            test_list += [(furniture, imp_image.split('/')[-1][:3], imp_image) for imp_image in imp_list]

        test_list = select_num(test_list, num_image_te)
        test_dataset = Loader_both(test_list, cad_base_folder, test_furniture, transform=transform,
                                   transform3D=transform3D, train=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)

    early(test_loss, model)
    if early.early_stop == True:
        best_model = early.best_model
        best_loss = early.val_loss_min
        torch.save(best_model.module.state_dict(), '/SSD1/furniture/result/%s/emb%d_lr%.6f_%s_%.3f.pt' \
                   %(folder_name, embedding_size, lr, optimizer_name, best_loss))
        break

