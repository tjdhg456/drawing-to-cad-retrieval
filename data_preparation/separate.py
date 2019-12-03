import os
from glob import glob
import random
import numpy as np

def delete_png():
    base_folder = '/SSD1/furniture/furniture_data/obj_imp'
    with open('./criterion.txt', 'r+') as f:
        criterion = f.readlines()

    for cri in criterion:
        if len(cri.split(',')) == 1:
            folder_remove = os.path.join(base_folder, cri.strip())
            os.system('rm -r -rf %s' %folder_remove)

        else:
            folder = cri.split(',')[0].strip()
            files = cri.split(',')[1:]
            for file in files:
                file = int(file.strip())
                file_remove = os.path.join(base_folder, folder, 'outline', '%03d_*.png' %file)
                os.system('rm -rf %s' %file_remove)

    folder_25 = os.path.join(base_folder, '25_*')
    os.system('rm -r -rf %s' %folder_25)

def delete_obj():
    base_folder = '/SSD1/furniture/furniture_data/obj_imp'
    with open('./criterion.txt', 'r+') as f:
        criterion = f.readlines()

    for cri in criterion:
        if len(cri.split(',')) == 1:
            folder_remove = os.path.join(base_folder, cri.strip())
            os.system('rm -r -rf %s' %folder_remove)

        else:
            folder = cri.split(',')[0].strip()
            files = cri.split(',')[1:]
            for file in files:
                file = int(file.strip())
                file_remove = os.path.join(base_folder, folder, '%03d.obj' %file)
                os.system('rm -rf %s' %file_remove)

    folder_25 = os.path.join(base_folder, '25_*')
    os.system('rm -r -rf %s' %folder_25)

def separate():
    base_folder = '/SSD1/furniture/furniture_data/obj_new'
    furniture_list = os.listdir(base_folder)

    tr_ratio = 0.7
    tr_list = np.random.choice(furniture_list, int(len(furniture_list) * tr_ratio), replace=False)
    te_list = list(set(furniture_list) - set(tr_list))

    base_tr = '/SSD1/furniture/furniture_data/new_train'
    for tr_name in tr_list:
        tr_old = os.path.join(base_folder, tr_name)
        tr_new = os.path.join(base_tr, tr_name)
        os.system('cp -r %s %s' %(tr_old, tr_new))

    base_te = '/SSD1/furniture/furniture_data/new_test'
    for te_name in te_list:
        te_old = os.path.join(base_folder, te_name)
        te_new = os.path.join(base_te, te_name)
        os.system('cp -r %s %s' %(te_old, te_new))

def new_dataset():
    base = '/SSD1/furniture/furniture_data/obj_imp'
    all_img = glob(os.path.join(base, '*', 'outline', '*.png'))

    base_new = '/SSD1/furniture/furniture_data/obj_new'
    for img in all_img:
        furniture = img.split('/')[-3]
        part_name = img.split('/')[-1]

        new_folder = os.path.join(base_new, furniture)
        new_path = os.path.join(new_folder, part_name)

        os.makedirs(new_folder, exist_ok=True)
        os.rename(img, new_path)


delete_obj()