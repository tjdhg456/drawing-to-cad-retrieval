from glob import glob
import subprocess
import numpy as np
import os

python = '/home/lilka/anaconda3/envs/nlp_/bin/python'

type_list = ['only']
gpu = '6'
input_folder = '/SSD1/furniture/furniture_data/test/'

for type in type_list:
    if type == 'only':
        folder_name = 'only2D'
    else:
        folder_name = '2D_3D'

    model_list = sorted(glob(os.path.join('/SSD1/furniture/result/', folder_name, '*.pt')))
    for model in model_list:
        emb_size = int(model.split('/')[-1].split('_')[0][3:])
        script = '%s inference.py --model_path %s --embedding_size %d --gpus %s --image_folder %s --type %s' \
                 % (python, model, emb_size, gpu, input_folder, type)
        os.system(script)

