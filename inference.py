import numpy as np
from glob import glob
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms
from PIL import Image
import argparse
from tqdm import tqdm
import random
from module.model import MVCNN
from module import transform_
from glob import glob

## Option
args = argparse.ArgumentParser()
# args.add_argument('--model_path', type=str, default='/SSD1/furniture/result/new/emb256_lr0.000100_adam_0.365_new.pt')
args.add_argument('--model_path', type=str, default='/SSD1/furniture/result/new_binary_resize/emb256_lr0.000100_adam_0.442_random_size.pt')
# args.add_argument('--model_path', type=str, default='/SSD1/furniture/result/new/emb256_lr0.000100_adam_0.523_new_binary_resize.pt')
args.add_argument('--te_all', type=int, default=1)

args.add_argument('--embedding_size', type=int, default=256)
args.add_argument('--gpus', type=str, default='5')
args.add_argument('--image_folder', type=str, default='/SSD1/furniture/external_test/image')
args.add_argument('--input_folder', type=str, default='/SSD1/furniture/external_test/test')
# args.add_argument('--image_folder', type=str, default='/SSD1/furniture/furniture_data/new_test')
# args.add_argument('--input_folder', type=str, default='/SSD1/furniture/furniture_data/new_test')
args.add_argument('--num_te', type=int, default=0)
args.add_argument('--type', type=str, default='only')
arg = args.parse_args()

if arg.te_all == 0:
    te_all = False
elif arg.te_all == 1:
    te_all = True

print('********************************')
print('Model : %s' %arg.model_path)
print('Type : %s' %arg.type)

## GPU
os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus

num = 0
count1, count2 = 0, 0

np.random.seed(512)
image_list = glob(os.path.join(arg.image_folder, '*', '*.png'))
input_list = glob(os.path.join(arg.input_folder, '*', '*.png'))
random.shuffle(input_list)

for input_path in input_list:
    ## Path
    cad_base = '/SSD1/furniture/furniture_data/obj_depth'

    ## Load the model
    saved_path = arg.model_path
    saved_weight = torch.load(saved_path)

    if arg.type == 'only':
        model = models.resnet34(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if arg.embedding_size != 1000:
            model.fc = nn.Linear(512, arg.embedding_size)

        model.load_state_dict(saved_weight)
        model = nn.DataParallel(model).cuda()

    else:
        model = MVCNN(emb_size=arg.embedding_size, infer=True)
        model.load_state_dict(torch.load(arg.model_path))

    model.eval()
    ## Input Image
    input_folder = '/'.join(input_path.split('/')[:-1])
    furniture = input_path.split('/')[-2]
    input_part = input_path.split('/')[-1].split('_')[0].strip()

    ## Query Image
    query_folder = os.path.join(arg.image_folder, furniture)

    if arg.type == 'only':
        query_part_set = set([part_image.split('_')[0] for part_image in os.listdir(query_folder)])
        query_dict = dict()
        for query_part in query_part_set:
            if arg.num_te != 0:
                query_dict[query_part] = np.random.choice([query_image for query_image in glob(os.path.join(query_folder, '%03d_*.png') %int(query_part))], arg.num_te)
            else:
                query_dict[query_part] = [query_image for query_image in glob(os.path.join(query_folder, '%03d_*.png') %int(query_part))]

    else:
        query_part_set = set([part_image.split('_')[0] for part_image in os.listdir(query_folder)])
        query_dict = dict()
        for query_part in query_part_set:
            query_dict[query_part] = [query_image for query_image in glob(os.path.join(cad_base, furniture, query_part, '*.png'))]

    ## Input surface vs Query surface (Similarity Comparison)
    # Transformation
    transform_input = transforms.Compose([
        transform_.Padded(),
        transform_.RandomResize(max_size=512, ratio=1.5),
        transforms.Resize([512, 512]),
        transform_.Binary(criterion=150),
        transforms.ToTensor(),
        transform_.MintoMax()
    ])

    transform_query = transforms.Compose([
        transforms.Resize([512, 512]),
        transform_.Binary(criterion=150),
        transforms.ToTensor(),
        transform_.MintoMax()
    ])

    transform3D = transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.ToTensor(),])

    # Image Loading
    if arg.type == 'only':
        input_image = torch.unsqueeze(transform_input(Image.open(input_path).convert('L')), dim=0).cuda()
    else:
        input_image = torch.unsqueeze(transform_input(Image.open(input_path).convert('L')), dim=0)

    # Query Loading
    similarity_dict = dict()
    for query in query_dict.keys():
        query_list = query_dict[query]
        image_list = []

        if arg.type == 'only':
            for query_path in query_list:
                im = torch.unsqueeze(transform_query(Image.open(query_path).convert('L')), dim=0)
                image_list.append(im)
            image_query = torch.cat(image_list, dim=0).cuda()

            input_embedding = model(input_image)
            total_similarity = 0
            for img in image_query:
                query_embedding = model(torch.unsqueeze(img, dim=0))
                if te_all == True:
                    similarity_ix = torch.sum((query_embedding - input_embedding) ** 2).item()
                else:
                    similarity_imp = (query_embedding - input_embedding) ** 2
                    similarity_ix = []
                    for imp in similarity_imp:
                        index = sorted(range(len(imp)), key=lambda x: imp[x])
                        imp = torch.unsqueeze(imp[index[:50]], dim=0)
                        similarity_ix.append(imp)
                    similarity_ix = torch.sum(torch.cat(similarity_ix, dim=0)).item()

                total_similarity += similarity_ix
                del query_embedding
            similarity_dict[query] = total_similarity/len(image_query)

        else:
            out_list = []
            input_embedding = model.model_2D(input_image)
            for query_path in query_list:
                im = torch.unsqueeze(transform3D(Image.open(query_path).convert('L')), dim=0)
                out1 = torch.unsqueeze(model.model_3D_1(im), dim=0)
                out_list.append(out1)
            out2 = torch.max(torch.cat(out_list, dim=1), dim=1)[0]
            out3 = model.model_3D_2(out2).view(1,-1)
            query_embedding = model.model_3D_fc(out3)
            similarity_dict[query] = torch.mean(torch.sum((query_embedding - input_embedding) ** 2, dim=1).view(-1))

    print('Input furniture: %s, obj: %s' %(furniture, input_part))
    ## Sorting the list (Make the lowest distance values into first element)
    similarity = [(keys, similarity) for keys, similarity in similarity_dict.items()]
    similarity_ix = sorted(range(len(similarity)), key=lambda x: similarity[x][1])
    similarity = [similarity[ix] for ix in similarity_ix]
    print(similarity)
    if similarity[0][0] == input_part:
        count1 += 1
    if input_part in [similarity[0][0], similarity[1][0]]:
        count2 += 1
    num += 1

    acc1 = count1 * 100 / num
    acc2 = count2 * 100 / num
    print('Accuracy - Top 1 : %.2f, Top 2: %.2f, Num : %d' %(acc1, acc2, num))

