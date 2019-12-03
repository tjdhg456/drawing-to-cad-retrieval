import PySimpleGUI as sg
import os
from glob import glob
from collections import defaultdict
from PIL import Image, ImageTk
from torchvision import transforms
import io
import matplotlib.pyplot as plt
























# all_data = glob(os.path.join('../../data/single_part/05_0056/surface/*.png'))
# all_data = [data_.replace('\\', '/') for data_ in all_data]
#
# input_path = all_data[0]
# input_furniture = input_path.split('/')[-3]
# input_part = input_path.split('/')[-1].split('_')[0]
#
# part_dict, part_set = defaultdict(list), []
# for data_ in all_data:
#     part_ = data_.split('/')[-1].split('_')[0]
#     part_set.append(part_)
#     part_dict[part_].append(data_)
#
# query_path = []
# for part in part_dict.keys():
#     query_path.append(part_dict[part][0])
#
# def get_img_data(f, size, first=False):
#     """Generate surface data using PIL
#     """
#     img = Image.open(f)
#     img = transforms.CenterCrop([1024,1024])(img).resize([size,size])
#     if first:                     # tkinter is inactive the first time
#         bio = io.BytesIO()
#         img.save(bio, format="PNG")
#         del img
#         return bio.getvalue()
#     return ImageTk.PhotoImage(img)
#
# image_elem = [sg.Image(data=get_img_data(input_path, size=256, first=True))]
# query_image_list = [(query.split('/')[-1].split('_')[0], sg.Image(data=get_img_data(query, size=(int(1350/len(query_path))), first=True))) for query in query_path]
# query_column_list = []
# for column_list in query_image_list:
#     query_column_list.append(sg.Column([[sg.Text('Obj : %s' %column_list[0])], [column_list[1]]]))
#
# img_column = [query_column for query_column in query_column_list]
#
# list_window = [[sg.Text(''),sg.Text("Input Furniture : %s, obj : %s" %(input_furniture, input_part), text_color='blue', size=(25,1))],
#                image_elem,
#                [sg.Text('')],
#                img_column]
#
# window = sg.Window("Window", list_window, size=(1600, 600))
# while True:
#     button, values = window.Read()
#     window.close()
#     exit()