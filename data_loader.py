# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 20:08
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_loader.py
# @Software: PyCharm
import os
import shutil

import cv2
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import albumentations as A

# def data_preparation(dataroot, scale, save_root):
#     file_folders = os.listdir(dataroot)
#     for file_folder in file_folders:
#         file_list = os.listdir(os.path.join(dataroot, file_folder))
#         os.mkdir(os.path.join(save_root, file_folder))
#         for file in file_list:
#             image = cv2.imread(os.path.join(dataroot, file_folder, file))
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#             H, W = image.shape[:2]
#             L_image = cv2.resize(image, (int(H / scale), int(W / scale)))
#             L_image = cv2.cvtColor(L_image, cv2.COLOR_YCrCb2BGR)
#             cv2.imwrite(os.path.join(save_root, file_folder, file), L_image)
#
#
AID_root = 'L:/2022_AID/NWPU-RESISC45'
# Scale = 4
# Save_root = 'L:/2022_AID/NWPU-RESISC45_x4'
# os.mkdir(Save_root)
# data_preparation(AID_root, Scale, Save_root)

with open(r'L:/2022_AID/NWPU-RESISC45_validation.txt', 'w') as f:
    file_folders = os.listdir(AID_root)
    for file_folder in file_folders:
        file_list = os.listdir(os.path.join(AID_root, file_folder))
        for file in file_list:
            if random.randint(0, 10) == 9:
                f.write(os.path.join(file_folder, file) + '\n')


class PairedImageDataset(Dataset):
    """Paired image dataset for image restoration.
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.
    There are three modes:
    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.
    """

    def __init__(self, dataroot_gt, dataroot_lq, gt_size, scale, transform, load_txt, repeat=1):
        self.dataroot_gt = dataroot_gt
        self.dataroot_lq = dataroot_lq
        self.gt_size = gt_size
        self.scale = scale
        self.transform = transform
        self.load_txt = load_txt
        self.filelist = []
        self.repeat = repeat
        with open(self.load_txt, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                self.filelist.append(line)
        self.filelist = self.filelist * self.repeat

    def __getitem__(self, index):
        lq_path = os.path.join(self.dataroot_lq, self.filelist[index])
        gt_path = os.path.join(self.dataroot_gt, self.filelist[index])

        img_lq = cv2.imread(lq_path)
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2YCrCb)
        img_gt = cv2.imread(gt_path)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2YCrCb)

        if self.transform:
            img_label = self.transform(image=img_lq, mask=img_gt)
            img_lq, img_gt = img_label['image'], img_label['mask']

        if self.scale:
            w, h = self.gt_size
            img_w, img_h = img_gt.shape[0], img_gt.shape[1]
            x, y = random.randint(0, img_w - w), random.randint(0, img_h - h)
            img_gt = img_gt[x:x + w, y:y + h, :]
            img_lq = img_lq[int(x / self.scale):int(x / self.scale) + int(w / self.scale),
                            int(y / self.scale):int(y / self.scale) + int(h / self.scale)]

        img_gt, img_lq = \
            transforms.ToTensor()(img_gt), transforms.ToTensor()(img_lq)
        img_gt, img_lq = \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_gt), \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_lq)

        return {'lq': img_lq[0:1], 'gt': img_gt[0:1], 'lq_RGB': img_lq, 'gt_RGB': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.filelist)


def data_transfer(root_dir, dst_dir):
    paths = os.listdir(root_dir)
    for path in paths:
        files = os.listdir(os.path.join(root_dir, path))
        for file in files[:10]:
            shutil.copyfile(os.path.join(root_dir, path, file), os.path.join(dst_dir, file))
