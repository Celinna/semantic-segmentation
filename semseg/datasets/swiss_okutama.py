# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

import random

class SwissOkutama(Dataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_classes=8,
                 inference=False):

        super().__init__()

        self.mean = [0.39313033, 0.48066333, 0.45113695] # for BGR channels
        self.std = [0.1179, 0.1003, 0.1139]
        self.root = root
        self.list_path = list_path
        self.n_classes = num_classes
       
        if not inference:
            self.img_list = [line.strip().split() for line in open(list_path)]

            self.files = self.read_files()
                
            self.class_weights = torch.FloatTensor([6.39009734, 0.70639623,
                                                    0.59038726, 0.50078311, 
                                                    17.07579971,  0.42550586,
                                                    7.51839971, 13.64782882]).cuda()    
        self.scale_factor = 16
        
        self.ignore_label = 255
        self.label_mapping = {0: self.ignore_label, 
                              1: 0, 2: 1, 
                              3: 2, 4: 3, 
                              5: 4, 6: 5, 
                              7: 6, 8: 7, 
                              9: self.ignore_label}
        
        self.CLASSES  = ["Outdoor structures", "Buildings", "Paved ground", "Non-paved ground", "Train tracks", "Plants", "Wheeled vehicles", "Water"]           
        self.colormap = {
                        # in RGB order
                        "Background": [0, 0, 0], 
                        "Outdoor structures": [237, 237, 237],
                        "Buildings": [181, 0, 0],
                        "Paved ground": [135, 135, 135],
                        "Non-paved ground": [189, 107, 0],
                        "Train tracks": [128, 0, 128],
                        "Plants": [31, 123, 22],
                        "Wheeled vehicles": [6, 0, 130],
                        "Water": [0, 168, 255],
                        "People": [240, 255, 0]
                    }
        self.idx2color = {k:v for k,v in enumerate(list(self.colormap.values()))}
 
    def __len__(self):
        return len(self.files)
    
    def read_files(self):
        files = []
        
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name,
                "weight": 1
            })
        return files
    
    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        return np.array(label).astype('int32')
    
    
    def multi_scale_aug(self, image, label=None,
                        rand_scale=1):
        
        h, w = image.shape[:2]
        
        long_size = np.int(w * rand_scale + 0.5)
        
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        return image, label
    
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        
        image = cv2.imread(os.path.join(self.root, item["img"]),
                           cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(self.root, item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        
        # image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_CUBIC)
        
        rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
        image, label = self.multi_scale_aug(image, label,
                                            rand_scale=rand_scale)
            
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))

        print(image.shape, label.shape)        
        label = self.convert_label(label)
        label = self.label_transform(label)

        return torch.from_numpy(image), torch.from_numpy(label).long()
        

    def category2mask(self, img):
        """ Convert a category image to color mask """
        if len(img) == 3:
            if img.shape[2] == 3:
                img = img[:, :, 0]
    
        mask = np.zeros(img.shape[:2] + (3, ), dtype='uint8')
    
        for category, mask_color in self.idx2color.items():
            locs = np.where(img == category)
            mask[locs] = mask_color
        
        return mask
    

    def save_pred(self, preds, sv_path, name, rgb=False):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            if rgb:
                mask_rgb = self.category2mask(pred)
                save_img = Image.fromarray(mask_rgb)
                save_img.save(os.path.join(sv_path, name +'.png'))
            else:
                save_img = Image.fromarray(pred)
                save_img.save(os.path.join(sv_path, name[i]+'.png'))
            
