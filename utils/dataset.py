import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg
from torchvision import transforms
from torchvision.transforms import functional as F

import random
from .autoaugment import AutoAugImageNetPolicy
from skimage import transform as transform_sk
import math



class CRC():
    def __init__(self, root, is_train=True, data_len=None, transform=None, vanilla=None, split=None, gan=False, gan_ratio=None):
        
        self.low_data = True

        self.base_folder = "Kather_texture_2016_image_tiles_5000/all"
        self.root = root
        self.data_dir = join(self.root, self.base_folder)

        self.is_train = is_train
        self.transform = transform

        if vanilla is not None:
            self.vanilla = vanilla

        shared_gan = True
        self.gan = gan
        if self.gan:
            self.gan_ratio = gan_ratio

            if split is not None:
                if shared_gan:
                    self.split_sep = split.split('_') #()

                    if int(self.split_sep[0]) >= 10:
                        self.split_train_gan = '50_50/3way_splits/generated/train_gan_10_sp1.txt'
                    elif int(self.split_sep[0]) == 5:
                        self.split_train_gan = '50_50/3way_splits/generated/train_gan_5_sp1.txt'
                    elif int(self.split_sep[0]) <= 3:
                        self.split_train_gan = '50_50/3way_splits/generated/train_gan_3_sp1.txt'
                    else:
                        print("[ERROR] No GAN-generated files found")
                        self.split_train_gan = '50_50/3way_splits/generated/train_gan_' + str(split[:-1]) + '1.txt'
                        #CRC_colorectal_cancer_histology/low_data/50_50/3way_splits/generated/train_gan_3_sp1.txt
                else:
                    #self.base_folder_gan = 'generated/3way_splits/' + str(split)[-1] +  '/train_gan_' + str(split)
                    #self.data_dir_gan = join(self.root, self.base_folder_gan)
                    self.split_train_gan = '50_50/3way_splits/' + str(split)[-1] + '/generated/train_gan_' + str(split) + '.txt'
            # else:
            #     print("[INFO] No GAN generated images found")

        real_gan_together = False # one text file for real and GAN images
        if split is not None:
            if len(str(split)) >= 5:
                if real_gan_together:
                    self.split_train = '50_50/3way_splits/' + str(split)[-1] + '/generated/with_real/train_real_gan_' + str(split) + '.txt'
                else:
                    #print('3way_splits/' + str(split)[-1] + '/train_' + str(split) + '.txt')
                    self.split_train = '50_50/3way_splits/' + str(split)[-1] + '/train_' + str(split) + '.txt'
                #print('3way_splits/' + str(split)[-1] + 'test_100_sp' + str(split)[-1] + '.txt')    
                self.split_test = '50_50/3way_splits/' + str(split)[-1] + '/test_100_sp' + str(split)[-1] + '.txt'
            else:
                self.split_train = 'train_' + str(split) + '.txt'
                self.split_test = 'test.txt'
        else:
            self.split_train = 'train.txt'
            self.split_test = 'test.txt'


        if not self.low_data: 

            if self.is_train:
                print("[INFO] Preparing train shape_hw list...")

                # image_name, target_class = self._flat_breed_images[index]
                # image_path = join(self.images_folder, image_name)
                # image = Image.open(image_path).convert('RGB')

                train_file_list = []

                for image_name in self.car_annotations:
                    img_name = join(self.data_dir, image_name[-1][0])
                    #print(img_name)

                    #img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                    img_temp = scipy.misc.imread(os.path.join(img_name))

                    train_file_list.append(image_name[-1][0])

            else:
                print("[INFO] Preparing test shape_hw list...")

                test_file_list = []

                for image_name in self.car_annotations:
                    img_name = join(self.data_dir, image_name[-1][0])

                    #img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                    img_temp = scipy.misc.imread(os.path.join(img_name))

                    test_file_list.append(image_name[-1][0])


        else:
            if self.is_train:
                print("[INFO] Preparing low-data regime training set...")

                train_file_list = []
                ld_label_list = []

                #data_list_file = os.path.join(root, self.image_list[list_name])
                #ld_train_val_file = open(os.path.join(self.root, 'low_data/50_50/my/', 'train_30.txt'))
                ld_train_val_file = open(os.path.join(self.root, 'low_data/', self.split_train))
                print('train:', ld_train_val_file)
                                        # train_100.txt, train_50.txt, train_30.txt, train_15.txt,  train_10.txt

                for line in ld_train_val_file:
                    split_line = line.split(' ') #()

                    if split_line[0] != '\n':
                        #print(split_line)
                        target = split_line[-1]
                        path = ' '.join(split_line[:-1])
                        #print(target[-1])

                        if target[-1] == '\n':
                            target = target[:-1]
                            target = int(target)
                            #print('xx')
                        #print(target)
                        #print(path)

                        train_file_list.append(path)
                        #print(line[:-1].split(' ')[-1])

                        ld_label_list.append(int(line[:-1].split(' ')[-1])) #- 1)

                print("[INFO] Real train samples number:" , len(train_file_list), ", and labels number:", len(ld_label_list))
            
                if self.gan:
                    print("[INFO] Preparing GAN-generated training set...")

                    train_file_list_gan = []
                    ld_label_list_gan = []

                    #data_list_file = os.path.join(root, self.image_list[list_name])
                    #ld_train_val_file = open(os.path.join(self.root, 'low_data/50_50/my/', 'train_30.txt'))
                    ld_train_val_file_gan = open(os.path.join(self.root, 'low_data/', self.split_train_gan))
                    print('train GAN:', ld_train_val_file_gan)
                                            # train_100.txt, train_50.txt, train_30.txt, train_15.txt,  train_10.txt

                    #print("max im_p_cl", int( len(train_file_list) / 8 ))

                    for ind, line in enumerate(ld_train_val_file_gan):

                        #print(ind)
                        split_line = line.split(' ') #()

                        #print(split_line)

                        if split_line[0] != '\n':
                            #print(split_line)

                            target = split_line[-1]
                            path = ' '.join(split_line[:-1])

                            #print(target[-1])

                            if target[-1] == '\n':
                                target = target[:-1]
                                target = int(target)
                                #print('xx')

                            #print(ld_label_list_gan)
                            #print(int( (len(train_file_list) / 8) * 0.2))

                            if (ind==0) or (target != ld_label_list_gan[-1]):
                                img_per_class_count = 0
                                #print("new_class")

                            #print(img_per_class_count)
                            img_per_class_max = math.ceil( (len(train_file_list) / 8) * self.gan_ratio ) # 0.1, 0.2, 0.25, 0.5 #/ 2 / 2 ): # 8 - number of classes
                            #print(img_per_class_max)

                            # if img_per_class_max < 5.0:
                            #     if img_per_class_max > 0.0 and img_per_class_max < 1.0:
                            #         img_per_class_max += 1
                            # elif img_per_class_max == 0:
                            #     print("[ERROR] Too small portion of the GAN data picked")

                            img_per_class_max = int(img_per_class_max)
                            #print(img_per_class_max)

                            if img_per_class_count < img_per_class_max: 
                                #print(target)
                                #print(path)
                                #print("im_p_cl:", img_per_class_count)
                                #print("bruh")
                                train_file_list_gan.append(path)

                                #print(line[:-1].split(' ')[-1])
                                #print(target)
                                #print(int(line[:-1].split(' ')[-1]))

                                #ld_label_list_gan.append(int(line[:-1].split(' ')[-1])) #- 1)
                                ld_label_list_gan.append(target) #- 1)

                                img_per_class_count += 1

                    print("[INFO] GAN-generated train samples number:" , len(train_file_list_gan), ", and labels number:", len(ld_label_list_gan))                    

                    train_file_list.extend(train_file_list_gan)
                    ld_label_list.extend(ld_label_list_gan)

                print("[INFO] Total train samples number:" , len(train_file_list), ", and labels number:", len(ld_label_list))

            else:
                print("[INFO] Preparing low-data regime test set")

                test_file_list = []
                ld_label_list = []

                #data_list_file = os.path.join(root, self.image_list['test'])
                #ld_train_val_file = open(os.path.join(self.root, 'low_data/50_50/my', 'test.txt'))
                ld_train_val_file = open(os.path.join(self.root, 'low_data/', self.split_test))
                print('test:', ld_train_val_file)
                                        # train_100.txt, train_50.txt, train_30.txt, train_15.txt,  train_10.txt

                for line in ld_train_val_file:
                    split_line = line.split(' ') #()
                    target = split_line[-1]
                    path = ' '.join(split_line[:-1])
                    target = int(target)

                    test_file_list.append(path)
                    ld_label_list.append(int(line[:-1].split(' ')[-1])) #- 1)

                print("[INFO] Test samples number:" , len(test_file_list), ", and labels number:", len(ld_label_list))

        if self.is_train:
            self.train_img = []
            self.train_mask = []

            print("[INFO] Preparing train files...")
            i = 0
            for train_file in train_file_list[:data_len]:
                train_img_temp = scipy.misc.imread(os.path.join(self.data_dir, train_file))
                #print("Train file:", train_file)

                if (train_img_temp.shape[0] > 500) or (train_img_temp.shape[1] > 500):  # for nabirds only

                    if i < 10:
                        print("Before:", train_img_temp.shape[0], train_img_temp.shape[1])

                        img_name = ("test/img_before_tr" + str(i) + ".png")
                        Image.fromarray(train_img_temp, mode='RGB').save(img_name)


                    if train_img_temp.shape[0] > train_img_temp.shape[1]:
                        max500 = train_img_temp.shape[0] / 500
                    else:
                        max500 = train_img_temp.shape[1] / 500
                    

                    train_img_temp = transform_sk.resize(train_img_temp, ( int( (train_img_temp.shape[0] // max500)) , int( (train_img_temp.shape[1] // max500)) ),mode='constant', anti_aliasing=True, anti_aliasing_sigma=None, preserve_range=True)
                    #train_img_mask = transform_sk.resize(train_img_mask, ( int( (train_img_mask.shape[0] // 2)) , int(( train_img_mask.shape[1] // 2)) ),mode='constant', anti_aliasing=True, anti_aliasing_sigma=None, preserve_range=True)
                    
                    if i < 10:
                        print("After:", train_img_temp.shape[0], train_img_temp.shape[1])

                        #train_img_temp = (train_img_temp * 255).astype(np.uint8)
                        train_img_temp = (train_img_temp).astype(np.uint8)

                        img_name = ("test/img_after_tr" + str(i) + ".png")
                        Image.fromarray(train_img_temp, mode='RGB').save(img_name)
                else:
                    if i < 5:
                        print("Normal:", train_img_temp.shape[0], train_img_temp.shape[1])

                        img_name = ("test/img_normal_tr" + str(i) + ".png")
                        Image.fromarray(train_img_temp, mode='RGB').save(img_name)

                self.train_img.append(train_img_temp)
                i = i+1

            if not self.low_data: 
                self.train_label = [  ( torch.from_numpy(np.array( x[-2][0][0].astype(np.float32) )).long() - 1 )  for x in self.car_annotations ][:data_len]
            else:
                self.train_label = ld_label_list[:data_len]

            self.train_imgname = [x for x in train_file_list[:data_len]]

        else:
            self.test_img = []
            self.test_mask = []

            print("[INFO] Preparing test files...")
            i = 0
            for test_file in test_file_list[:data_len]:
                test_img_temp = scipy.misc.imread(os.path.join(self.data_dir, test_file))
                #print("Test file:", test_file)
                test_img_temp = (test_img_temp).astype(np.uint8)

                if (test_img_temp.shape[0] > 500) or (test_img_temp.shape[1] > 500):  # for nabirds only

                    #test_img_temp = (test_img_temp).astype(np.uint8)

                    # if i < 10:
                    #     print("Before:", test_img_temp.shape[0], test_img_temp.shape[1])

                    #     img_name = ("test/img_before_test" + str(i) + ".png")
                    #     Image.fromarray(test_img_temp, mode='RGB').save(img_name)


                    if test_img_temp.shape[0] > test_img_temp.shape[1]:
                        max500 = test_img_temp.shape[0] / 500
                    else:
                        max500 = test_img_temp.shape[1] / 500


                    test_img_temp = transform_sk.resize(test_img_temp, ( int( (test_img_temp.shape[0] // max500)) , int( (test_img_temp.shape[1] // max500)) ),mode='constant', anti_aliasing=True, anti_aliasing_sigma=None, preserve_range=True)
                    #train_img_mask = transform_sk.resize(train_img_mask, ( int( (train_img_mask.shape[0] // 2)) , int(( train_img_mask.shape[1] // 2)) ),mode='constant', anti_aliasing=True, anti_aliasing_sigma=None, preserve_range=True)
                    
                self.test_img.append(test_img_temp)

                if (i % 1000 == 0):
                    print(i)
                i = i+1

            if not self.low_data: 
                self.test_label = [  ( torch.from_numpy(np.array( x[-2][0][0].astype(np.float32) )).long() - 1 )  for x in self.car_annotations ][:data_len]
            else:
                self.test_label = ld_label_list[:data_len]            
            
            self.test_imgname = [x for x in test_file_list[:data_len]]



    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)



    def __getitem__(self, index):

        if self.is_train:
    
            img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]

            double_crop = False # True # two different crops
            crop_only = self.vanilla # False

            rand_crop_im_mask = True # True
            if rand_crop_im_mask:
                h_max_img = img.shape[0]
                w_max_img = img.shape[1]

                portion1side = torch.distributions.uniform.Uniform(0.1,0.5).sample([1]) # 0.5,0.67 # 0.67,0.8 # 0.7,0.95,  0.6,0.8

                if double_crop:
                    portion1side_2 = torch.distributions.uniform.Uniform(0.8,0.9).sample([1]) # 0.67,0.8 # 0.8,0.9 # 0.7,0.95,  0.6,0.8
                    
                h_crop_mid_img = int(h_max_img * portion1side) 
                w_crop_mid_img = int(w_max_img * portion1side)

                h_crop_min_img = random.randint(0, (h_max_img - h_crop_mid_img)) # 40) #, 400-360) #, h - th)
                w_crop_min_img = random.randint(0, (w_max_img - w_crop_mid_img)) # 40)  #, 400-360) #, w - tw)

                h_crop_max_img = h_crop_mid_img + h_crop_min_img
                w_crop_max_img = w_crop_mid_img + w_crop_min_img

                # Crop image for bbox:
                if len(img.shape) == 3:
                    img_crop = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img), :] # h, w, ch
                else:
                    img_crop = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)] # h, w
                    img_crop = np.stack([img_crop] * 3, 2)

                if double_crop:
                    h_crop_mid_img = int(h_max_img * portion1side_2) 
                    w_crop_mid_img = int(w_max_img * portion1side_2)

                    h_crop_min_img = random.randint(0, (h_max_img - h_crop_mid_img)) # 40) #, 400-360) #, h - th)
                    w_crop_min_img = random.randint(0, (w_max_img - w_crop_mid_img)) # 40)  #, 400-360) #, w - tw)

                    h_crop_max_img = h_crop_mid_img + h_crop_min_img
                    w_crop_max_img = w_crop_mid_img + w_crop_min_img

                    # Crop image for bbox:
                    if len(img.shape) == 3:
                        img_crop2 = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img), :] # h, w, ch
                    else:
                        img_crop2 = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)] # h, w
                        img_crop2 = np.stack([img_crop2] * 3, 2)

            if len(img.shape) == 2:
                # print(img.shape)
                img = np.stack([img] * 3, 2)
                # print(mask.shape)
                # print(img.shape)

            img = (img).astype(np.uint8)
            img = Image.fromarray(img, mode='RGB')

            if rand_crop_im_mask:
                img_crop = (img_crop).astype(np.uint8)
                img_crop = Image.fromarray(img_crop, mode='RGB')

                if double_crop:
                    img_crop2 = (img_crop2).astype(np.uint8)
                    img_crop2 = Image.fromarray(img_crop2, mode='RGB')
                    
            if index < 10:
                # # import time
                from torchvision.utils import save_image
                img_tem = transforms.ToTensor()(img)
                img_name = ("test/img_bef" + str(index) + ".png")
                save_image( img_tem, img_name)

                if rand_crop_im_mask:
                    img_tem_crop = transforms.ToTensor()(img_crop)
                    img_name_crop = ("test/img_bef_crop" + str(index) + ".png")
                    save_image( img_tem_crop, img_name_crop)
                    
                    if double_crop:
                        img_tem_crop2 = transforms.ToTensor()(img_crop2)
                        img_name_crop2 = ("test/img_bef_crop2_" + str(index) + ".png")
                        save_image( img_tem_crop2, img_name_crop2)

            if self.transform is not None:
                img = self.transform(img)
                
                if rand_crop_im_mask:
                    transform_img_flip = transforms.Compose([
                                    transforms.Resize((96, 96),Image.BILINEAR), # my for bbox

                                    transforms.RandomHorizontalFlip(), # !!! FLIPPING in dataset.py !!!
                                    transforms.RandomVerticalFlip(), # from air

                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    #transforms.Normalize([0.650, 0.472, 0.584], [0.158, 0.164, 0.143]), # for CRC (our manual)
                                    ])

                    #img_crop = self.transform(img_crop)
                    img_crop = transform_img_flip(img_crop)

                    if double_crop:
                        #img_crop2 = self.transform(img_crop2)
                        img_crop2 = transform_img_flip(img_crop2)

            # #import time
            if index < 10:
                from torchvision.utils import save_image
                #print(img.shape)
                #print("next img")
                img_name = ("test/img_aft" + str(index) + ".png")
                save_image( img, img_name)

                if rand_crop_im_mask:
                    img_name_crop = ("test/img_aft_crop" + str(index) + ".png")
                    save_image( img_crop, img_name_crop)
                    
                    if double_crop:
                        img_name_crop2 = ("test/img_aft_crop2_" + str(index) + ".png")
                        save_image( img_crop2, img_name_crop2)


        else:
            img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]

            if len(img.shape) == 2:
                # print(img.shape)                
                img = np.stack([img] * 3, 2)
                # print(mask.shape)
                # print(img.shape)
                        
            img = (img).astype(np.uint8)
            img = Image.fromarray(img, mode='RGB')

            if self.transform is not None:
                img = self.transform(img)

        if self.is_train:
            if double_crop:
                return img, img_crop, img_crop2, target
            else:
                if crop_only:
                    return img, target
                else:
                    return img, img_crop, target

        else:
            return img, target
