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


# My:
from torchvision import transforms

import random
from torchvision.transforms import functional as F

from .autoaugment import AutoAugImageNetPolicy

from skimage import transform as transform_sk

#import U2Net
#from U2Net.u2net_test import mask_hw
#


# Try to add?
# def set_seed(args):
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
    
#     #
#     #torch.cuda.manual_seed(args.seed)
#     #

#     if args.n_gpu > 0:
#         torch.cuda.manual_seed_all(args.seed)
# #        



class CRC():
    #def __init__(self, root, is_train=True, data_len=None, transform=None):
    def __init__(self, root, is_train=True, data_len=None, transform=None, vanilla=None, split=None):
        
        self.low_data = True
        saliency_check = False #False

        self.base_folder = "Kather_texture_2016_image_tiles_5000/all"
        self.root = root
        self.data_dir = join(self.root, self.base_folder)

        self.is_train = is_train
        self.transform = transform

        if split is not None:
            #print('3way_splits/' + str(split)[-1] + '/train_' + str(split) + '.txt')
            #print('3way_splits/' + str(split)[-1] + 'test_100_sp' + str(split)[-1] + '.txt')

            self.split_train = '3way_splits/' + str(split)[-1] + '/train_' + str(split) + '.txt'
            self.split_test = '3way_splits/' + str(split)[-1] + '/test_100_sp' + str(split)[-1] + '.txt'
        else:
            self.split_train = 'train_' + str(split) + '.txt'
            self.split_test = 'test.txt'

        if vanilla is not None:
            self.vanilla = vanilla

        if not self.low_data: 
            # train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
            # train_file_list_full = [ os.path.join(self.root, 'images', x)  for i, x in zip(train_test_list, img_name_list) if i]
            
            # self.full_data_set = io.loadmat(os.path.join(self.root,'cars_annos.mat'))

            # self.car_annotations = self.full_data_set['annotations']
            # self.car_annotations = self.car_annotations[0]



            if self.is_train:
                print("[INFO] Preparing train shape_hw list...")


                if saliency_check:
                    train_shape_hw_list = []

                    train_file_list_full = []

                # image_name, target_class = self._flat_breed_images[index]
                # image_path = join(self.images_folder, image_name)
                # image = Image.open(image_path).convert('RGB')

                train_file_list = []


                # for img_name in train_file_list:
                    #     # remove in further code and in function mask_hw !
                    # img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                    # shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)
                    # #print(img_name)
                    # #print(shape_hw_temp)
                    # train_shape_hw_list.append(shape_hw_temp)


                #for image_name, target_class in self._flat_breed_images:
                for image_name in self.car_annotations:

                    # img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
                    # image = Image.open(img_name).convert('RGB')

                    #img_name = join(self.images_folder, image_name)
                    img_name = join(self.data_dir, image_name[-1][0])
                    #print(img_name)


                    #img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                    img_temp = scipy.misc.imread(os.path.join(img_name))


                    if saliency_check:
                        shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)
                        
                        
                        
                        if (shape_hw_temp[0] > 500) or (shape_hw_temp[1] > 500):
                            if shape_hw_temp[0] > shape_hw_temp[1]:
                                max500 = shape_hw_temp[0] / 500
                            else:
                                max500 = shape_hw_temp[1] / 500
                                

                            shape_hw_temp[0] = int( shape_hw_temp[0] // max500 )
                            shape_hw_temp[1] = int( shape_hw_temp[1] // max500 )


                    
                        #print(shape_hw_temp)
                        train_shape_hw_list.append(shape_hw_temp)

                        train_file_list_full.append(img_name)

                    train_file_list.append(image_name[-1][0])

            else:
                print("[INFO] Preparing test shape_hw list...")

                if saliency_check:
                    test_shape_hw_list = []

                    test_file_list_full = []

                test_file_list = []

                # for img_name in test_file_list: 
                #     # remove in further code and in function mask_hw !
                #     img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                #     shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)
                #     #print(img_name)
                #     #print(shape_hw_temp)
                #     test_shape_hw_list.append(shape_hw_temp)

                for image_name in self.car_annotations:
                    # img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
                    # image = Image.open(img_name).convert('RGB')

                    #img_name = join(self.images_folder, image_name)
                    img_name = join(self.data_dir, image_name[-1][0])

                    #img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                    img_temp = scipy.misc.imread(os.path.join(img_name))


                    if saliency_check:

                        shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)

                        if (shape_hw_temp[0] > 500) or (shape_hw_temp[1] > 500):
                            if shape_hw_temp[0] > shape_hw_temp[1]:
                                max500 = shape_hw_temp[0] / 500
                            else:
                                max500 = shape_hw_temp[1] / 500
                                
                            shape_hw_temp[0] = int( shape_hw_temp[0] // max500 )
                            shape_hw_temp[1] = int( shape_hw_temp[1] // max500 )

                        #print(img_name)
                        #print(shape_hw_temp)
                        test_shape_hw_list.append(shape_hw_temp)

                        test_file_list_full.append(img_name)

                    test_file_list.append(image_name[-1][0])


        else:

            if self.is_train:

                print("[INFO] Low data regime training set")

                train_file_list = []
                ld_label_list = []

                #data_list_file = os.path.join(root, self.image_list[list_name])
                #ld_train_val_file = open(os.path.join(self.root, 'low_data/50_50/my/', 'train_30.txt'))
                ld_train_val_file = open(os.path.join(self.root, 'low_data/50_50/', self.split_train))
                print('train:', ld_train_val_file)
                                        # train_100.txt, train_50.txt, train_30.txt, train_15.txt,  train_10.txt

                for line in ld_train_val_file:
                    split_line = line.split(' ') #()
                    target = split_line[-1]
                    path = ' '.join(split_line[:-1])

                    target = int(target)

                    train_file_list.append(path)

                    #print(line[:-1].split(' ')[-1])

                    ld_label_list.append(int(line[:-1].split(' ')[-1])) #- 1)


                print("[INFO] Train samples number:" , len(train_file_list), ", and labels number:", len(ld_label_list))
            

            else:

                print("[INFO] Low data regime test set")

                test_file_list = []
                ld_label_list = []

                #data_list_file = os.path.join(root, self.image_list['test'])
                #ld_train_val_file = open(os.path.join(self.root, 'low_data/50_50/my', 'test.txt'))
                ld_train_val_file = open(os.path.join(self.root, 'low_data/50_50/', self.split_test))
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
            # My:
            self.train_img = []
            self.train_mask = []


            if saliency_check:
                full_ds = True
                if full_ds:
                    # train_mask_u2_list = []
                    # train_x_u2_list = []
                    # train_y_u2_list = []
                    # train_h_u2_list = []
                    # train_w_u2_list = []

                    #train_shape_hw_list = [x for i, x in zip(train_test_list, shape_hw_list) if i]  ## train + test together
                    train_mask_u2_list, train_x_u2_list, train_y_u2_list, train_h_u2_list, train_w_u2_list = mask_hw(full_ds=full_ds, img_path=train_file_list_full, shape_hw=train_shape_hw_list)
                    #print(train_file_list_full)

                else:
                    #scipy.misc.imsave('/l/users/20020067/Activities/FGIC/FFVT/Combined/FFVT_my/U2Net/images/img.png', train_img_temp)
                    train_img_path = os.path.join(self.root, 'images', train_file_list)
                    '''
                    mask_u2, x_u2, y_u2, h_u2, w_u2 = mask_hw(full_ds=full_ds, img_path=train_img_path, shape_hw=(h_max, w_max))
                    '''
                    #print(mask_u2, x_u2, y_u2, h_u2, w_u2)
                    # WORKERS !!!!!!!!!!!!!!!!!!!!!!!


            print("[INFO] Preparing train files...")
            i = 0
            for train_file in train_file_list[:data_len]:

                train_img_temp = scipy.misc.imread(os.path.join(self.data_dir, train_file))

                #print("Train file:", train_file)

                h_max = train_img_temp.shape[0] # y
                w_max = train_img_temp.shape[1] # x
                #ch_max = train_img_temp.shape[2]



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



                if saliency_check:

                    mask_u2 = train_mask_u2_list[i]
                    x_u2 = train_x_u2_list[i]
                    y_u2 = train_y_u2_list[i]
                    h_u2 = train_h_u2_list[i]
                    w_u2 = train_w_u2_list[i]

                    # mask_u2, x_u2, y_u2, h_u2, w_u2 = mask_hw(img_path=train_img_path, shape_hw=(h_max, w_max))
                    #print("_________________________________________________")
                    #print(mask_u2, x_u2, y_u2, h_u2, w_u2)
                    # WORKERS !!!!!!!!!!!!!!!!!!!!!!!


                    train_img_mask = mask_u2
                    #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                #


                    x, y, h, w = x_u2, y_u2, h_u2, w_u2
                    #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


                    # padding
                    padding = True
                    if padding:
                        p = 15 # extra space around bbox
                    else:
                        p = 0

                    x_min = x - p 
                    if x_min < 0:
                        x_min = 0
                    x_max = x + w + p
                    if x_max > w_max:
                        x_max = w_max

                    y_min = y - p
                    if y_min < 0:
                        y_min = 0
                    y_max = y + h + p
                    if y_max > h_max:
                        y_max = h_max

                    #print("[SALIENCY]")
                    #
                
                else:
                    x_min = w_max * 0.05 # 0.1
                    x_max = w_max * 0.95 # # 0.9 w -> x

                    y_min = h_max * 0.05 # 0.1
                    y_max = h_max * 0.95 # # 0.9 h -> y                  
                    #



                # CHEEEEEEEEEEEEECKEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
                if h_max <=1:
                    print("bad_h")
                    print(h_max)

                if w_max <=1:
                    print("bad_w")
                    print(w_max)

                if y_min >= y_max:
                    print("bad_y")
                    print("min:", y_min)
                    print("max:", y_max)
                    print("y:", y)
                    print("h:", h)
                    
                    # y_min = 0
                    # y_max = h_max

                if x_min >= x_max:
                    print("bad_x")
                    print("min:", x_min)
                    print("max:", x_max)  
                    print("x:", x)
                    print("w:", w)                                    
                    # x_min = 0
                    # x_max = w_max
                #

                
                # Crop image for bbox:
                if len(train_img_temp.shape) == 3:
                    # Black mask:
                    # for j in range(3):
                    #     train_img_temp[:, :, j] = train_img_temp[:, :, j] * img_mask
                    #

                    #if i < 5: print(train_img_temp.shape)
                    #test_img_temp = test_img_temp[int(y):int(y + h), int(x):int(x + w), :] # h, w, ch
                    train_img_temp = train_img_temp[int(y_min):int(y_max), int(x_min):int(x_max), :] # h, w, ch
                    #if i < 5: print(train_img_temp.shape)
                else:
                    # Black mask:
                    #train_img_temp[:, :] = train_img_temp[:, :] * img_mask
                    #

                    #if i < 5: print(train_img_temp.shape)
                    train_img_temp = train_img_temp[int(y_min):int(y_max), int(x_min):int(x_max)] # h, w
                    #if i < 5: print(train_img_temp.shape)


                if saliency_check:
                    # Crop mask for bbox:
                    train_img_mask = train_img_mask[int(y_min):int(y_max), int(x_min):int(x_max)]
                    
                    if ( (train_img_temp.shape[0] != train_img_mask.shape[0]) or (train_img_temp.shape[1] != train_img_mask.shape[1]) ):
                        print("_____Wrong Index is:", i)
                        print(train_img_temp.shape, train_img_mask.shape)

                    self.train_mask.append(train_img_mask)

                self.train_img.append(train_img_temp)

                i = i+1
            #


            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'images', train_file)) for train_file in
            #                   train_file_list[:data_len]]

            if not self.low_data: 
                self.train_label = [  ( torch.from_numpy(np.array( x[-2][0][0].astype(np.float32) )).long() - 1 )  for x in self.car_annotations ][:data_len]
            else:
                self.train_label = ld_label_list[:data_len]

            self.train_imgname = [x for x in train_file_list[:data_len]]

            # print("Pass 1")

    


        else:
            # My:
            self.test_img = []
            self.test_mask = []


            if saliency_check:
                full_ds = True
                if full_ds:
                    #test_shape_hw_list = [x for i, x in zip(train_test_list, shape_hw_list) if not i]  ## train + test together
                    test_mask_u2_list, test_x_u2_list, test_y_u2_list, test_h_u2_list, test_w_u2_list = mask_hw(full_ds=full_ds, img_path=test_file_list_full, shape_hw=test_shape_hw_list)

                else:
                    test_img_path = os.path.join(self.root, 'images', test_file_list)
                    mask_u2, x_u2, y_u2, h_u2, w_u2 = mask_hw(full_ds=full_ds, img_path=test_img_path, shape_hw=(h_max, w_max))


            print("[INFO] Preparing test files...")
            i = 0
            for test_file in test_file_list[:data_len]:
                
                test_img_temp = scipy.misc.imread(os.path.join(self.data_dir, test_file))

                #print("Test file:", test_file)

                h_max = test_img_temp.shape[0]
                w_max = test_img_temp.shape[1]
                #ch_max = test_img_temp.shape[2]





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
                    



                if saliency_check:

                    mask_u2 = test_mask_u2_list[i]
                    x_u2 = test_x_u2_list[i]
                    y_u2 = test_y_u2_list[i]
                    h_u2 = test_h_u2_list[i]
                    w_u2 = test_w_u2_list[i]


                    test_img_mask = mask_u2
                    #


                    x, y, h, w = x_u2, y_u2, h_u2, w_u2


                    # padding
                    padding = True
                    if padding:
                        p = 15 # extra space around bbox
                    else:
                        p = 0

                    x_min = x - p
                    if x_min < 0:
                        x_min = 0
                    x_max = x + w + p
                    if x_max > w_max:
                        x_max = w_max

                    y_min = y - p
                    if y_min < 0:
                        y_min = 0
                    y_max = y + h + p
                    if y_max > h_max:
                        y_max = h_max
                    
                    #print("[SALIENCY]")
                    #

                else:
                    x_min = w_max * 0.05 # 0.1
                    x_max = w_max * 0.95 # # 0.9 w -> x

                    y_min = h_max * 0.05 # 0.1
                    y_max = h_max * 0.95 # # 0.9 h -> y                   
                    #



                # CHEEEEEEEEEEEEECKEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
                if h_max <=1:
                    print("bad_h")
                    print(h_max)

                if w_max <=1:
                    print("bad_w")
                    print(w_max)

                if y_min >= y_max:
                    print("bad_y")
                    print("min:", y_min)
                    print("max:", y_max)
                    print("y:", y)
                    print("h:", h)
                    
                    # y_min = 0
                    # y_max = h_max

                if x_min >= x_max:
                    print("bad_x")
                    print("min:", x_min)
                    print("max:", x_max)  
                    print("x:", x)
                    print("w:", w)                                    
                    # x_min = 0
                    # x_max = w_max
                #

                
                # Crop image for bbox:
                if len(test_img_temp.shape) == 3:
                    # Black mask:
                    # for j in range(3):
                    #     test_img_temp[:, :, j] = test_img_temp[:, :, j] * img_mask
                    #

                    #test_img_temp = test_img_temp[int(y):int(y + h), int(x):int(x + w), :] # h, w, ch
                    test_img_temp = test_img_temp[int(y_min):int(y_max), int(x_min):int(x_max), :] # h, w, ch
                else:
                    # Black mask:
                    #test_img_temp[:, :] = test_img_temp[:, :] * img_mask
                    #

                    test_img_temp = test_img_temp[int(y_min):int(y_max), int(x_min):int(x_max)] # h, w


                if saliency_check:
                    # Crop mask for bbox:
                    test_img_mask = test_img_mask[int(y_min):int(y_max), int(x_min):int(x_max)]
                    
                    self.test_mask.append(test_img_mask)

                self.test_img.append(test_img_temp)

                if (i % 1000 == 0):
                    print(i)

                i = i+1
            #


            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in
            #                  test_file_list[:data_len]]

            
            if not self.low_data: 
                self.test_label = [  ( torch.from_numpy(np.array( x[-2][0][0].astype(np.float32) )).long() - 1 )  for x in self.car_annotations ][:data_len]
            else:
                self.test_label = ld_label_list[:data_len]            
            
            
            self.test_imgname = [x for x in test_file_list[:data_len]]

            # print("Pass test_1")





    def __len__(self):
        #return len(self.car_annotations)
        
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)




    def __getitem__(self, index):

        saliency_check = False #False


        if self.is_train:
            
            # if self.count < 5: print(self.count)

            #img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]
            # print("Pass 22222")

            # if self.count < 5: print(index)


            # With mask:
            
            if saliency_check:
                img, target, imgname, mask = self.train_img[index], self.train_label[index], self.train_imgname[index], self.train_mask[index]
            else:
                img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]


            # if self.count < 5: print(img.shape)
            # if self.count < 5: print(mask.shape)

            #print(imgname)
            # print(img.shape)
            # print(mask.shape)



            double_crop = False # True # two different crops
            crop_only = self.vanilla # False

            rand_crop_im_mask = True # True
            if rand_crop_im_mask:
                #print("Crop_inf")

                #h_max_im = 400
                h_max_img = img.shape[0]

                #w_max_im = 400
                w_max_img = img.shape[1]



                if saliency_check:
                    # portion1side = torch.rand()
                    portion1side = torch.distributions.uniform.Uniform(0.5,0.8).sample([1]) # 0.7,0.95,  0.6,0.8
                    #if index < 10: print(portion1side)

                else:
                    portion1side = torch.distributions.uniform.Uniform(0.1,0.5).sample([1]) # 0.5,0.67 # 0.67,0.8 # 0.7,0.95,  0.6,0.8

                    if double_crop:
                        portion1side_2 = torch.distributions.uniform.Uniform(0.8,0.9).sample([1]) # 0.67,0.8 # 0.8,0.9 # 0.7,0.95,  0.6,0.8
                        


                h_crop_mid_img = int(h_max_img * portion1side) 
                ##h_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
                #h_crop_mid_img = int(h_max_img * 0.5) #* 0.7) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75), 282 (50 % - 0.7)

                w_crop_mid_img = int(w_max_img * portion1side)
                ##w_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
                #w_crop_mid_img = int(w_max_img * 0.5) #* 0.7) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75), 282 (50 % - 0.7)

                h_crop_min_img = random.randint(0, (h_max_img - h_crop_mid_img)) # 40) #, 400-360) #, h - th)
                w_crop_min_img = random.randint(0, (w_max_img - w_crop_mid_img)) # 40)  #, 400-360) #, w - tw)

                h_crop_max_img = h_crop_mid_img + h_crop_min_img
                w_crop_max_img = w_crop_mid_img + w_crop_min_img


                # Crop image for bbox:
                if len(img.shape) == 3:
                    # Black mask:
                    # for j in range(3):
                    #     test_img_temp[:, :, j] = test_img_temp[:, :, j] * img_mask
                    #

                    #test_img_temp = test_img_temp[int(y):int(y + h), int(x):int(x + w), :] # h, w, ch
                    img_crop = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img), :] # h, w, ch
                else:
                    # Black mask:
                    #test_img_temp[:, :] = test_img_temp[:, :] * img_mask
                    #

                    img_crop = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)] # h, w

                    img_crop = np.stack([img_crop] * 3, 2)

                if saliency_check:
                    # Crop mask for bbox:
                    mask_crop = mask[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)]


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
                # print("222222222")
                # print(img.shape)
                img = np.stack([img] * 3, 2)
                # print(mask.shape)
                # print(img.shape)
            
            # if rand_crop_im_mask:
            #     if len(img_crop.shape) == 2:
            #         img_crop = np.stack([img_crop] * 3, 2)


            # try:
            #     torch.set_printoptions(profile="full")
            #     np.printoptions(threshold=np.inf)                
            #     print(img.tile)
            #     print("tile???")
            # except:
            #     pass



            img = (img).astype(np.uint8)

            img = Image.fromarray(img, mode='RGB')


            if rand_crop_im_mask:
                img_crop = (img_crop).astype(np.uint8)
                
                img_crop = Image.fromarray(img_crop, mode='RGB')

                if double_crop:
                    img_crop2 = (img_crop2).astype(np.uint8)

                    img_crop2 = Image.fromarray(img_crop2, mode='RGB')
                    
            #print("Pass 3333333333333333333")


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


            flip_mask_as_image = False #True # if False - turn on RandomHorizontalFlip in data_utils !!!
            
            flipped = False # temp
           
            if self.transform is not None:
                #img = self.transform(img)

                if not flip_mask_as_image: # normal
                    img = self.transform(img)
                    
                    if rand_crop_im_mask:

                        img_crop_tiny = int(196 * portion1side) 
                        transform_img_flip = transforms.Compose([
                                        #transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR),
                                        #transforms.Resize((560, 560), Image.BILINEAR), #transFG 600
                                        #transforms.RandomCrop((args.img_size, args.img_size)),
                                        
                                         # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                        transforms.Resize((88, 88),Image.BILINEAR), # my for bbox
                                        #transforms.Resize((img_crop_tiny, img_crop_tiny),Image.BILINEAR), # my for bbox
                                        #transforms.Resize((448, 448),Image.BILINEAR), # my for bbox
                                        
                                        #transforms.RandomHorizontalFlip(p=1.0), # !!! FLIPPING in dataset.py !!!
                                        transforms.RandomHorizontalFlip(), # !!! FLIPPING in dataset.py !!!
                                        transforms.RandomVerticalFlip(), # from air

                                        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add (FFVT)
                                        #AutoAugImageNetPolicy(),

                                        transforms.ToTensor(),
                                        #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        #transforms.Normalize([0.650, 0.472, 0.584], [0.158, 0.164, 0.143]), # for CRC (our manual)
                                        ])


                        #img_crop = self.transform(img_crop)
                        img_crop = transform_img_flip(img_crop)

                        if double_crop:
                            img_crop2 = self.transform(img_crop2)
                            #img_crop2 = transform_img_flip(img_crop2)


                else:
                    if random.random() < 0.5:
                        flipped = False
                        #print("[INFO]: No Flip")
                        img = self.transform(img)

                        if rand_crop_im_mask:
                            img_crop = self.transform(img_crop)

                            if double_crop:
                                img_crop2 = self.transform(img_crop2)

                    else: # MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        flipped = True
                        #print("[INFO]: Flip convert")

                        transform_img_flip = transforms.Compose([
                                        #transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR),
                                        #transforms.Resize((560, 560), Image.BILINEAR), #transFG 600
                                        #transforms.RandomCrop((args.img_size, args.img_size)),
                                        
                                         # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                        transforms.Resize((224, 224),Image.BILINEAR), # my for bbox
                                        #transforms.Resize((448, 448),Image.BILINEAR), # my for bbox

                                        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add (FFVT)
                                        #AutoAugImageNetPolicy(),
                                        
                                        #transforms.RandomHorizontalFlip(p=1.0), # !!! FLIPPING in dataset.py !!!
                                        transforms.RandomHorizontalFlip(), # !!! FLIPPING in dataset.py !!!
                                        transforms.RandomVerticalFlip(), # from air

                                        transforms.ToTensor(),
                                        #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        #transforms.Normalize([0.650, 0.472, 0.584], [0.158, 0.164, 0.143]), # for CRC (our manual)
                                        ])

                        img = transform_img_flip(img)

                        if rand_crop_im_mask:
                            img_crop = transform_img_flip(img_crop)
                            if double_crop:
                                img_crop2 = transform_img_flip(img_crop2)

            # My:
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



            if saliency_check:

                mask = (mask * 255).astype(np.uint8)
                # if self.count < 5: print("mask 255", mask)
                mask = Image.fromarray(mask, mode='L')
                # if self.count < 5: print("mask tensor before", mask)
                
                if rand_crop_im_mask:
                    mask_crop = (mask_crop * 255).astype(np.uint8)
                    mask_crop = Image.fromarray(mask_crop, mode='L')

                if index < 10:
                    # # import time
                    from torchvision.utils import save_image
                    mask_tem = transforms.ToTensor()(mask)
                    img_name = ("test/mask_bef" + str(index) + ".png")
                    save_image( mask_tem, img_name)

                    if rand_crop_im_mask:
                        mask_tem_crop = transforms.ToTensor()(mask_crop)
                        img_name_crop = ("test/mask_bef_crop" + str(index) + ".png")
                        save_image( mask_tem_crop, img_name_crop)


                if not flip_mask_as_image: # normal
                    transform_mask = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.ToPILImage(), #(mode='1'),

                        # non-overlapped and size 224:
                        transforms.Resize((14,14), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST                 

                        # non-overlapped and size 400:
                        #transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        
                        # non-overlapped and size 448:
                        #transforms.Resize((28,28), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        

                        # non-overlapped and size 400:
                        # transforms.Resize((29,29), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        # #transforms.Resize((27,27), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        # transforms.CenterCrop((25,25)), # !!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                        # overlapped patch 12 and size 400:
                        #transforms.Resize((33,33), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        
                        transforms.ToTensor()])
                else:
                    if flipped:
                        transform_mask = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.ToPILImage(), #(mode='1'),
                            transforms.RandomHorizontalFlip(p=1.0),
                            
                            # non-overlapped and size 224:
                            transforms.Resize((14,14), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST                 

                            # non-overlapped and size 400:
                            #transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            
                            # non-overlapped and size 448:
                            #transforms.Resize((28,28), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            
                        
                            # non-overlapped and size 400:
                            # transforms.Resize((29,29), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            # #transforms.Resize((27,27), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            # transforms.CenterCrop((25,25)), # !!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                            # overlapped patch 12 and size 400:
                            #transforms.Resize((33,33), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            
                            transforms.ToTensor()])
                    else:
                        transform_mask = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.ToPILImage(), #(mode='1'),

                            # non-overlapped and size 224:
                            transforms.Resize((14,14), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST                 

                            # non-overlapped and size 400:
                            #transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            
                            # non-overlapped and size 448:
                            #transforms.Resize((28,28), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            


                            # non-overlapped and size 400:
                            # transforms.Resize((29,29), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            # #transforms.Resize((27,27), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            # transforms.CenterCrop((25,25)), # !!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                            # overlapped patch 12 and size 400:
                            #transforms.Resize((33,33), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST

                            transforms.ToTensor()])


                mask = transform_mask(mask)
                
                if rand_crop_im_mask:
                    mask_crop = transform_mask(mask_crop)


                if index < 10:
                    # import time
                    # #print(img.shape)
                    # #mask_img = transforms.ToPILImage()(mask)
                    # print("next mask")

                    img_name = ("test/mask_aft" + str(index) + ".png")
                    save_image( mask, img_name)

                    # time.sleep(5)

                    if rand_crop_im_mask:
                        img_name_crop = ("test/mask_aft_crop" + str(index) + ".png")
                        save_image( mask_crop, img_name_crop)


                # mask1 = transforms.Resize((400, 400))(mask) # 400 - args.img_size
                # mask2 = transforms.ToTensor()(mask1),

                # mask = mask2
                # if self.count < 5: print(mask.shape)
                # if self.count < 5: print(mask)


                mask = torch.flatten(mask)

                if rand_crop_im_mask:
                    mask_crop = torch.flatten(mask_crop)


                # if self.count < 5: print(mask.shape)
                # if self.count < 5: print(mask)

                # mask_patch = np.ones((int(25), int(25)))

                # for part_n in range(len(train_parts)):
                #     part = train_parts[part_n]
                #     #if i < 5: print(len(part))

                #     if part[0] != 0:
                #         #if i < 5: print(part[1], part[2])
                #         x_min_p = part[1] - p_part
                #         if x_min_p < 0:
                #             x_min_p = 0
                #         x_max_p = part[1] + p_part
                #         if x_max_p > w_max:
                #             x_max_p = w_max

                #         y_min_p = part[2] - p_part
                #         if y_min_p < 0:
                #             y_min_p = 0
                #         y_max_p = part[2] + p_part
                #         if y_max_p > h_max:
                #             y_max_p = h_max

                #         #img_mask[int(y_min_p):int(y_max_p), int(x_min_p):int(x_max_p)] = 1 # Black mask
                #         train_img_mask[int(y_min_p):int(y_max_p), int(x_min_p):int(x_max_p)] = 0


                # Convert mask 400x400 to 25x25 (1 px = 1 patch of 16x16)

                # mask = mask[0, :, :]
                # #if self.count < 5: print(mask.shape)

                # mask_25 = torch.ones(625)
                # #mask_clear = mask >= 0.9


                # mask_16_list = []

                # for px in range(400):
                #     for py in range(400):
                #         x_t = 16 * px 
                #         y_t = 16 * py 

                #         mask_16 = mask[0:16,0:16]


                # for patch_id in range(625): # cuz 16x16
                #     mask_16_temp = mask_16[patch_id]
                #     indices = mask_16_temp.nonzero(as_tuple=True)

                #     if indices :
                #         mask_25[patch_id] = 0

                    # if mask[]:
                    #     mask_25_new = torch.where(mask_25 > 0, mask_25, 1.)
                    
                #type(mask)
                #print(mask.shape)

                # self.count = self.count + 1


        else:

            #img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]


            saliency_check = False #False
            if saliency_check:
                # With mask:
                img, target, imgname, mask = self.test_img[index], self.test_label[index], self.test_imgname[index], self.test_mask[index]
                #
            else:
                img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]

            #print(imgname)
            # print(img.shape)
            # print(mask.shape)

            if len(img.shape) == 2:
                # print("222222222")
                # print(imgname)
                # print(img.shape)                
                img = np.stack([img] * 3, 2)
                # print(mask.shape)
                # print(img.shape)
                
            # try:
            #     torch.set_printoptions(profile="full")
            #     np.printoptions(threshold=np.inf)                
            #     print(img.tile)
            #     print("tile???")
            # except:
            #     pass            
            
            img = (img).astype(np.uint8)
            
            img = Image.fromarray(img, mode='RGB')

            #print("Pass 3333333333333333333 test")

            if self.transform is not None:
                img = self.transform(img)


            if saliency_check:

                ### Mask:
                mask = (mask * 255).astype(np.uint8)
                mask = Image.fromarray(mask, mode='L')
                
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                transform_mask = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(), #(mode='1'),

                    # non-overlapped and size 224:
                    transforms.Resize((14,14), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST                 

                    # non-overlapped and size 400:
                    #transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    
                    # non-overlapped and size 448:
                    #transforms.Resize((28,28), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    


                    # non-overlapped and size 400:
                    # transforms.Resize((29,29), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    # #transforms.Resize((27,27), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    # transforms.CenterCrop((25,25)), # !!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    # overlapped patch 12 and size 400:
                    #transforms.Resize((33,33), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST

                    transforms.ToTensor()])

                mask = transform_mask(mask)

                mask = torch.flatten(mask)



        if self.is_train:

            if saliency_check:
                if double_crop:
                    return img, img_crop, img_crop, target, mask, mask_crop
                else:
                    return img, img_crop, target, mask, mask_crop
            else:
                if double_crop:
                    return img, img_crop, img_crop, target
                else:
                    if crop_only:
                        return img, target
                    else:
                        return img, img_crop, target

        else:
            if saliency_check:
                return img, target, mask
            else:
                return img, target





class CUB():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        
        self.low_data = True
        
        self.root = root
        self.is_train = is_train
        self.transform = transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        
        #if not self.low_data: 
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))

        # My:
        bounding_boxes_file = open(os.path.join(self.root, 'bounding_boxes.txt'))

        parts_file = open(os.path.join(self.root, 'parts/part_locs.txt'))
        #

        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])

        # if not self.low_data: 
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)


        if self.is_train:
            train_test_list = []
            for line in train_val_file:
                train_test_list.append(int(line[:-1].split(' ')[-1]))

            if not self.low_data: 
                train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
                train_file_list_full = [ os.path.join(self.root, 'images', x)  for i, x in zip(train_test_list, img_name_list) if i]

            else:
                print("[INFO] Low data regime training set")

                ld_train_val_file = open(os.path.join(self.root, 'low_data/CUB200/image_list/', 'train_10.txt'))
                                        # train_100.txt, train_50.txt, train_30.txt, train_15.txt,  train_10.txt

                train_file_list = []
                train_file_list_full = []
                ld_label_list = []

                for line in ld_train_val_file:

                    #data_list = []
                    #for line in f.readlines():
                    split_line = line.split(' ') #()
                    target = split_line[-1]
                    path = ' '.join(split_line[:-1])
                    # if not os.path.isabs(path):
                    #     path = os.path.join(self.root, path)
                    target = int(target)
                    #data_list.append((path, target))

                    train_file_list.append(path)
                    train_file_list_full.append(os.path.join(self.root, 'images', path))

                    ld_label_list.append(int(line[:-1].split(' ')[-1])) #- 1)

                    #print("Done")
                    #self.samples = self.parse_data_file(data_list_file)

                    #def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
                    # """Parse file to data list

                    # Args:
                    #     file_name (str): The path of data file
                    #     return (list): List of (image path, class_index) tuples
                    # """
                    # with open(file_name, "r") as f:
                    #     data_list = []
                    #     for line in f.readlines():
                    #         split_line = line.split()
                    #         target = split_line[-1]
                    #         path = ' '.join(split_line[:-1])
                    #         if not os.path.isabs(path):
                    #             path = os.path.join(self.root, path)
                    #         target = int(target)
                    #         data_list.append((path, target))
                    # return data_list
                print("[INFO] Train samples number:" , len(train_file_list), ", and labels number:", len(ld_label_list))

        else:
            test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
            test_file_list_full = [ os.path.join(self.root, 'images', x)  for i, x in zip(train_test_list, img_name_list) if not i]

            print("[INFO] Test samples number:" , len(test_file_list)) #, ", and labels number:", len(ld_label_list))



        ground_truth_bb = False
        if ground_truth_bb:
            # My (bounding boxes):
            bb_list = []

            for line in bounding_boxes_file:
                bb_list_x = line[:-1].split(' ')[-4]
                bb_list_y = line[:-1].split(' ')[-3]
                bb_list_w = line[:-1].split(' ')[-2]
                bb_list_h = line[:-1].split(' ')[-1]

                bb_list.append( [ int(bb_list_x.split('.')[0]),
                                    int(bb_list_y.split('.')[0]),
                                    int(bb_list_w.split('.')[0]),
                                    int(bb_list_h.split('.')[0]) ]
                                    )
            #

        else: # TODO: fix when it's redone twice (first same for train, then same for test) ? actually it's not ?

            saliency_check = False #False
            if saliency_check:

                if is_train: 
                    print("[INFO] Preparing train shape_hw list...")

                    train_shape_hw_list = []

                    for img_name in train_file_list:
                        # remove in further code and in function mask_hw !
                        img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                        shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)
                        #print(img_name)
                        #print(shape_hw_temp)
                        train_shape_hw_list.append(shape_hw_temp)

                else:
                    print("[INFO] Preparing test shape_hw list...")

                    test_shape_hw_list = []

                    for img_name in test_file_list: 
                        # remove in further code and in function mask_hw !
                        img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                        shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)
                        #print(img_name)
                        #print(shape_hw_temp)

                        test_shape_hw_list.append(shape_hw_temp)

                ## train + test together
                # shape_hw_list = []
                # print("[INFO] Preparing shape_hw list...")
                # for img_name in img_name_list:
                #     img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                #     shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)
                #     #print(img_name)
                #     #print(shape_hw_temp)

                #     shape_hw_list.append(shape_hw_temp)


        ground_truth_parts = False
        if ground_truth_parts:
            # My (parts):
            PARTS_NUM = 15

            parts_list = []
            part_t = []
            part_count = 0

            i = 0
            for line in parts_file:
                part_t_raw_x = line[:-1].split(' ')[-3]
                part_t_raw_y = line[:-1].split(' ')[-2]
                part_t_pres = line[:-1].split(' ')[-1]

                part_t.append ( [ int(part_t_pres),
                                    int(part_t_raw_x.split('.')[0]),
                                    int(part_t_raw_y.split('.')[0]) ]
                                    )

                part_count = part_count + 1

                if (part_count >= PARTS_NUM):
                    #if i < 31: print(part_t)
                    parts_list.append( part_t )
                    part_t = []
                    part_count = 0

                i = i + 1
            #


        if self.is_train:
            # My:
            self.train_img = []
            self.train_mask = []


            if ground_truth_bb:
                train_bb_list = [x for i, x in zip(train_test_list, bb_list) if i]


            else:
                if saliency_check:
                    full_ds = True
                    if full_ds:
                        # train_mask_u2_list = []
                        # train_x_u2_list = []
                        # train_y_u2_list = []
                        # train_h_u2_list = []
                        # train_w_u2_list = []

                        #train_shape_hw_list = [x for i, x in zip(train_test_list, shape_hw_list) if i]  ## train + test together
                        train_mask_u2_list, train_x_u2_list, train_y_u2_list, train_h_u2_list, train_w_u2_list = mask_hw(full_ds=full_ds, img_path=train_file_list_full, shape_hw=train_shape_hw_list)
                        #print(train_file_list_full)

                    else:
                        #scipy.misc.imsave('/l/users/20020067/Activities/FGIC/FFVT/Combined/FFVT_my/U2Net/images/img.png', train_img_temp)
                        train_img_path = os.path.join(self.root, 'images', train_file_list)
                        '''
                        mask_u2, x_u2, y_u2, h_u2, w_u2 = mask_hw(full_ds=full_ds, img_path=train_img_path, shape_hw=(h_max, w_max))
                        '''
                        #print(mask_u2, x_u2, y_u2, h_u2, w_u2)
                        # WORKERS !!!!!!!!!!!!!!!!!!!!!!!


            if ground_truth_parts:
                train_parts_list = [x for i, x in zip(train_test_list, parts_list) if i]
                #print(len(train_parts_list))


            print("[INFO] Preparing train files...")
            i = 0
            for train_file in train_file_list[:data_len]:

                train_img_temp = scipy.misc.imread(os.path.join(self.root, 'images', train_file))

                #print("Train file:", train_file)

                h_max = train_img_temp.shape[0] # y
                w_max = train_img_temp.shape[1] # x
                #ch_max = train_img_temp.shape[2]



                if saliency_check:

                    if not ground_truth_bb:
                        mask_u2 = train_mask_u2_list[i]
                        x_u2 = train_x_u2_list[i]
                        y_u2 = train_y_u2_list[i]
                        h_u2 = train_h_u2_list[i]
                        w_u2 = train_w_u2_list[i]

                        # mask_u2, x_u2, y_u2, h_u2, w_u2 = mask_hw(img_path=train_img_path, shape_hw=(h_max, w_max))
                        #print("_________________________________________________")
                        #print(mask_u2, x_u2, y_u2, h_u2, w_u2)
                        # WORKERS !!!!!!!!!!!!!!!!!!!!!!!


                    ## parts:
                    if ground_truth_parts:
                        train_parts = train_parts_list[i] # list of 15 parts with [x, y] center corrdinates
                        #if i < 5: print(len(train_parts))


                        #img_mask = np.zeros((int(h_max), int(w_max))) # Black mask
                        train_img_mask = np.ones((int(h_max), int(w_max)))

                        p_part = 16*3 # padding around center point

                        for part_n in range(len(train_parts)):
                            part = train_parts[part_n]
                            #if i < 5: print(len(part))

                            if part[0] != 0:
                                #if i < 5: print(part[1], part[2])
                                x_min_p = part[1] - p_part
                                if x_min_p < 0:
                                    x_min_p = 0
                                x_max_p = part[1] + p_part
                                if x_max_p > w_max:
                                    x_max_p = w_max

                                y_min_p = part[2] - p_part
                                if y_min_p < 0:
                                    y_min_p = 0
                                y_max_p = part[2] + p_part
                                if y_max_p > h_max:
                                    y_max_p = h_max

                                #img_mask[int(y_min_p):int(y_max_p), int(x_min_p):int(x_max_p)] = 1 # Black mask
                                train_img_mask[int(y_min_p):int(y_max_p), int(x_min_p):int(x_max_p)] = 0
                    
                    else:
                        train_img_mask = mask_u2
                        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                    #


                    ## bbox:
                    if ground_truth_bb:
                        x, y, w, h = train_bb_list[i] # x - distance from top up left (width), y - distance from top up left (height)
                    else:
                        x, y, h, w = x_u2, y_u2, h_u2, w_u2
                        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


                    # padding
                    padding = True
                    if padding:
                        p = 15 # extra space around bbox
                    else:
                        p = 0

                    x_min = x - p 
                    if x_min < 0:
                        x_min = 0
                    x_max = x + w + p
                    if x_max > w_max:
                        x_max = w_max

                    y_min = y - p
                    if y_min < 0:
                        y_min = 0
                    y_max = y + h + p
                    if y_max > h_max:
                        y_max = h_max

                    #print("[SALIENCY]")
                    #
                
                else:
                    x_min = w_max * 0.05 # 0.1
                    x_max = w_max * 0.95 # # 0.9 w -> x

                    y_min = h_max * 0.05 # 0.1
                    y_max = h_max * 0.95 # # 0.9 h -> y                  
                    #



                # CHEEEEEEEEEEEEECKEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
                if h_max <=1:
                    print("bad_h")
                    print(h_max)

                if w_max <=1:
                    print("bad_w")
                    print(w_max)

                if y_min >= y_max:
                    print("bad_y")
                    print("min:", y_min)
                    print("max:", y_max)
                    print("y:", y)
                    print("h:", h)
                    
                    # y_min = 0
                    # y_max = h_max

                if x_min >= x_max:
                    print("bad_x")
                    print("min:", x_min)
                    print("max:", x_max)  
                    print("x:", x)
                    print("w:", w)                                    
                    # x_min = 0
                    # x_max = w_max
                #

                
                # Crop image for bbox:
                if len(train_img_temp.shape) == 3:
                    # Black mask:
                    # for j in range(3):
                    #     train_img_temp[:, :, j] = train_img_temp[:, :, j] * img_mask
                    #

                    #if i < 5: print(train_img_temp.shape)
                    #test_img_temp = test_img_temp[int(y):int(y + h), int(x):int(x + w), :] # h, w, ch
                    train_img_temp = train_img_temp[int(y_min):int(y_max), int(x_min):int(x_max), :] # h, w, ch
                    #if i < 5: print(train_img_temp.shape)
                else:
                    # Black mask:
                    #train_img_temp[:, :] = train_img_temp[:, :] * img_mask
                    #

                    #if i < 5: print(train_img_temp.shape)
                    train_img_temp = train_img_temp[int(y_min):int(y_max), int(x_min):int(x_max)] # h, w
                    #if i < 5: print(train_img_temp.shape)


                if saliency_check:
                    # Crop mask for bbox:
                    train_img_mask = train_img_mask[int(y_min):int(y_max), int(x_min):int(x_max)]
                    
                    if ( (train_img_temp.shape[0] != train_img_mask.shape[0]) or (train_img_temp.shape[1] != train_img_mask.shape[1]) ):
                        print("_____Wrong Index is:", i)
                        print(train_img_temp.shape, train_img_mask.shape)

                    self.train_mask.append(train_img_mask)

                self.train_img.append(train_img_temp)

                i = i+1
            #


            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'images', train_file)) for train_file in
            #                   train_file_list[:data_len]]

            if not self.low_data: 
                self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            else:
                self.train_label = ld_label_list[:data_len]

            self.train_imgname = [x for x in train_file_list[:data_len]]

            # print("Pass 1")

    


        if not self.is_train:
            # My:
            self.test_img = []
            self.test_mask = []

            saliency_check = False #False

            if ground_truth_bb:
                test_bb_list = [x for i, x in zip(train_test_list, bb_list) if not i]

            else:
                if saliency_check:
                    full_ds = True
                    if full_ds:
                        #test_shape_hw_list = [x for i, x in zip(train_test_list, shape_hw_list) if not i]  ## train + test together
                        test_mask_u2_list, test_x_u2_list, test_y_u2_list, test_h_u2_list, test_w_u2_list = mask_hw(full_ds=full_ds, img_path=test_file_list_full, shape_hw=test_shape_hw_list)

                    else:
                        test_img_path = os.path.join(self.root, 'images', test_file_list)
                        mask_u2, x_u2, y_u2, h_u2, w_u2 = mask_hw(full_ds=full_ds, img_path=test_img_path, shape_hw=(h_max, w_max))



            if ground_truth_parts:
                test_parts_list = [x for i, x in zip(train_test_list, parts_list) if not i]


            print("[INFO] Preparing test files...")
            i = 0
            for test_file in test_file_list[:data_len]:
                
                test_img_temp = scipy.misc.imread(os.path.join(self.root, 'images', test_file))

                #print("Test file:", test_file)

                h_max = test_img_temp.shape[0]
                w_max = test_img_temp.shape[1]
                #ch_max = test_img_temp.shape[2]


                if saliency_check:

                    if not ground_truth_bb:
                        mask_u2 = test_mask_u2_list[i]
                        x_u2 = test_x_u2_list[i]
                        y_u2 = test_y_u2_list[i]
                        h_u2 = test_h_u2_list[i]
                        w_u2 = test_w_u2_list[i]


                    ## parts:
                    if ground_truth_parts:
                        test_parts = test_parts_list[i] # list of 15 parts with [x, y] center corrdinates


                        #img_mask = np.zeros((int(h_max), int(w_max))) # Black mask
                        test_img_mask = np.ones((int(h_max), int(w_max)))

                        p_part = 16*3 # padding around center point


                        for part_n in range(len(test_parts)):
                            part = test_parts[part_n]
                            #if i < 5: print(len(part))

                            if part[0] != 0:
                                #if i < 5: print(part[1], part[2])
                                x_min_p = part[1] - p_part
                                if x_min_p < 0:
                                    x_min_p = 0
                                x_max_p = part[1] + p_part
                                if x_max_p > w_max:
                                    x_max_p = w_max

                                y_min_p = part[2] - p_part
                                if y_min_p < 0:
                                    y_min_p = 0
                                y_max_p = part[2] + p_part
                                if y_max_p > h_max:
                                    y_max_p = h_max

                                #img_mask[int(y_min_p):int(y_max_p), int(x_min_p):int(x_max_p)] = 1 # Black mask
                                test_img_mask[int(y_min_p):int(y_max_p), int(x_min_p):int(x_max_p)] = 0
                    
                    else:
                        test_img_mask = mask_u2
                    #


                    ## bbox:
                    if ground_truth_bb:
                        x, y, w, h = test_bb_list[i] # x - distance from top up left (width), y - distance from top up left (height)
                    else:
                        x, y, h, w = x_u2, y_u2, h_u2, w_u2


                    # padding
                    padding = True
                    if padding:
                        p = 15 # extra space around bbox
                    else:
                        p = 0

                    x_min = x - p
                    if x_min < 0:
                        x_min = 0
                    x_max = x + w + p
                    if x_max > w_max:
                        x_max = w_max

                    y_min = y - p
                    if y_min < 0:
                        y_min = 0
                    y_max = y + h + p
                    if y_max > h_max:
                        y_max = h_max
                    
                    #print("[SALIENCY]")
                    #

                else:
                    x_min = w_max * 0.05 # 0.1
                    x_max = w_max * 0.95 # # 0.9 w -> x

                    y_min = h_max * 0.05 # 0.1
                    y_max = h_max * 0.95 # # 0.9 h -> y                   
                    #



                # CHEEEEEEEEEEEEECKEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
                if h_max <=1:
                    print("bad_h")
                    print(h_max)

                if w_max <=1:
                    print("bad_w")
                    print(w_max)

                if y_min >= y_max:
                    print("bad_y")
                    print("min:", y_min)
                    print("max:", y_max)
                    print("y:", y)
                    print("h:", h)
                    
                    # y_min = 0
                    # y_max = h_max

                if x_min >= x_max:
                    print("bad_x")
                    print("min:", x_min)
                    print("max:", x_max)  
                    print("x:", x)
                    print("w:", w)                                    
                    # x_min = 0
                    # x_max = w_max
                #

                
                # Crop image for bbox:
                if len(test_img_temp.shape) == 3:
                    # Black mask:
                    # for j in range(3):
                    #     test_img_temp[:, :, j] = test_img_temp[:, :, j] * img_mask
                    #

                    #test_img_temp = test_img_temp[int(y):int(y + h), int(x):int(x + w), :] # h, w, ch
                    test_img_temp = test_img_temp[int(y_min):int(y_max), int(x_min):int(x_max), :] # h, w, ch
                else:
                    # Black mask:
                    #test_img_temp[:, :] = test_img_temp[:, :] * img_mask
                    #

                    test_img_temp = test_img_temp[int(y_min):int(y_max), int(x_min):int(x_max)] # h, w


                if saliency_check:
                    # Crop mask for bbox:
                    test_img_mask = test_img_mask[int(y_min):int(y_max), int(x_min):int(x_max)]
                    
                    self.test_mask.append(test_img_mask)

                self.test_img.append(test_img_temp)

                i = i+1
            #


            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in
            #                  test_file_list[:data_len]]

            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            self.test_imgname = [x for x in test_file_list[:data_len]]

            # print("Pass test_1")





    def __getitem__(self, index):

        # set_seed = True
        # seed_my = 42
        # if set_seed:
        #     random.seed(seed_my + index)
        #     np.random.seed(seed_my + index)
        #     torch.manual_seed(seed_my + index)
            
        #     #
        #     torch.cuda.manual_seed(seed_my + index)
        #     #


        if self.is_train:
            
            # if self.count < 5: print(self.count)

            #img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]
            # print("Pass 22222")

            # if self.count < 5: print(index)


            # With mask:
            
            saliency_check = False #False
            if saliency_check:
                img, target, imgname, mask = self.train_img[index], self.train_label[index], self.train_imgname[index], self.train_mask[index]
            else:
                img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]


            # if self.count < 5: print(img.shape)
            # if self.count < 5: print(mask.shape)

            #print(imgname)
            # print(img.shape)
            # print(mask.shape)




            rand_crop_im_mask = True # True
            if rand_crop_im_mask:
                #print("Crop_inf")

                #h_max_im = 400
                h_max_img = img.shape[0]

                #w_max_im = 400
                w_max_img = img.shape[1]


                double_crop = True #True # two different crops

                if saliency_check:
                    # portion1side = torch.rand()
                    portion1side = torch.distributions.uniform.Uniform(0.5,0.8).sample([1]) # 0.7,0.95,  0.6,0.8
                    #if index < 10: print(portion1side)

                else:
                    portion1side = torch.distributions.uniform.Uniform(0.5,0.67).sample([1]) # 0.5,0.67 # 0.7,0.95,  0.6,0.8

                    if double_crop:
                        portion1side_2 = torch.distributions.uniform.Uniform(0.67,0.8).sample([1]) # 0.67,0.8 # 0.7,0.95,  0.6,0.8
                        


                h_crop_mid_img = int(h_max_img * portion1side) 
                ##h_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
                #h_crop_mid_img = int(h_max_img * 0.5) #* 0.7) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75), 282 (50 % - 0.7)

                w_crop_mid_img = int(w_max_img * portion1side)
                ##w_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
                #w_crop_mid_img = int(w_max_img * 0.5) #* 0.7) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75), 282 (50 % - 0.7)

                h_crop_min_img = random.randint(0, (h_max_img - h_crop_mid_img)) # 40) #, 400-360) #, h - th)
                w_crop_min_img = random.randint(0, (w_max_img - w_crop_mid_img)) # 40)  #, 400-360) #, w - tw)

                h_crop_max_img = h_crop_mid_img + h_crop_min_img
                w_crop_max_img = w_crop_mid_img + w_crop_min_img


                # Crop image for bbox:
                if len(img.shape) == 3:
                    # Black mask:
                    # for j in range(3):
                    #     test_img_temp[:, :, j] = test_img_temp[:, :, j] * img_mask
                    #

                    #test_img_temp = test_img_temp[int(y):int(y + h), int(x):int(x + w), :] # h, w, ch
                    img_crop = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img), :] # h, w, ch
                else:
                    # Black mask:
                    #test_img_temp[:, :] = test_img_temp[:, :] * img_mask
                    #

                    img_crop = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)] # h, w

                    img_crop = np.stack([img_crop] * 3, 2)

                if saliency_check:
                    # Crop mask for bbox:
                    mask_crop = mask[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)]


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



            full_rand_crop_im_mask = False
            if full_rand_crop_im_mask:
                #print("Crop_inf")

                #h_max_im = 400
                h_max_img = img.shape[0]

                #w_max_im = 400
                w_max_img = img.shape[1]

                # portion1side = torch.rand()
                portion1side = torch.distributions.uniform.Uniform(0.95,1.0).sample([1])

                h_crop_mid_img = int(h_max_img * portion1side) 
                ##h_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
                #h_crop_mid_img = int(h_max_img * 0.7) #* 0.7) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75), 282 (50 % - 0.7)
                if index < 10: print(portion1side)

                w_crop_mid_img = int(w_max_img * portion1side)
                ##w_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
                #w_crop_mid_img = int(w_max_img * 0.7) #* 0.7) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75), 282 (50 % - 0.7)


                h_crop_min_img = random.randint(0, (h_max_img - h_crop_mid_img)) # 40) #, 400-360) #, h - th)
                w_crop_min_img = random.randint(0, (w_max_img - w_crop_mid_img)) # 40)  #, 400-360) #, w - tw)

                h_crop_max_img = h_crop_mid_img + h_crop_min_img
                w_crop_max_img = w_crop_mid_img + w_crop_min_img


                # Crop image for bbox:
                if len(img.shape) == 3:

                    img = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img), :] # h, w, ch
                else:

                    img = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)] # h, w

                    img = np.stack([img] * 3, 2)

                # Crop mask for bbox:
                mask = mask[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)]






            if len(img.shape) == 2:
                # print("222222222")
                # print(img.shape)
                img = np.stack([img] * 3, 2)
                # print(mask.shape)
                # print(img.shape)
            
            # if rand_crop_im_mask:
            #     if len(img_crop.shape) == 2:
            #         img_crop = np.stack([img_crop] * 3, 2)


            # try:
            #     torch.set_printoptions(profile="full")
            #     np.printoptions(threshold=np.inf)                
            #     print(img.tile)
            #     print("tile???")
            # except:
            #     pass


            img = Image.fromarray(img, mode='RGB')

            if rand_crop_im_mask:
                img_crop = Image.fromarray(img_crop, mode='RGB')
                if double_crop:
                    img_crop2 = Image.fromarray(img_crop2, mode='RGB')
                    
            #print("Pass 3333333333333333333")


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


            flip_mask_as_image = True #True # if False - turn on RandomHorizontalFlip in data_utils !!!
            
            flipped = False # temp
           
            if self.transform is not None:
                #img = self.transform(img)

                if not flip_mask_as_image: # normal
                    img = self.transform(img)
                    
                    if rand_crop_im_mask:
                        img_crop = self.transform(img_crop)

                        if double_crop:
                            img_crop2 = self.transform(img_crop2)

                else:
                    if random.random() < 0.5:
                        flipped = False
                        #print("[INFO]: No Flip")
                        img = self.transform(img)

                        if rand_crop_im_mask:
                            img_crop = self.transform(img_crop)

                            if double_crop:
                                img_crop2 = self.transform(img_crop2)

                    else: # MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        flipped = True
                        #print("[INFO]: Flip convert")

                        transform_img_flip = transforms.Compose([
                                        #transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR),
                                        #transforms.Resize((560, 560), Image.BILINEAR), #transFG 600
                                        #transforms.RandomCrop((args.img_size, args.img_size)),
                                        
                                         # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                        transforms.Resize((224, 224),Image.BILINEAR), # my for bbox
                                        #transforms.Resize((400, 400),Image.BILINEAR), # my for bbox
                                        #transforms.Resize((448, 448),Image.BILINEAR), # my for bbox

                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add (FFVT)
                                        
                                        transforms.RandomHorizontalFlip(p=1.0), # !!! FLIPPING in dataset.py !!!
                                        #transforms.RandomHorizontalFlip(), # !!! FLIPPING in dataset.py !!!

                                        transforms.ToTensor(),
                                        #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])

                        img = transform_img_flip(img)

                        if rand_crop_im_mask:
                            img_crop = transform_img_flip(img_crop)
                            if double_crop:
                                img_crop2 = transform_img_flip(img_crop2)

            # My:
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



            ### Mask:
            # if self.count < 5: print("mask before", mask)
            # if self.count < 5: print("mask before", mask.shape)

            crop_mask = False # True # if False - ? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            mid_val = False

            if crop_mask:
                #print("Crop_inf")

                #h_max_im = 400
                h_max_im = mask.shape[0]

                #w_max_im = 400
                w_max_im = mask.shape[1]


                #h_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
                h_crop_mid = int(h_max_im * 0.84) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75)

                #w_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
                w_crop_mid = int(w_max_im * 0.84) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75)


                #cropped = np.zeros_like(mask)
                cropped = np.ones_like(mask)

                if mid_val:
                    #cropped = cropped * 0.26 # 0.25 ? # 0.5 ?
                    #cropped = cropped * 0.33 # 0.165 # (for 0.25), 0.33 # (for 0.2) , 0.26 # 0.25 ? # 0.5 ?
                    cropped = cropped * 0.125 # (for 0.2)

                # print("Before:")
                # print(mask)
                # print(cropped)
                # print(mask.shape)
                # print(cropped.shape)

                h_crop_min = random.randint(0, (h_max_im - h_crop_mid)) # 40) #, 400-360) #, h - th)
                w_crop_min = random.randint(0, (w_max_im - w_crop_mid)) # 40)  #, 400-360) #, w - tw)

                h_crop_max = h_crop_mid + h_crop_min
                w_crop_max = w_crop_mid + w_crop_min

                # print("Min hw:", h_crop_min, w_crop_min)
                # print("Max hw:", h_crop_max, w_crop_max)

                #test_img_temp = test_img_temp[int(y_min):int(y_max), int(x_min):int(x_max)] # h, w
                cropped[int(h_crop_min):int(h_crop_max), int(w_crop_min):int(w_crop_max)] = 0
                
                mask = mask + cropped

                # WORKERS !!!!!!!!!!!

                # print("After:")
                # print(mask)
                # print(cropped)
                # print(mask.shape)
                # print(cropped.shape)

                if mid_val:
                    mask[mask > 1.1] = 1
                else:
                    mask[mask > 1] = 1 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                # print("Mask After:")
                # print(mask)


                # import time
                # time.sleep(3)


            #mask = int(mask)
            #mask = np.stack([mask] * 3, 2)
            #mask = Image.fromarray(mask, mode='RGB')


            if saliency_check:

                mask = (mask * 255).astype(np.uint8)
                # if self.count < 5: print("mask 255", mask)
                mask = Image.fromarray(mask, mode='L')
                # if self.count < 5: print("mask tensor before", mask)
                
                if rand_crop_im_mask:
                    mask_crop = (mask_crop * 255).astype(np.uint8)
                    mask_crop = Image.fromarray(mask_crop, mode='L')

                if index < 10:
                    # # import time
                    from torchvision.utils import save_image
                    mask_tem = transforms.ToTensor()(mask)
                    img_name = ("test/mask_bef" + str(index) + ".png")
                    save_image( mask_tem, img_name)

                    if rand_crop_im_mask:
                        mask_tem_crop = transforms.ToTensor()(mask_crop)
                        img_name_crop = ("test/mask_bef_crop" + str(index) + ".png")
                        save_image( mask_tem_crop, img_name_crop)


                if not flip_mask_as_image: # normal
                    transform_mask = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.ToPILImage(), #(mode='1'),

                        # non-overlapped and size 224:
                        transforms.Resize((14,14), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST                 

                        # non-overlapped and size 400:
                        #transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        
                        # non-overlapped and size 448:
                        #transforms.Resize((28,28), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        

                        # non-overlapped and size 400:
                        # transforms.Resize((29,29), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        # #transforms.Resize((27,27), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        # transforms.CenterCrop((25,25)), # !!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                        # overlapped patch 12 and size 400:
                        #transforms.Resize((33,33), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        
                        transforms.ToTensor()])
                else:
                    if flipped:
                        transform_mask = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.ToPILImage(), #(mode='1'),
                            transforms.RandomHorizontalFlip(p=1.0),
                            
                            # non-overlapped and size 224:
                            transforms.Resize((14,14), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST                 

                            # non-overlapped and size 400:
                            #transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            
                            # non-overlapped and size 448:
                            #transforms.Resize((28,28), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            
                        
                            # non-overlapped and size 400:
                            # transforms.Resize((29,29), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            # #transforms.Resize((27,27), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            # transforms.CenterCrop((25,25)), # !!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                            # overlapped patch 12 and size 400:
                            #transforms.Resize((33,33), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            
                            transforms.ToTensor()])
                    else:
                        transform_mask = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.ToPILImage(), #(mode='1'),

                            # non-overlapped and size 224:
                            transforms.Resize((14,14), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST                 

                            # non-overlapped and size 400:
                            #transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            
                            # non-overlapped and size 448:
                            #transforms.Resize((28,28), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            


                            # non-overlapped and size 400:
                            # transforms.Resize((29,29), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            # #transforms.Resize((27,27), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            # transforms.CenterCrop((25,25)), # !!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                            # overlapped patch 12 and size 400:
                            #transforms.Resize((33,33), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST

                            transforms.ToTensor()])


                mask = transform_mask(mask)
                
                if rand_crop_im_mask:
                    mask_crop = transform_mask(mask_crop)


                if index < 10:
                    # import time
                    # #print(img.shape)
                    # #mask_img = transforms.ToPILImage()(mask)
                    # print("next mask")

                    img_name = ("test/mask_aft" + str(index) + ".png")
                    save_image( mask, img_name)

                    # time.sleep(5)

                    if rand_crop_im_mask:
                        img_name_crop = ("test/mask_aft_crop" + str(index) + ".png")
                        save_image( mask_crop, img_name_crop)


                # mask1 = transforms.Resize((400, 400))(mask) # 400 - args.img_size
                # mask2 = transforms.ToTensor()(mask1),

                # mask = mask2
                # if self.count < 5: print(mask.shape)
                # if self.count < 5: print(mask)


                mask = torch.flatten(mask)

                if rand_crop_im_mask:
                    mask_crop = torch.flatten(mask_crop)


                # if self.count < 5: print(mask.shape)
                # if self.count < 5: print(mask)

                # mask_patch = np.ones((int(25), int(25)))

                # for part_n in range(len(train_parts)):
                #     part = train_parts[part_n]
                #     #if i < 5: print(len(part))

                #     if part[0] != 0:
                #         #if i < 5: print(part[1], part[2])
                #         x_min_p = part[1] - p_part
                #         if x_min_p < 0:
                #             x_min_p = 0
                #         x_max_p = part[1] + p_part
                #         if x_max_p > w_max:
                #             x_max_p = w_max

                #         y_min_p = part[2] - p_part
                #         if y_min_p < 0:
                #             y_min_p = 0
                #         y_max_p = part[2] + p_part
                #         if y_max_p > h_max:
                #             y_max_p = h_max

                #         #img_mask[int(y_min_p):int(y_max_p), int(x_min_p):int(x_max_p)] = 1 # Black mask
                #         train_img_mask[int(y_min_p):int(y_max_p), int(x_min_p):int(x_max_p)] = 0


                # Convert mask 400x400 to 25x25 (1 px = 1 patch of 16x16)

                # mask = mask[0, :, :]
                # #if self.count < 5: print(mask.shape)

                # mask_25 = torch.ones(625)
                # #mask_clear = mask >= 0.9


                # mask_16_list = []

                # for px in range(400):
                #     for py in range(400):
                #         x_t = 16 * px 
                #         y_t = 16 * py 

                #         mask_16 = mask[0:16,0:16]


                # for patch_id in range(625): # cuz 16x16
                #     mask_16_temp = mask_16[patch_id]
                #     indices = mask_16_temp.nonzero(as_tuple=True)

                #     if indices :
                #         mask_25[patch_id] = 0

                    # if mask[]:
                    #     mask_25_new = torch.where(mask_25 > 0, mask_25, 1.)
                    
                #type(mask)
                #print(mask.shape)

                # self.count = self.count + 1


        else:

            #img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]


            saliency_check = False #False
            if saliency_check:
                # With mask:
                img, target, imgname, mask = self.test_img[index], self.test_label[index], self.test_imgname[index], self.test_mask[index]
                #
            else:
                img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]

            #print(imgname)
            # print(img.shape)
            # print(mask.shape)

            if len(img.shape) == 2:
                # print("222222222")
                # print(imgname)
                # print(img.shape)                
                img = np.stack([img] * 3, 2)
                # print(mask.shape)
                # print(img.shape)
                
            # try:
            #     torch.set_printoptions(profile="full")
            #     np.printoptions(threshold=np.inf)                
            #     print(img.tile)
            #     print("tile???")
            # except:
            #     pass            
            
            img = Image.fromarray(img, mode='RGB')

            #print("Pass 3333333333333333333 test")

            if self.transform is not None:
                img = self.transform(img)


            if saliency_check:

                ### Mask:
                mask = (mask * 255).astype(np.uint8)
                mask = Image.fromarray(mask, mode='L')
                
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                transform_mask = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(), #(mode='1'),

                    # non-overlapped and size 224:
                    transforms.Resize((14,14), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST                 

                    # non-overlapped and size 400:
                    #transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    
                    # non-overlapped and size 448:
                    #transforms.Resize((28,28), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    


                    # non-overlapped and size 400:
                    # transforms.Resize((29,29), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    # #transforms.Resize((27,27), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    # transforms.CenterCrop((25,25)), # !!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    # overlapped patch 12 and size 400:
                    #transforms.Resize((33,33), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST

                    transforms.ToTensor()])

                mask = transform_mask(mask)

                mask = torch.flatten(mask)



        if self.is_train:

            if saliency_check:
                if double_crop:
                    return img, img_crop, img_crop, target, mask, mask_crop
                else:
                    return img, img_crop, target, mask, mask_crop
            else:
                if double_crop:
                    return img, img_crop, img_crop, target
                else:
                    return img, img_crop, target

        else:
            if saliency_check:
                return img, target, mask
            else:
                return img, target



    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)





class CarsDataset(Dataset):

    #def __init__(self, mat_anno, data_dir, car_names, cleaned=None, transform=None):
    #def __init__(self, mat_anno, data_dir, car_names, root, is_train=True, transform=None, cleaned=None, data_len=None):
    def __init__(self, root, is_train=True, transform=None, cleaned=None, data_len=None):

        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.base_folder = "images"
        self.root = root
        self.data_dir = join(self.root, self.base_folder)

        self.is_train = is_train
        self.transform = transform

        self.low_data = True
        saliency_check = False #False

        self.classes = join(self.root,'cars_meta.mat')
        self.car_names = scipy.io.loadmat(self.classes)['class_names']
        self.car_names = np.array(self.car_names[0])


        if not self.low_data: 
            # train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
            # train_file_list_full = [ os.path.join(self.root, 'images', x)  for i, x in zip(train_test_list, img_name_list) if i]
            
            self.full_data_set = io.loadmat(os.path.join(self.root,'cars_annos.mat'))

            self.car_annotations = self.full_data_set['annotations']
            self.car_annotations = self.car_annotations[0]

            if cleaned is not None:
                cleaned_annos = []
                print("Cleaning up data set (only take pics with rgb chans)...")
                clean_files = np.loadtxt(cleaned, dtype=str)
                for c in self.car_annotations:
                    if c[-1][0] in clean_files:
                        cleaned_annos.append(c)
                self.car_annotations = cleaned_annos



            if self.is_train:
                print("[INFO] Preparing train shape_hw list...")


                if saliency_check:
                    train_shape_hw_list = []

                    train_file_list_full = []

                # image_name, target_class = self._flat_breed_images[index]
                # image_path = join(self.images_folder, image_name)
                # image = Image.open(image_path).convert('RGB')

                train_file_list = []


                # for img_name in train_file_list:
                    #     # remove in further code and in function mask_hw !
                    # img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                    # shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)
                    # #print(img_name)
                    # #print(shape_hw_temp)
                    # train_shape_hw_list.append(shape_hw_temp)


                #for image_name, target_class in self._flat_breed_images:
                for image_name in self.car_annotations:

                    # img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
                    # image = Image.open(img_name).convert('RGB')

                    #img_name = join(self.images_folder, image_name)
                    img_name = join(self.data_dir, image_name[-1][0])
                    #print(img_name)


                    #img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                    img_temp = scipy.misc.imread(os.path.join(img_name))


                    if saliency_check:
                        shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)
                        
                        
                        
                        if (shape_hw_temp[0] > 500) or (shape_hw_temp[1] > 500):
                            if shape_hw_temp[0] > shape_hw_temp[1]:
                                max500 = shape_hw_temp[0] / 500
                            else:
                                max500 = shape_hw_temp[1] / 500
                                

                            shape_hw_temp[0] = int( shape_hw_temp[0] // max500 )
                            shape_hw_temp[1] = int( shape_hw_temp[1] // max500 )


                    
                        #print(shape_hw_temp)
                        train_shape_hw_list.append(shape_hw_temp)

                        train_file_list_full.append(img_name)

                    train_file_list.append(image_name[-1][0])

            else:
                print("[INFO] Preparing test shape_hw list...")

                if saliency_check:
                    test_shape_hw_list = []

                    test_file_list_full = []

                test_file_list = []

                # for img_name in test_file_list: 
                #     # remove in further code and in function mask_hw !
                #     img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                #     shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)
                #     #print(img_name)
                #     #print(shape_hw_temp)
                #     test_shape_hw_list.append(shape_hw_temp)

                for image_name in self.car_annotations:
                    # img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
                    # image = Image.open(img_name).convert('RGB')

                    #img_name = join(self.images_folder, image_name)
                    img_name = join(self.data_dir, image_name[-1][0])

                    #img_temp = scipy.misc.imread(os.path.join(self.root, 'images', img_name))
                    img_temp = scipy.misc.imread(os.path.join(img_name))


                    if saliency_check:

                        shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)

                        if (shape_hw_temp[0] > 500) or (shape_hw_temp[1] > 500):
                            if shape_hw_temp[0] > shape_hw_temp[1]:
                                max500 = shape_hw_temp[0] / 500
                            else:
                                max500 = shape_hw_temp[1] / 500
                                
                            shape_hw_temp[0] = int( shape_hw_temp[0] // max500 )
                            shape_hw_temp[1] = int( shape_hw_temp[1] // max500 )

                        #print(img_name)
                        #print(shape_hw_temp)
                        test_shape_hw_list.append(shape_hw_temp)

                        test_file_list_full.append(img_name)

                    test_file_list.append(image_name[-1][0])


        
        else:

            if self.is_train:

                print("[INFO] Low data regime training set")

                train_file_list = []
                ld_label_list = []

                #data_list_file = os.path.join(root, self.image_list[list_name])
                ld_train_val_file = open(os.path.join(self.root, 'low_data/', 'train_10.txt'))
                                        # train_100.txt, train_50.txt, train_30.txt, train_15.txt,  train_10.txt

                for line in ld_train_val_file:

                    split_line = line.split(' ') #()
                    target = split_line[-1]
                    path = ' '.join(split_line[:-1])

                    target = int(target)

                    train_file_list.append(path)

                    ld_label_list.append(int(line[:-1].split(' ')[-1])) #- 1)


                print("[INFO] Train samples number:" , len(train_file_list), ", and labels number:", len(ld_label_list))


                # self.samples = self.parse_data_file(data_list_file)
                # self.classes = classes
                # self.class_to_idx = {cls: idx
                #                     for idx, cls in enumerate(self.classes)}
                # self.loader = default_loader
                # self.data_list_file = data_list_file

                # def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
                #     """Parse file to data list

                #     Args:
                #         file_name (str): The path of data file
                #         return (list): List of (image path, class_index) tuples
                #     """
                #     with open(file_name, "r") as f:
                #         data_list = []
                #         for line in f.readlines():
                #             split_line = line.split()
                #             target = split_line[-1]
                #             path = ' '.join(split_line[:-1])
                            
                #             if not os.path.isabs(path):
                #                 #path = os.path.join(self.root, path)
                #                 #path = os.path.join("/l/users/20020067/Datasets/CUB_200_2011/CUB_200_2011/CUB_200_2011/images/", path)
                #                 path = os.path.join("/l/users/20020067/Datasets/Stanford Cars/Stanford Cars/images/", path)

                #             print(path)

                #             target = int(target)
                #             data_list.append((path, target))
                #     return data_list


                #     path, target = self.samples[index]
                #     img = self.loader(path)
                #     if self.transform is not None:
                #         img = self.transform(img)
                #     if self.target_transform is not None and target is not None:
                #         target = self.target_transform(target)
                #     return img, target


                # elif split == 'unlabeled_train':
                #     data_list_file = os.path.join(root, "image_list/unlabeled_" + str(label_ratio) + ".txt")
                #     # if not os.path.exists(data_list_file):
                #     train_list_name = 'train' + str(label_ratio)
                #     full_list_name = 'train'
                #     assert train_list_name in self.image_list
                #     assert full_list_name in self.image_list
                #     train_list_file = os.path.join(root, self.image_list[train_list_name])
                #     full_list_file = os.path.join(root, self.image_list[full_list_name])
                #     train_list = read_list_from_file(train_list_file)
                #     full_list = read_list_from_file(full_list_file)
                #     unlabel_list = list(set(full_list) - set(train_list))
                #     save_list_to_file(data_list_file, unlabel_list)
            
            else:

                print("[INFO] Low data regime test set")

                test_file_list = []
                ld_label_list = []

                #data_list_file = os.path.join(root, self.image_list['test'])
                ld_train_val_file = open(os.path.join(self.root, 'low_data/', 'test.txt'))
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
            # My:
            self.train_img = []
            self.train_mask = []


            if saliency_check:
                full_ds = True
                if full_ds:
                    # train_mask_u2_list = []
                    # train_x_u2_list = []
                    # train_y_u2_list = []
                    # train_h_u2_list = []
                    # train_w_u2_list = []

                    #train_shape_hw_list = [x for i, x in zip(train_test_list, shape_hw_list) if i]  ## train + test together
                    train_mask_u2_list, train_x_u2_list, train_y_u2_list, train_h_u2_list, train_w_u2_list = mask_hw(full_ds=full_ds, img_path=train_file_list_full, shape_hw=train_shape_hw_list)
                    #print(train_file_list_full)

                else:
                    #scipy.misc.imsave('/l/users/20020067/Activities/FGIC/FFVT/Combined/FFVT_my/U2Net/images/img.png', train_img_temp)
                    train_img_path = os.path.join(self.root, 'images', train_file_list)
                    '''
                    mask_u2, x_u2, y_u2, h_u2, w_u2 = mask_hw(full_ds=full_ds, img_path=train_img_path, shape_hw=(h_max, w_max))
                    '''
                    #print(mask_u2, x_u2, y_u2, h_u2, w_u2)
                    # WORKERS !!!!!!!!!!!!!!!!!!!!!!!


            print("[INFO] Preparing train files...")
            i = 0
            for train_file in train_file_list[:data_len]:

                train_img_temp = scipy.misc.imread(os.path.join(self.data_dir, train_file))

                #print("Train file:", train_file)

                h_max = train_img_temp.shape[0] # y
                w_max = train_img_temp.shape[1] # x
                #ch_max = train_img_temp.shape[2]



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
                    if i < 10:
                        print("Normal:", train_img_temp.shape[0], train_img_temp.shape[1])

                        img_name = ("test/img_normal_tr" + str(i) + ".png")
                        Image.fromarray(train_img_temp, mode='RGB').save(img_name)



                if saliency_check:

                    mask_u2 = train_mask_u2_list[i]
                    x_u2 = train_x_u2_list[i]
                    y_u2 = train_y_u2_list[i]
                    h_u2 = train_h_u2_list[i]
                    w_u2 = train_w_u2_list[i]

                    # mask_u2, x_u2, y_u2, h_u2, w_u2 = mask_hw(img_path=train_img_path, shape_hw=(h_max, w_max))
                    #print("_________________________________________________")
                    #print(mask_u2, x_u2, y_u2, h_u2, w_u2)
                    # WORKERS !!!!!!!!!!!!!!!!!!!!!!!


                    train_img_mask = mask_u2
                    #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                #


                    x, y, h, w = x_u2, y_u2, h_u2, w_u2
                    #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


                    # padding
                    padding = True
                    if padding:
                        p = 15 # extra space around bbox
                    else:
                        p = 0

                    x_min = x - p 
                    if x_min < 0:
                        x_min = 0
                    x_max = x + w + p
                    if x_max > w_max:
                        x_max = w_max

                    y_min = y - p
                    if y_min < 0:
                        y_min = 0
                    y_max = y + h + p
                    if y_max > h_max:
                        y_max = h_max

                    #print("[SALIENCY]")
                    #
                
                else:
                    x_min = w_max * 0.05 # 0.1
                    x_max = w_max * 0.95 # # 0.9 w -> x

                    y_min = h_max * 0.05 # 0.1
                    y_max = h_max * 0.95 # # 0.9 h -> y                  
                    #



                # CHEEEEEEEEEEEEECKEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
                if h_max <=1:
                    print("bad_h")
                    print(h_max)

                if w_max <=1:
                    print("bad_w")
                    print(w_max)

                if y_min >= y_max:
                    print("bad_y")
                    print("min:", y_min)
                    print("max:", y_max)
                    print("y:", y)
                    print("h:", h)
                    
                    # y_min = 0
                    # y_max = h_max

                if x_min >= x_max:
                    print("bad_x")
                    print("min:", x_min)
                    print("max:", x_max)  
                    print("x:", x)
                    print("w:", w)                                    
                    # x_min = 0
                    # x_max = w_max
                #

                
                # Crop image for bbox:
                if len(train_img_temp.shape) == 3:
                    # Black mask:
                    # for j in range(3):
                    #     train_img_temp[:, :, j] = train_img_temp[:, :, j] * img_mask
                    #

                    #if i < 5: print(train_img_temp.shape)
                    #test_img_temp = test_img_temp[int(y):int(y + h), int(x):int(x + w), :] # h, w, ch
                    train_img_temp = train_img_temp[int(y_min):int(y_max), int(x_min):int(x_max), :] # h, w, ch
                    #if i < 5: print(train_img_temp.shape)
                else:
                    # Black mask:
                    #train_img_temp[:, :] = train_img_temp[:, :] * img_mask
                    #

                    #if i < 5: print(train_img_temp.shape)
                    train_img_temp = train_img_temp[int(y_min):int(y_max), int(x_min):int(x_max)] # h, w
                    #if i < 5: print(train_img_temp.shape)


                if saliency_check:
                    # Crop mask for bbox:
                    train_img_mask = train_img_mask[int(y_min):int(y_max), int(x_min):int(x_max)]
                    
                    if ( (train_img_temp.shape[0] != train_img_mask.shape[0]) or (train_img_temp.shape[1] != train_img_mask.shape[1]) ):
                        print("_____Wrong Index is:", i)
                        print(train_img_temp.shape, train_img_mask.shape)

                    self.train_mask.append(train_img_mask)

                self.train_img.append(train_img_temp)

                i = i+1
            #


            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'images', train_file)) for train_file in
            #                   train_file_list[:data_len]]

            if not self.low_data: 
                self.train_label = [  ( torch.from_numpy(np.array( x[-2][0][0].astype(np.float32) )).long() - 1 )  for x in self.car_annotations ][:data_len]
            else:
                self.train_label = ld_label_list[:data_len]

            self.train_imgname = [x for x in train_file_list[:data_len]]

            # print("Pass 1")

    


        else:
            # My:
            self.test_img = []
            self.test_mask = []


            if saliency_check:
                full_ds = True
                if full_ds:
                    #test_shape_hw_list = [x for i, x in zip(train_test_list, shape_hw_list) if not i]  ## train + test together
                    test_mask_u2_list, test_x_u2_list, test_y_u2_list, test_h_u2_list, test_w_u2_list = mask_hw(full_ds=full_ds, img_path=test_file_list_full, shape_hw=test_shape_hw_list)

                else:
                    test_img_path = os.path.join(self.root, 'images', test_file_list)
                    mask_u2, x_u2, y_u2, h_u2, w_u2 = mask_hw(full_ds=full_ds, img_path=test_img_path, shape_hw=(h_max, w_max))


            print("[INFO] Preparing test files...")
            i = 0
            for test_file in test_file_list[:data_len]:
                
                test_img_temp = scipy.misc.imread(os.path.join(self.data_dir, test_file))

                #print("Test file:", test_file)

                h_max = test_img_temp.shape[0]
                w_max = test_img_temp.shape[1]
                #ch_max = test_img_temp.shape[2]





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
                    



                if saliency_check:

                    mask_u2 = test_mask_u2_list[i]
                    x_u2 = test_x_u2_list[i]
                    y_u2 = test_y_u2_list[i]
                    h_u2 = test_h_u2_list[i]
                    w_u2 = test_w_u2_list[i]


                    test_img_mask = mask_u2
                    #


                    x, y, h, w = x_u2, y_u2, h_u2, w_u2


                    # padding
                    padding = True
                    if padding:
                        p = 15 # extra space around bbox
                    else:
                        p = 0

                    x_min = x - p
                    if x_min < 0:
                        x_min = 0
                    x_max = x + w + p
                    if x_max > w_max:
                        x_max = w_max

                    y_min = y - p
                    if y_min < 0:
                        y_min = 0
                    y_max = y + h + p
                    if y_max > h_max:
                        y_max = h_max
                    
                    #print("[SALIENCY]")
                    #

                else:
                    x_min = w_max * 0.05 # 0.1
                    x_max = w_max * 0.95 # # 0.9 w -> x

                    y_min = h_max * 0.05 # 0.1
                    y_max = h_max * 0.95 # # 0.9 h -> y                   
                    #



                # CHEEEEEEEEEEEEECKEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
                if h_max <=1:
                    print("bad_h")
                    print(h_max)

                if w_max <=1:
                    print("bad_w")
                    print(w_max)

                if y_min >= y_max:
                    print("bad_y")
                    print("min:", y_min)
                    print("max:", y_max)
                    print("y:", y)
                    print("h:", h)
                    
                    # y_min = 0
                    # y_max = h_max

                if x_min >= x_max:
                    print("bad_x")
                    print("min:", x_min)
                    print("max:", x_max)  
                    print("x:", x)
                    print("w:", w)                                    
                    # x_min = 0
                    # x_max = w_max
                #

                
                # Crop image for bbox:
                if len(test_img_temp.shape) == 3:
                    # Black mask:
                    # for j in range(3):
                    #     test_img_temp[:, :, j] = test_img_temp[:, :, j] * img_mask
                    #

                    #test_img_temp = test_img_temp[int(y):int(y + h), int(x):int(x + w), :] # h, w, ch
                    test_img_temp = test_img_temp[int(y_min):int(y_max), int(x_min):int(x_max), :] # h, w, ch
                else:
                    # Black mask:
                    #test_img_temp[:, :] = test_img_temp[:, :] * img_mask
                    #

                    test_img_temp = test_img_temp[int(y_min):int(y_max), int(x_min):int(x_max)] # h, w


                if saliency_check:
                    # Crop mask for bbox:
                    test_img_mask = test_img_mask[int(y_min):int(y_max), int(x_min):int(x_max)]
                    
                    self.test_mask.append(test_img_mask)

                self.test_img.append(test_img_temp)

                if (i % 1000 == 0):
                    print(i)

                i = i+1
            #


            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in
            #                  test_file_list[:data_len]]

            
            if not self.low_data: 
                self.test_label = [  ( torch.from_numpy(np.array( x[-2][0][0].astype(np.float32) )).long() - 1 )  for x in self.car_annotations ][:data_len]
            else:
                self.test_label = ld_label_list[:data_len]            
            
            
            self.test_imgname = [x for x in test_file_list[:data_len]]

            # print("Pass test_1")





    def __len__(self):
        #return len(self.car_annotations)
        
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)




    def __getitem__(self, index):

        saliency_check = False #False


        if self.is_train:
            
            # if self.count < 5: print(self.count)

            #img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]
            # print("Pass 22222")

            # if self.count < 5: print(index)


            # With mask:
            
            if saliency_check:
                img, target, imgname, mask = self.train_img[index], self.train_label[index], self.train_imgname[index], self.train_mask[index]
            else:
                img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]


            # if self.count < 5: print(img.shape)
            # if self.count < 5: print(mask.shape)

            #print(imgname)
            # print(img.shape)
            # print(mask.shape)




            rand_crop_im_mask = True # True
            if rand_crop_im_mask:
                #print("Crop_inf")

                #h_max_im = 400
                h_max_img = img.shape[0]

                #w_max_im = 400
                w_max_img = img.shape[1]


                double_crop = True #True # two different crops

                if saliency_check:
                    # portion1side = torch.rand()
                    portion1side = torch.distributions.uniform.Uniform(0.5,0.8).sample([1]) # 0.7,0.95,  0.6,0.8
                    #if index < 10: print(portion1side)

                else:
                    portion1side = torch.distributions.uniform.Uniform(0.5,0.8).sample([1]) # 0.5,0.67 # 0.67,0.8 # 0.7,0.95,  0.6,0.8

                    if double_crop:
                        portion1side_2 = torch.distributions.uniform.Uniform(0.8,0.9).sample([1]) # 0.67,0.8 # 0.8,0.9 # 0.7,0.95,  0.6,0.8
                        


                h_crop_mid_img = int(h_max_img * portion1side) 
                ##h_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
                #h_crop_mid_img = int(h_max_img * 0.5) #* 0.7) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75), 282 (50 % - 0.7)

                w_crop_mid_img = int(w_max_img * portion1side)
                ##w_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
                #w_crop_mid_img = int(w_max_img * 0.5) #* 0.7) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75), 282 (50 % - 0.7)

                h_crop_min_img = random.randint(0, (h_max_img - h_crop_mid_img)) # 40) #, 400-360) #, h - th)
                w_crop_min_img = random.randint(0, (w_max_img - w_crop_mid_img)) # 40)  #, 400-360) #, w - tw)

                h_crop_max_img = h_crop_mid_img + h_crop_min_img
                w_crop_max_img = w_crop_mid_img + w_crop_min_img


                # Crop image for bbox:
                if len(img.shape) == 3:
                    # Black mask:
                    # for j in range(3):
                    #     test_img_temp[:, :, j] = test_img_temp[:, :, j] * img_mask
                    #

                    #test_img_temp = test_img_temp[int(y):int(y + h), int(x):int(x + w), :] # h, w, ch
                    img_crop = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img), :] # h, w, ch
                else:
                    # Black mask:
                    #test_img_temp[:, :] = test_img_temp[:, :] * img_mask
                    #

                    img_crop = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)] # h, w

                    img_crop = np.stack([img_crop] * 3, 2)

                if saliency_check:
                    # Crop mask for bbox:
                    mask_crop = mask[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)]


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
                # print("222222222")
                # print(img.shape)
                img = np.stack([img] * 3, 2)
                # print(mask.shape)
                # print(img.shape)
            
            # if rand_crop_im_mask:
            #     if len(img_crop.shape) == 2:
            #         img_crop = np.stack([img_crop] * 3, 2)


            # try:
            #     torch.set_printoptions(profile="full")
            #     np.printoptions(threshold=np.inf)                
            #     print(img.tile)
            #     print("tile???")
            # except:
            #     pass



            img = (img).astype(np.uint8)

            img = Image.fromarray(img, mode='RGB')


            if rand_crop_im_mask:
                img_crop = (img_crop).astype(np.uint8)
                
                img_crop = Image.fromarray(img_crop, mode='RGB')

                if double_crop:
                    img_crop2 = (img_crop2).astype(np.uint8)

                    img_crop2 = Image.fromarray(img_crop2, mode='RGB')
                    
            #print("Pass 3333333333333333333")


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


            flip_mask_as_image = True # if False - turn on RandomHorizontalFlip in data_utils !!!
            
            flipped = False # temp
           
            if self.transform is not None:
                #img = self.transform(img)

                if not flip_mask_as_image: # normal
                    img = self.transform(img)
                    
                    if rand_crop_im_mask:
                        img_crop = self.transform(img_crop)

                        if double_crop:
                            img_crop2 = self.transform(img_crop2)

                else:
                    if random.random() < 0.5:
                        flipped = False
                        #print("[INFO]: No Flip")
                        img = self.transform(img)

                        if rand_crop_im_mask:
                            img_crop = self.transform(img_crop)

                            if double_crop:
                                img_crop2 = self.transform(img_crop2)

                    else: # MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        flipped = True
                        #print("[INFO]: Flip convert")

                        transform_img_flip = transforms.Compose([
                                        #transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR),
                                        #transforms.Resize((560, 560), Image.BILINEAR), #transFG 600
                                        #transforms.RandomCrop((args.img_size, args.img_size)),
                                        
                                         # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                        transforms.Resize((224, 224),Image.BILINEAR), # my for bbox
                                        #transforms.Resize((448, 448),Image.BILINEAR), # my for bbox

                                        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add (FFVT)
                                        AutoAugImageNetPolicy(),
                                        
                                        #transforms.RandomHorizontalFlip(p=1.0), # !!! FLIPPING in dataset.py !!!
                                        transforms.RandomHorizontalFlip(), # !!! FLIPPING in dataset.py !!!

                                        transforms.ToTensor(),
                                        #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])

                        img = transform_img_flip(img)

                        if rand_crop_im_mask:
                            img_crop = transform_img_flip(img_crop)
                            if double_crop:
                                img_crop2 = transform_img_flip(img_crop2)

            # My:
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



            if saliency_check:

                mask = (mask * 255).astype(np.uint8)
                # if self.count < 5: print("mask 255", mask)
                mask = Image.fromarray(mask, mode='L')
                # if self.count < 5: print("mask tensor before", mask)
                
                if rand_crop_im_mask:
                    mask_crop = (mask_crop * 255).astype(np.uint8)
                    mask_crop = Image.fromarray(mask_crop, mode='L')

                if index < 10:
                    # # import time
                    from torchvision.utils import save_image
                    mask_tem = transforms.ToTensor()(mask)
                    img_name = ("test/mask_bef" + str(index) + ".png")
                    save_image( mask_tem, img_name)

                    if rand_crop_im_mask:
                        mask_tem_crop = transforms.ToTensor()(mask_crop)
                        img_name_crop = ("test/mask_bef_crop" + str(index) + ".png")
                        save_image( mask_tem_crop, img_name_crop)


                if not flip_mask_as_image: # normal
                    transform_mask = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.ToPILImage(), #(mode='1'),

                        # non-overlapped and size 224:
                        transforms.Resize((14,14), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST                 

                        # non-overlapped and size 400:
                        #transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        
                        # non-overlapped and size 448:
                        #transforms.Resize((28,28), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        

                        # non-overlapped and size 400:
                        # transforms.Resize((29,29), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        # #transforms.Resize((27,27), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        # transforms.CenterCrop((25,25)), # !!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                        # overlapped patch 12 and size 400:
                        #transforms.Resize((33,33), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        
                        transforms.ToTensor()])
                else:
                    if flipped:
                        transform_mask = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.ToPILImage(), #(mode='1'),
                            transforms.RandomHorizontalFlip(p=1.0),
                            
                            # non-overlapped and size 224:
                            transforms.Resize((14,14), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST                 

                            # non-overlapped and size 400:
                            #transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            
                            # non-overlapped and size 448:
                            #transforms.Resize((28,28), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            
                        
                            # non-overlapped and size 400:
                            # transforms.Resize((29,29), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            # #transforms.Resize((27,27), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            # transforms.CenterCrop((25,25)), # !!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                            # overlapped patch 12 and size 400:
                            #transforms.Resize((33,33), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            
                            transforms.ToTensor()])
                    else:
                        transform_mask = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.ToPILImage(), #(mode='1'),

                            # non-overlapped and size 224:
                            transforms.Resize((14,14), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST                 

                            # non-overlapped and size 400:
                            #transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            
                            # non-overlapped and size 448:
                            #transforms.Resize((28,28), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            


                            # non-overlapped and size 400:
                            # transforms.Resize((29,29), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            # #transforms.Resize((27,27), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                            # transforms.CenterCrop((25,25)), # !!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                            # overlapped patch 12 and size 400:
                            #transforms.Resize((33,33), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST

                            transforms.ToTensor()])


                mask = transform_mask(mask)
                
                if rand_crop_im_mask:
                    mask_crop = transform_mask(mask_crop)


                if index < 10:
                    # import time
                    # #print(img.shape)
                    # #mask_img = transforms.ToPILImage()(mask)
                    # print("next mask")

                    img_name = ("test/mask_aft" + str(index) + ".png")
                    save_image( mask, img_name)

                    # time.sleep(5)

                    if rand_crop_im_mask:
                        img_name_crop = ("test/mask_aft_crop" + str(index) + ".png")
                        save_image( mask_crop, img_name_crop)


                # mask1 = transforms.Resize((400, 400))(mask) # 400 - args.img_size
                # mask2 = transforms.ToTensor()(mask1),

                # mask = mask2
                # if self.count < 5: print(mask.shape)
                # if self.count < 5: print(mask)


                mask = torch.flatten(mask)

                if rand_crop_im_mask:
                    mask_crop = torch.flatten(mask_crop)


                # if self.count < 5: print(mask.shape)
                # if self.count < 5: print(mask)

                # mask_patch = np.ones((int(25), int(25)))

                # for part_n in range(len(train_parts)):
                #     part = train_parts[part_n]
                #     #if i < 5: print(len(part))

                #     if part[0] != 0:
                #         #if i < 5: print(part[1], part[2])
                #         x_min_p = part[1] - p_part
                #         if x_min_p < 0:
                #             x_min_p = 0
                #         x_max_p = part[1] + p_part
                #         if x_max_p > w_max:
                #             x_max_p = w_max

                #         y_min_p = part[2] - p_part
                #         if y_min_p < 0:
                #             y_min_p = 0
                #         y_max_p = part[2] + p_part
                #         if y_max_p > h_max:
                #             y_max_p = h_max

                #         #img_mask[int(y_min_p):int(y_max_p), int(x_min_p):int(x_max_p)] = 1 # Black mask
                #         train_img_mask[int(y_min_p):int(y_max_p), int(x_min_p):int(x_max_p)] = 0


                # Convert mask 400x400 to 25x25 (1 px = 1 patch of 16x16)

                # mask = mask[0, :, :]
                # #if self.count < 5: print(mask.shape)

                # mask_25 = torch.ones(625)
                # #mask_clear = mask >= 0.9


                # mask_16_list = []

                # for px in range(400):
                #     for py in range(400):
                #         x_t = 16 * px 
                #         y_t = 16 * py 

                #         mask_16 = mask[0:16,0:16]


                # for patch_id in range(625): # cuz 16x16
                #     mask_16_temp = mask_16[patch_id]
                #     indices = mask_16_temp.nonzero(as_tuple=True)

                #     if indices :
                #         mask_25[patch_id] = 0

                    # if mask[]:
                    #     mask_25_new = torch.where(mask_25 > 0, mask_25, 1.)
                    
                #type(mask)
                #print(mask.shape)

                # self.count = self.count + 1


        else:

            #img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]


            saliency_check = False #False
            if saliency_check:
                # With mask:
                img, target, imgname, mask = self.test_img[index], self.test_label[index], self.test_imgname[index], self.test_mask[index]
                #
            else:
                img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]

            #print(imgname)
            # print(img.shape)
            # print(mask.shape)

            if len(img.shape) == 2:
                # print("222222222")
                # print(imgname)
                # print(img.shape)                
                img = np.stack([img] * 3, 2)
                # print(mask.shape)
                # print(img.shape)
                
            # try:
            #     torch.set_printoptions(profile="full")
            #     np.printoptions(threshold=np.inf)                
            #     print(img.tile)
            #     print("tile???")
            # except:
            #     pass            
            
            img = (img).astype(np.uint8)
            
            img = Image.fromarray(img, mode='RGB')

            #print("Pass 3333333333333333333 test")

            if self.transform is not None:
                img = self.transform(img)


            if saliency_check:

                ### Mask:
                mask = (mask * 255).astype(np.uint8)
                mask = Image.fromarray(mask, mode='L')
                
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                transform_mask = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(), #(mode='1'),

                    # non-overlapped and size 224:
                    transforms.Resize((14,14), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST                 

                    # non-overlapped and size 400:
                    #transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    
                    # non-overlapped and size 448:
                    #transforms.Resize((28,28), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    


                    # non-overlapped and size 400:
                    # transforms.Resize((29,29), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    # #transforms.Resize((27,27), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    # transforms.CenterCrop((25,25)), # !!!!!!!!!!!!!!!!!!!!!! MANUAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    # overlapped patch 12 and size 400:
                    #transforms.Resize((33,33), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST

                    transforms.ToTensor()])

                mask = transform_mask(mask)

                mask = torch.flatten(mask)



        if self.is_train:

            if saliency_check:
                if double_crop:
                    return img, img_crop, img_crop, target, mask, mask_crop
                else:
                    return img, img_crop, target, mask, mask_crop
            else:
                if double_crop:
                    return img, img_crop, img_crop, target
                else:
                    return img, img_crop, target

        else:
            if saliency_check:
                return img, target, mask
            else:
                return img, target


        # img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        # image = Image.open(img_name).convert('RGB')
        # car_class = self.car_annotations[idx][-2][0][0]
        # car_class = torch.from_numpy(np.array(car_class.astype(np.float32))).long() - 1
        # assert car_class < 196
        
        # if self.transform:
        #     image = self.transform(image)

        # # return image, car_class, img_name
        # return image, car_class




    def map_class(self, id):
        id = np.ravel(id)
        ret = self.car_names[id - 1][0][0]
        return ret

    def show_batch(self, img_batch, class_batch):

        for i in range(img_batch.shape[0]):
            ax = plt.subplot(1, img_batch.shape[0], i + 1)
            title_str = self.map_class(int(class_batch[i]))
            img = np.transpose(img_batch[i, ...], (1, 2, 0))
            ax.imshow(img)
            ax.set_title(title_str.__str__(), {'fontsize': 5})
            plt.tight_layout()



def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images
    
def find_classes(classes_file):
    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)





class dogs(Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'dog'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self,
                 root,
                 train=True,
                 cropped=False,
                 transform=None,
                 target_transform=None,
                 download=False):

        # self.root = join(os.path.expanduser(root), self.folder)
        self.root = root
        self.train = train
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx)
                                        for box in self.get_boxes(join(self.annotations_folder, annotation))]
                                        for annotation, idx in split]
            self._flat_breed_annotations = sum(self._breed_annotations, [])

            self._flat_breed_images = [(annotation+'.jpg', idx) for annotation, box, idx in self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation+'.jpg', idx) for annotation, idx in split]

            self._flat_breed_images = self._breed_images

        self.classes = ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.cropped:
            image = image.crop(self._flat_breed_annotations[index][1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes

    def load_split(self):
        if self.train:
            # split = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            # labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
            split = scipy.io.loadmat(join(self.root, 'splits/train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'splits/train_list.mat'))['labels']
        else:
            # split = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            # labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']
            split = scipy.io.loadmat(join(self.root, 'splits/test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'splits/test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0]-1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)"%(len(self._flat_breed_images), len(counts.keys()), float(len(self._flat_breed_images))/float(len(counts.keys()))))

        return counts





class NABirds(Dataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'nabirds/images'

    def __init__(self, root, train=True, transform=None):
        dataset_path = os.path.join(root, 'nabirds')
        self.root = root
        self.loader = default_loader
        self.train = train
        self.transform = transform

        image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        # Since the raw labels are non-continuous, map them to new ones
        self.label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        # Load in the train / test split
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # Load in the class data
        self.class_names = load_class_names(dataset_path)
        self.class_hierarchy = load_hierarchy(dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = self.label_map[sample.target]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}

def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names

def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents





class INat2017(VisionDataset):
    """`iNaturalist 2017 <https://github.com/visipedia/inat_comp/blob/master/2017/README.md>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'train_val_images/'
    file_list = {
        'imgs': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val_images.tar.gz',
                 'train_val_images.tar.gz',
                 '7c784ea5e424efaec655bd392f87301f'),
        'annos': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val2017.zip',
                  'train_val2017.zip',
                  '444c835f6459867ad69fcb36478786e7')
    }

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(INat2017, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            if not (os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))
                    and os.path.exists(os.path.join(self.root, self.file_list['annos'][1]))):
                print('Downloading...')
                self._download()
            print('Extracting...')
            extract_archive(os.path.join(self.root, self.file_list['imgs'][1]))
            extract_archive(os.path.join(self.root, self.file_list['annos'][1]))
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        anno_filename = split + '2017.json'
        with open(os.path.join(self.root, anno_filename), 'r') as fp:
            all_annos = json.load(fp)

        self.annos = all_annos['annotations']
        self.images = all_annos['images']

    def __getitem__(self, index):
        path = os.path.join(self.root, self.images[index]['file_name'])
        target = self.annos[index]['category_id']

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.base_folder))

    def _download(self):
        for url, filename, md5 in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
            if not check_integrity(os.path.join(self.root, filename), md5):
                raise RuntimeError("File not found or corrupted.")





class soyloc():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.mode = 'train' if is_train else 'test'
        anno_txt_file = open(os.path.join(self.root, 'anno',self.mode+'.txt'))
        self.labels = []
        self.imgs_name = []
        for line in anno_txt_file:
            self.imgs_name.append(line.strip().split(' ')[0])
            self.labels.append(int(line.strip().split(' ')[1])-1)
        #self.imgs = [scipy.misc.imread(os.path.join(self.root, 'images', img_name)) for img_name in self.imgs_name ]
        
    def __getitem__(self, index): 
        img_path = os.path.join(self.root, 'images', self.imgs_name[index])
        img, target, imgname = scipy.misc.imread(img_path), self.labels[index], self.imgs_name[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs_name)






class cotton():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.mode = 'train' if is_train else 'test'
        anno_txt_file = open(os.path.join(self.root, 'anno',self.mode+'.txt'))
        self.labels = []
        self.imgs_name = []
        for line in anno_txt_file:
            self.imgs_name.append(line.strip().split(' ')[0])
            self.labels.append(int(line.strip().split(' ')[1])-1)
        #self.imgs = [scipy.misc.imread(os.path.join(self.root, 'images', img_name)) for img_name in self.imgs_name ]
        
    def __getitem__(self, index): 
        img_path = os.path.join(self.root, 'images', self.imgs_name[index])
        img, target, imgname = scipy.misc.imread(img_path), self.labels[index], self.imgs_name[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs_name)





class FGVC_aircraft():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        train_img_path = os.path.join(self.root, 'data', 'images')
        test_img_path = os.path.join(self.root, 'data', 'images')
        train_label_file = open(os.path.join(self.root, 'data', 'train.txt'))
        test_label_file = open(os.path.join(self.root, 'data', 'test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = scipy.misc.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

        else:
            img, target = scipy.misc.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_img_label)
        else:
            return len(self.test_img_label)
