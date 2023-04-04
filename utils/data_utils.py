import logging

import torch

from torchvision import transforms, datasets
from .dataset import *
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from PIL import Image
from .autoaugment import AutoAugImageNetPolicy
import os

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset== "CRC":
        train_transform=transforms.Compose([
                                    transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR), # if no saliency
                                    transforms.RandomCrop((args.img_size, args.img_size)), # if no saliency

                                    transforms.RandomHorizontalFlip(), # !!! FLIPPING in dataset.py !!!
                                    transforms.RandomVerticalFlip(), # from air

                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    #transforms.Normalize([0.650, 0.472, 0.584], [0.158, 0.164, 0.143]), # for CRC (our manual)
                                    ])

        test_transform=transforms.Compose([
                                    transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR), # if no saliency
                                    transforms.CenterCrop((args.img_size, args.img_size)), # if no saliency
                                
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    #transforms.Normalize([0.650, 0.472, 0.584], [0.158, 0.164, 0.143]), # for CRC (our manual)
                                    ])

        trainset = eval(args.dataset)(root=args.data_root, is_train=True, transform=train_transform, 
                                        vanilla=args.vanilla, split=args.split, gan=args.gan, gan_ratio=args.gan_ratio)
        testset = eval(args.dataset)(root=args.data_root, is_train=False, transform=test_transform, 
                                        vanilla=args.vanilla, split=args.split, gan=args.gan, gan_ratio=args.gan_ratio)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              
                              #num_workers=4,
                              num_workers=20, #20,

                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,

                             #num_workers=4,
                             num_workers=20, #20,

                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
