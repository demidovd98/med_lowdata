# coding=utf-8
from __future__ import absolute_import, division, print_function

# WnB:
#import wandb
#wandb.init(project="fgvc_combined_ld_refine", entity="demidovd98")
#

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta
import time

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

import torch.nn.functional as F
#import timm
from torchvision import models

# SAM:
from SAM.models.classifier import Classifier
from SAM.models.method import SAM

from SAM.src.utils import load_network, load_data


logger = logging.getLogger(__name__)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    if args.feature_fusion:
        config.feature_fusion=True
    config.num_token = args.num_token
    
    if args.dataset == "cifar10":
        num_classes=10
    elif args.dataset == 'CRC':
        num_classes = 8
        args.pretrained_dir = "output/CRC_rn50_vanilla_stepAuto_224x192_bs32_lr001_10k_20sp3_checkpoint.bin"


    print("[INFO] Pre-trained model used: ", args.pretrained_dir)


    if args.split is not None:
        print(f"[INFO] A {args.split} split is used")

    if args.vanilla:
        print("[INFO] A vanilla (unmodified) model is used")

    SAM_check = False #True

    if SAM_check:
        backbone_name = 'resnet50'

        #pretrained_path = '~/.torch/models/moco_v2_800ep_pretrain.pth.tar'
        pretrained_path = None
        projector_dim = 1024 # or 2048 ?

        # Initialize model
        network, feature_dim = load_network(backbone_name)
        model = SAM(network=network, backbone=backbone_name, projector_dim=projector_dim,
                        class_num=num_classes, pretrained=True, pretrained_path=pretrained_path)#.to(args.device)
        classifier = Classifier(2048, num_classes)#.to(args.device)   #2048/num of bilinear 2048*16
        # classifier.classifier_layer.apply(init_weights)

    else:
        #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True) 

        model = models.resnet50(pretrained=True) #, num_classes=200)
        ##model = models.resnet50(pretrained=False) #, num_classes=200)
        #model = models.resnet18(pretrained=True) #, num_classes=200)
        
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
        #model.fc.apply(init_weights) ?
        
        print("[INFO] A pre-trained ResNet-50 model is used")


    model.load_state_dict(torch.load(args.pretrained_dir, map_location=torch.device('cpu')))


    if SAM_check:
        model.to(args.device)
        model.eval()

        classifier.to(args.device)
        #print(model)
        #print(classifier)
    
        print("backbone params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6 / 2))
        print("classifier params: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1e6))
    
    else:
        model.to(args.device)
        model.eval()

        num_params = count_parameters(model)

        #save_model(args, model)
        #print(model)

        #logger.info("{}".format(config))
        logger.info("Training parameters %s", args)
        logger.info("Total Parameter: \t%2.1fM" % num_params)
        print(num_params)

    if SAM_check:
        return args, model, classifier, num_classes
    else:
        return args, model, num_classes


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step, classifier=None):

    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    SAM_check = False #True
    if SAM_check:
        model.eval()
        classifier.eval()
    else:
        model.eval()
        
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):

        #wandb.log({"step": step})

        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch

        with torch.no_grad():

            SAM_check = False #True
            if SAM_check:
                feat_labeled = model(x)[0]
                logits = classifier(feat_labeled.cuda())[0] #feat_labeled/bp_out_feat
            else:
                logits = model(x)

            eval_loss = loss_fct(logits, y)
            #eval_loss = loss_fct(logits.view(-1, 200), y.view(-1))
            #eval_loss = eval_loss.mean() # for contrastive learning!!! # transFG

            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0 )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0 )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    #from torchmetrics.classification import MulticlassConfusionMatrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    y_pred = all_preds
    y_true = all_label

    # constant for classes
    classes = ('Tumor', 'Stroma', 'Lymph', 'Complex', 'Debris',
            'Mucosa', 'Adipose', 'Empty')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)

    matrix_name = "matrix_" + str(args.name) + '_output.png'
    plt.savefig(matrix_name)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    print("Valid Accuracy:", accuracy)
    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)

    #wandb.log({"acc_test": accuracy})

    return accuracy


def train(args, model, classifier=None, num_classes=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
        
    best_step=0
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    test_loader = get_loader(args)

    # Prepare optimizer and scheduler

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    '''
    SAM_check = False #True
    if SAM_check:
        optimizer = torch.optim.SGD(model.parameters(), 
                    lr= args.learning_rate, 
                    momentum=0.9, 
                    weight_decay=args.weight_decay, 
                    nesterov= True, # True
                    )
        
        milestones = [ int(args.num_steps * 0.5),
                    int(args.num_steps * 0.75),
                    int(args.num_steps * 0.90),
                    int(args.num_steps * 0.95),
                    int(args.num_steps * 1.0) ]
        print("[INFO] Milestones for the lr scheduler are:", milestones)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    else:
        lr_ratio = 10.0
        optimizer = torch.optim.SGD([
                    {'params': model.parameters()},
                    {'params': classifier.parameters(), 'lr': args.learning_rate * lr_ratio}, ], 
                    lr= args.learning_rate, 
                    momentum=0.9, 
                    weight_decay=args.weight_decay, 
                    nesterov=True)
        milestones = [6000, 12000, 18000, 24000, 30000]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    '''

    t_total = args.num_steps 
    '''
    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20
    '''

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    start_time = time.time()
    logger.info("***** Running testing *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    # logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             args.train_batch_size * args.gradient_accumulation_steps * (
    #                 torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    #model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    #losses = AverageMeter()
    global_step, best_acc = 0, 0
                    
    SAM_check = False #True
    if SAM_check:
        accuracy = valid(args, model, writer, test_loader, global_step, classifier)
    else:
        accuracy = valid(args, model, writer, test_loader, global_step)

    if args.local_rank in [-1, 0]:
        writer.close()
    end_time = time.time()

    '''
    split_sep = args.split.split('_') #()
    print(split_sep)

    res_split = "Split: " +  str(args.split) + "\n"
    res_name = "Name: " +  str(args.name) + "\n"
    res_steps = "Steps: " + str(global_step) + "\n"
    res_bestAcc = "Best valid accuracy: " + str(best_acc) + "\n"
    res_bestStep = "Best valid accuracy in step: " + str(best_step) + "\n"
    res_time = "Total Training Time: " + str(((end_time - start_time) / 3600)) + "\n"
    res_trainAcc = "Train accuracy so far: " + str(train_accuracy) + "\n"
    res_args = "Training parameters: " + str(args) + "\n"
    res_newLine = str("\n")

    # f = open("./models/statistics.txt", "a")
    with open("./results/" + split_sep[0] + "_all.txt", "a") as f:
        # text_train = "Epoch: " + str(epoch_print) + ", " + "Train Loss: " + str(loss_print) + ", " + "Train Accuracy: " + str(acc_print) + "\n"
        f.write(res_newLine)
        f.write(res_split)
        f.write(res_name)
        f.write(res_steps)
        f.write(res_bestAcc)
        f.write(res_bestStep)
        f.write(res_time)
        f.write(res_trainAcc)
        f.write(res_args)
        f.write(res_newLine)
    # f.close()
    '''
    logger.info("Best Accuracy: \t%f" % accuracy)
    logger.info("Total Testing Time: \t%f" % ((end_time - start_time) / 3600))
    logger.info("End Testing!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "soyloc","cotton", "CUB", "dogs","cars","air", "CRC"], default="cotton",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")

    #parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
    parser.add_argument("--pretrained_dir", type=str, default="",
                        help="Where to search for pretrained ViT models.")

    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--resize_size", default=600, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_token", default=12, type=int,
                        help="the number of selected token in each layer, 12 for soy.loc, cotton and cub, 24 for dog.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--feature_fusion', action='store_true',
                        help="Whether to use feature fusion")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    
    parser.add_argument('--vanilla', action='store_true',
                        help="Whether to use the vanilla model")
    parser.add_argument("--split", required=True,
                        choices=["1i", "1p", "2", "3", "4", "5", "10", "15", "30", "50", "100", 
                                "1_sp1", "3_sp1", "5_sp1", "10_sp1", "20_sp1", "30_sp1", "40_sp1", "50_sp1", "60_sp1", "70_sp1", "75_sp1", "80_sp1", "90_sp1", "100_sp1",
                                "1_sp2", "3_sp2", "5_sp2", "10_sp2", "20_sp2", "30_sp2", "40_sp2", "50_sp2", "60_sp2", "70_sp2", "75_sp2", "80_sp2", "90_sp2", "100_sp2",
                                "1_sp3", "3_sp3", "5_sp3", "10_sp3", "20_sp3", "30_sp3", "40_sp3", "50_sp3", "60_sp3", "70_sp3", "75_sp3", "80_sp3", "90_sp3", "100_sp3",
                                "1_sp4", "3_sp4", "5_sp4", "10_sp4", "20_sp4", "30_sp4", "40_sp4", "50_sp4", "75_sp4", "100_sp4",
                                "1_sp5", "3_sp5", "5_sp5", "10_sp5", "20_sp5", "30_sp5", "40_sp5", "50_sp5", "75_sp5", "100_sp5",
                                ],
                        help="Name of the split")
    
    parser.add_argument('--gan', action='store_true',
                        help="Whether to use GAN generated images")
    parser.add_argument("--gan_ratio", default=0.2, type=float,
                        help="Amount of generated data as a percentage of the real data.")

    #parser.add_argument('--data_root', type=str, default='./data') # Originall
    parser.add_argument('--data_root', type=str, default='/l/users/20020067/Datasets/CRC_colorectal_cancer_histology') # CRC Medical

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    
    args = parser.parse_args()
    
    #args.data_root = '{}/{}'.format(args.data_root, args.dataset)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    SAM_check = False #True
    if SAM_check:
        args, model, classifier, num_classes = setup(args)

        #wandb.watch(model)

        # Training
        train(args, model, classifier, num_classes)
    else:    

        args, model, num_classes = setup(args)

        #wandb.watch(model)
        #torch.autograd.set_detect_anomaly(True)

        # Training
        train(args, model, num_classes=num_classes)


if __name__ == "__main__":
    main()
