#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)
import torch.utils.data
from backbone.iresnet import iresnet50
from torch.nn import DataParallel
from margin.ArcMarginProduct import ArcMarginProduct
from margin.MultiMarginProduct import MultiMarginProduct
from margin.CosineMarginProduct import CosineMarginProduct
from margin.InnerProduct import InnerProduct
from utility.log import init_log
from utility.hook import feature_hook
from dataset.casia_webface import CASIAWebFace
from dataset.agedb import AgeDB30
from dataset.cfp import CFP_FP
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import time
from evaluation.eval_lfw import evaluation_10_fold, getFeatureFromTorch
import numpy as np
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy
import random


def set_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def cosine_loss(l , h):
    l = l.view(l.size(0), -1)
    h = h.view(h.size(0), -1)
    return torch.mean(1.0 - F.cosine_similarity(l, h))


def train(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logging = init_log(save_dir)
    _print = logging.info

    # dataset loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    # validation dataset
    trainset = CASIAWebFace(args.train_root, args.train_file_list, args.down_size, transform=transform, equal=args.equal)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8, drop_last=False)
    

    # define backbone and margin layer
    net = iresnet50(attention_type=args.mode)


    # Margin
    if args.margin_type == 'ArcFace':
        margin = ArcMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'MultiMargin':
        margin = MultiMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'CosFace':
        margin = CosineMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size, m=0.25)
    elif args.margin_type == 'Softmax':
        margin = InnerProduct(args.feature_dim, trainset.class_nums)
    elif args.margin_type == 'SphereFace':
        pass
    else:
        print(args.margin_type, 'is not available!')


    # Load Teacher Model and Freeze
    aux_net = iresnet50(attention_type=args.mode)
    aux_margin = deepcopy(margin)

    
    # Load Pretrained Teacher
    net_ckpt = torch.load(args.teacher_path, map_location='cpu')['net_state_dict']
    aux_net.load_state_dict(net_ckpt)
    aux_margin.load_state_dict(torch.load(args.teacher_path.replace('_net.ckpt', '_margin.ckpt'), map_location='cpu')['net_state_dict'])
    
    for param in aux_net.parameters():
        param.requires_grad = False
    
    for param in aux_margin.parameters():
        param.requires_grad = False


    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[18000, 28000, 36000, 44000], gamma=0.1)

    if multi_gpus:
        net = DataParallel(net).to(device)
        aux_net = DataParallel(aux_net).to(device)
        margin = DataParallel(margin).to(device)
        aux_margin = DataParallel(aux_margin).to(device)
            
    else:
        net = net.to(device)
        aux_net = aux_net.to(device)
        margin = margin.to(device)
        aux_margin = aux_margin.to(device)

    total_iters = 0


    # Hook
    if args.distill_type == 'F_SKD':
        target_layer = 'feature_target_block'
        distill_func = cosine_loss

    elif args.distill_type == 'A_SKD':
        assert args.mode == 'cbam'
        target_layer = 'attention_target_all'
        distill_func = cosine_loss

    else:
        raise('Select Proper Distill Type')
    
    HR_manager = feature_hook(aux_net, multi_gpu=multi_gpus, target_layer=target_layer)    
    LR_manager = feature_hook(net, multi_gpu=multi_gpus, target_layer=target_layer)


    # Run
    GOING = True
    while GOING:
        # train model and freeze batchnorm for main network
        net.train()        
        margin.train()
        aux_net.eval()
        aux_margin.eval()

        since = time.time()
        for data in tqdm(trainloader):            
            HR_img, LR_img, label = data[0].to(device), data[1].to(device), data[2].to(device)                  
            
            HR_manager.attention = []
            LR_manager.attention = []
            
            with torch.no_grad():
                HR_logits = aux_net(HR_img)
                HR_out = aux_margin(HR_logits, label)
            
            LR_logits = net(LR_img)
            LR_out = margin(LR_logits, label)
            
            # Recognition Loss
            cri_loss = criterion(LR_out, label)
            
            # Distill Loss
            distill_loss = 0.
            for HR_feat, LR_feat in zip(HR_manager.attention, LR_manager.attention):
                if args.distill_type == 'A_SKD' : 
                    distill_loss += (distill_func(LR_feat[0], HR_feat[0]) + distill_func(LR_feat[1], HR_feat[1])) / len(HR_manager.attention) / 2
                elif args.distill_type == 'F_SKD':
                    distill_loss += distill_func(LR_feat, HR_feat) / len(HR_manager.attention)
                else:
                    raise('Select Proper Distill Type')

            # Total Loss
            total_loss = cri_loss + distill_loss * args.distill_param

            # Optim
            optimizer_ft.zero_grad()
            total_loss.backward()
            optimizer_ft.step()

            # Clear Features
            HR_manager.attention = []
            LR_manager.attention = []
            
            # print train information
            if total_iters % 100 == 0:
                # current training accuracy
                _, predict = torch.max(LR_out.data, 1)
                total = label.size(0)

                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                _print("Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters, total_loss.item(), correct/total, time_cur, exp_lr_scheduler.get_lr()[0]))

            # save model
            if total_iters % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(total_iters)
                _print(msg)
                if multi_gpus:
                    net_state_dict = net.module.state_dict()
                    margin_state_dict = margin.module.state_dict()
                else:
                    net_state_dict = net.state_dict()
                    margin_state_dict = margin.state_dict()
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                    
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': net_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_net.ckpt' % total_iters))
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % total_iters))
            
            
            exp_lr_scheduler.step()
            
            # Stop
            if total_iters == 47000:
                GOING = False
                break
        
            # Next Iterations
            total_iters += 1
            
    
    # Remove Hook
    HR_manager.remove_hook()
    LR_manager.remove_hook()  

    # Save Last Epoch
    msg = 'Saving checkpoint: {}'.format(total_iters)
    _print(msg)
    if multi_gpus:
        net_state_dict = net.module.state_dict()
        margin_state_dict = margin.module.state_dict()
    else:
        net_state_dict = net.state_dict()
        margin_state_dict = margin.state_dict()
        
    torch.save({
        'iters': total_iters,
        'net_state_dict': net_state_dict},
        os.path.join(save_dir, 'last_net.ckpt'))
    torch.save({
        'iters': total_iters,
        'net_state_dict': margin_state_dict},
        os.path.join(save_dir, 'last_margin.ckpt'))


    # test dataset
    net.eval()
    margin.eval()
    
    print('Evaluation on LFW, AgeDB-30. CFP')
    os.makedirs(os.path.join(args.save_dir, 'result'), exist_ok=True)
    
    if args.down_size == 1:
        eval_list = [112, 56, 28, 14]
    elif args.down_size == 0:
        eval_list = [112]
    else: 
        eval_list = [args.down_size]
        
    average_age = 0.
    average_cfp = 0.
    for down_size in eval_list:
        agedbdataset = AgeDB30(args.agedb_test_root, args.agedb_file_list, down_size, transform=transform)
        cfpfpdataset = CFP_FP(args.cfpfp_test_root, args.cfpfp_file_list, down_size, transform=transform)
        
        agedbloader = torch.utils.data.DataLoader(agedbdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
        cfpfploader = torch.utils.data.DataLoader(cfpfpdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

        # test model on AgeDB30
        getFeatureFromTorch(os.path.join(args.save_dir, 'result/cur_agedb30_result.mat'), net, device, agedbdataset, agedbloader)
        age_accs = evaluation_10_fold(os.path.join(args.save_dir, 'result/cur_agedb30_result.mat'))
        print('Evaluation Result on AgeDB-30 %dX - %.2f' %(down_size, np.mean(age_accs) * 100))

        # test model on CFP-FP
        getFeatureFromTorch(os.path.join(args.save_dir, 'result/cur_cfpfp_result.mat'), net, device, cfpfpdataset, cfpfploader)
        cfp_accs = evaluation_10_fold(os.path.join(args.save_dir, 'result/cur_cfpfp_result.mat'))
        print('Evaluation Result on CFP-ACC %dX - %.2f' %(down_size, np.mean(cfp_accs) * 100))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--data_dir', type=str, default='Face/')
    parser.add_argument('--down_size', type=int, default=1) # 1 : all type, 0 : high, others : low
    parser.add_argument('--save_dir', type=str, default='checkpoint/', help='model save dir')
    parser.add_argument('--mode', type=str, default='ir', help='attention type', choices=['ir', 'cbam'])
    parser.add_argument('--margin_type', type=str, default='CosFace', help='ArcFace, CosFace, SphereFace, MultiMargin, Softmax')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512')
    parser.add_argument('--scale_size', type=float, default=30.0, help='scale size')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--save_freq', type=int, default=10000, help='save frequency')
    parser.add_argument('--equal', type=lambda x: x.lower()=='true', default=True)
    parser.add_argument('--gpus', type=str, default='5', help='model prefix')
    parser.add_argument('--seed', type=int, default=1)
    
    parser.add_argument('--distill_param', type=float, default=1.0, help='hyperparams for distillation loss')
    parser.add_argument('--distill_type', type=str, default='F_SKD', help='distillation types', choices=['F_SKD', 'A_SKD'])
    parser.add_argument('--teacher_path', type=str, default='checkpoint/teacher/iresnet50-ir/last_net.ckpt')
    args = parser.parse_args()


    # Path
    args.train_root = os.path.join(args.data_dir, 'faces_webface_112x112/image')
    args.train_file_list = os.path.join(args.data_dir, 'faces_webface_112x112/train.list')
    args.lfw_test_root = os.path.join(args.data_dir, 'evaluation/lfw')
    args.lfw_file_list = os.path.join(args.data_dir, 'evaluation/lfw.txt')
    args.agedb_test_root = os.path.join(args.data_dir, 'evaluation/agedb_30')
    args.agedb_file_list = os.path.join(args.data_dir, 'evaluation/agedb_30.txt')
    args.cfpfp_test_root = os.path.join(args.data_dir, 'evaluation/cfp_fp')
    args.cfpfp_file_list = os.path.join(args.data_dir, 'evaluation/cfp_fp.txt')


    # Seed
    set_random_seed(args.seed)
    
    # Run    
    train(args)