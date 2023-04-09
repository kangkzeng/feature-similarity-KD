import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append('../')

import torch.utils.data
from torch.nn import DataParallel
import torchvision.transforms as transforms
import argparse
import subprocess
import torch
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import tinyface_helper
from backbone.iresnet import iresnet50
from scipy.spatial import distance

import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))

# DataLoader
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ListDataset(Dataset):
    def __init__(self, img_list):
        super(ListDataset, self).__init__()
        self.img_list = img_list
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # Load Image
        image_path = self.img_list[idx]
        img = cv2.imread(image_path)
        img = img[:, :, :3]

        # To Tensor
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, idx



def prepare_dataloader(img_list, batch_size, num_workers=0):
    image_dataset = ListDataset(img_list)
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    return dataloader


# Evaluation
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def infer(model, dataloader, use_flip_test):
    features = []
    with torch.no_grad():
        for images, idx in tqdm(dataloader):
            images = images.to('cuda')
            feature = model(images)
            
            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                flipped_feature = model(fliped_images.to("cuda"))

                fused_feature = (feature + flipped_feature) / 2
                features.append(fused_feature.cpu().numpy())
            else:
                features.append(feature.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features


def load_model(args):
    # gpu init
    multi_gpus = False

    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda')

    # define backbone and margin layer
    if args.backbone == 'iresnet50':
        net = iresnet50(attention_type=args.mode)
    else:
        raise('Select Proper Backbone Network')
    
    net.load_state_dict(torch.load(args.net_path, map_location='cpu')['net_state_dict'], strict=False)
    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    net.eval()
    return net


def calc_accuracy(tinyface_test, probe, gallery, do_norm=True):
    if do_norm: 
        probe = probe / np.linalg.norm(probe, ord=2, axis=1).reshape(-1,1)
        gallery = gallery / np.linalg.norm(gallery, ord=2, axis=1).reshape(-1,1)
        
    # Similarity
    result = (probe @ gallery.T)
    
    index = np.argsort(-result, axis=1)
    
    p_l = np.array(tinyface_test.probe_labels)
    g_l = np.array(tinyface_test.gallery_labels)
    
    acc_list = []
    for rank in [1, 5, 10, 20]:
        correct = 0
        for ix, probe_label in enumerate(p_l):
            pred_label = g_l[index[ix][:rank]]
            
            if probe_label in pred_label:
                correct += 1
                
        acc = correct / len(p_l)
        acc_list += [acc * 100]
    
    print(acc_list)
    pd.DataFrame({'rank':[1, 5, 10, 20], 'values':acc_list}).to_csv(os.path.join(save_path, 'result_norm_%s.csv' %str(do_norm)), index=False)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tinyface')
    parser.add_argument('--data_root', default='Face/tinyface/new')
    parser.add_argument('--net_path', type=str, default='checkpoint/lr_face_recognition/fskd_cbam_cosface/equal_true/F_SKD_BLOCK_false_20/all/last_net.ckpt', help='scale size')
    
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--batch_size', default=1024, type=int, help='')

    parser.add_argument('--backbone', type=str, default='iresnet50', help='attention type')
    parser.add_argument('--mode', type=str, default='cbam', help='attention type')

    parser.add_argument('--save_name', type=str, default='name')
    parser.add_argument('--result_dir', type=str, default='./result_flip_true')
    parser.add_argument('--use_flip_test', type=str2bool, default='True')
    
    args = parser.parse_args()
    
    # load model
    model = load_model(args)
    tinyface_test = tinyface_helper.TinyFaceTest(tinyface_root=args.data_root)

    # set save root
    save_path = os.path.join(args.result_dir, args.save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('save_path: {}'.format(save_path))

    probe_loader = prepare_dataloader(tinyface_test.probe_paths, args.batch_size, num_workers=8)
    gallery_loader = prepare_dataloader(tinyface_test.gallery_paths, args.batch_size, num_workers=8)
    
    print('probe images : {}'.format(len(tinyface_test.probe_paths)))
    print('gallery images : {}'.format(len(tinyface_test.gallery_paths)))
    
    probe_features = infer(model, probe_loader, use_flip_test=args.use_flip_test)
    gallery_features = infer(model, gallery_loader, use_flip_test=args.use_flip_test)
    
    print('------------------ Start -------------------')
    print('Strategy : %s' %args.save_name)
    calc_accuracy(tinyface_test, probe_features, gallery_features, do_norm=False)
    calc_accuracy(tinyface_test, probe_features, gallery_features, do_norm=True)
    print('------------------- End ---------------------')