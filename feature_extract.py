'''
NetVLAD: Relja Arandjelović, et al, NetVLAD: CNN architecture for weakly supervised place recognition, CVPR, 2016.
PatchNetVLAD:Stephen Hausler, et al, Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition, CVPR, 2021.
-> two stage, e.g. TransVPR, SuperGlue
cosplace: Gabriele Berton et al. Rethinking Visual Geo-localization for Large-Scale Applications, CVPR, 2022.
mixvpr: Amar Ali-bey, et al, MixVPR: Feature Mixing for Visual Place Recognition, WACV, 2023.
GeM: Filip Radenovic, et al, Finetuning CNN image retrieval with no human annotation, TPAMI, 2018,
convAP: Amar Ali-bey, et al, GSV-Cities: Toward Appropriate Supervised Visual Place Recognition, Neurocomputing, 2022.
transVPR: Ruotong Wang, et al, TransVPR: Transformer-based place recognition with multi-level attention aggregation, CVPR, 2022.
'''

import argparse
import configparser
import torch

from pathlib import Path
from torch.utils.data import DataLoader

WEIGHTS = {
    'netvlad' : './pretrained_models/mapillary_WPCA512.pth.tar',
    'cosplace' : './pretrained_models/cosplace_resnet152_512.pth',
    'mixpvr' : './pretrained_models/resnet50_MixVPR_512_channels(256)_rows(2).ckpt',
    'tranvpr' : './pretrained_models/TransVPR_MSLS.pth'
}

config = configparser.ConfigParser()
config['global_params'] = {
    'pooling' : 'netvlad',
    'resumepath' : './pretrained_models/mapillary_WPCA',
    'threads' : 0,
    'num_pcs' : 512,
    'ngpu' : 1,
    'patch_sizes' : 5,
    'strides' : 1
}

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--method', type=str, default='netvlad',
                      help='VPR method name, e.g., netvlad.')
    args.add_argument('--dataset', type=str, default='/media/moon/moon_ssd/moon_ubuntu/icrca/0519',
                      help='dataset directory')
    args.add_argument('--save_dir', type=str, default='/media/moon/T7 Shield/master_research',
                      help='save directory')
    
    if not torch.cuda.is_available():
        raise Exception("CUDA must need")
    
    device = torch.device('cuda')
    
    options = args.parse_args()

    method = options.method
    dataset_dir = options.dataset
    save_dir = options.save_dir

if __name__ == '__main__':
    main()