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
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from custom_dataset import CustomDataset
from extractor import Extractor

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--method', type=str, default='netvlad',
                      help='VPR method name, e.g., netvlad.')
    args.add_argument('--dataset', type=str, default='/media/moon/moon_ssd/moon_ubuntu/icrca/0519',
                      help='dataset directory')
    args.add_argument('--save_dir', type=str, default='/media/moon/T7 Shield/master_research',
                      help='save directory')
    
    options = args.parse_args()

    save_dir = options.save_dir
    
    loader = DataLoader(CustomDataset(Path(options.dataset)),
                        batch_size = 5,
                        num_workers = 0)

if __name__ == '__main__':
    main()