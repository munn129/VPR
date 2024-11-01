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
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader

from custom_dataset import CustomDataset
from extractor import Extractor

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--method', type=str, default='netvlad',
                      help='VPR method name, e.g., netvlad, cosplace, mixpvr, gem, convap, transvpr')
    args.add_argument('--dataset_dir', type=str, default='/media/moon/moon_ssd/moon_ubuntu/icrca/0519')
    args.add_argument('--dataset_name', type=str, default='oxford')
    args.add_argument('--save_dir', type=str, default='/media/moon/T7 Shield/master_research')
    args.add_argument('--batch_size', type=int, default=16)
    
    options = args.parse_args()

    method = options.method
    save_dir = Path(options.save_dir)
    dataset_name = options.dataset_name
    dataset_name_tail = '0519'
    batch_size = options.batch_size
    
    loader = DataLoader(CustomDataset(Path(options.dataset_dir)),
                        batch_size = batch_size,
                        num_workers = 0)
    
    extractor = Extractor(method, loader)
    extractor.feature_extract()

    save_file_dir = save_dir / (method + '_' + dataset_name + '_' + dataset_name_tail + '.npy')
    
    np.save(save_file_dir, extractor.get_matrix())

if __name__ == '__main__':
    main()