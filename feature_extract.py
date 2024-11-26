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
from transvlad import TransVLAD


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--method', type=str, default='transvlad',
                      help='VPR method name, e.g., netvlad, cosplace, mixpvr, gem, convap, transvpr')
    args.add_argument('--dataset_dir', type=str, default='/media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/front')
    args.add_argument('--dataset_name', type=str, default='oxford')
    args.add_argument('--save_dir', type=str, default='/media/moon/T7 Shield/multiview_results')
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--version', type=str, default='')
    
    options = args.parse_args()

    method = options.method
    save_dir = Path(options.save_dir)
    dataset_name = options.dataset_name
    batch_size = options.batch_size
    version = options.version

    if options.method == 'transvlad':
        # TODO
        # batch_size > 1
        batch_size = 1
    
    loader = DataLoader(CustomDataset(Path(options.dataset_dir)),
                        batch_size = batch_size,
                        num_workers = 0)
    
    extractor = TransVLAD(loader) if options.method == 'transvlad' else Extractor(method, loader)

    extractor.feature_extract()

    #####
    dataset_name_tail = options.dataset_dir.split('/')[-2]

    #####

    save_file_dir = save_dir / (method + '_' + dataset_name + '_' + dataset_name_tail + version + '.npy')
    
    np.save(save_file_dir, extractor.get_matrix())

def main2():

    # for test just one method

    args = argparse.ArgumentParser()
    args.add_argument('--method', type=str, default='transvlad',
                      help='VPR method name, e.g., netvlad, cosplace, mixpvr, gem, convap, transvpr')
    args.add_argument('--dataset_dir', type=str, default='/media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/concat')
    args.add_argument('--dataset_name', type=str, default='oxford')
    args.add_argument('--save_dir', type=str, default='/media/moon/T7 Shield/multiview_results')
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--version', type=str, default='')
    args.add_argument('--dim', type=int, default=512)
    args.add_argument('--image_size', type=int, default=1280)
    
    options = args.parse_args()

    method = options.method
    save_dir = Path(options.save_dir)
    dataset_name = options.dataset_name
    batch_size = options.batch_size
    VERSION = options.version
    dim = options.dim

    if options.method == 'transvlad':
        # TODO
        # batch_size > 1
        batch_size = 1
    
    loader = DataLoader(CustomDataset(Path(options.dataset_dir), 0, image_size=options.image_size),
                        batch_size = batch_size,
                        num_workers = 0)
    
    extractor = TransVLAD(loader, dim=dim) if options.method == 'transvlad' else Extractor(method, loader, dim=dim)

    extractor.feature_extract()

    #####
    dataset_name_tail = options.dataset_dir.split('/')[-2]

    #####

    save_file_dir = save_dir / (method + VERSION + '_' + dataset_name + '_' + dataset_name_tail + '.npy')
    
    np.save(save_file_dir, extractor.get_matrix())

if __name__ == '__main__':
    # main()
    main2()