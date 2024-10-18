'''
NetVLAD: Relja Arandjelović, et al, NetVLAD: CNN architecture for weakly supervised place recognition, CVPR, 2016.
PatchNetVLAD:Stephen Hausler, et al, Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition, CVPR, 2021.
-> two stage, e.g. TransVPR, SuperGlue
cosplace: Gabriele Berton et al. Rethinking Visual Geo-localization for Large-Scale Applications, CVPR, 2022.
mixvpr: Amar Ali-bey, et al, MixVPR: Feature Mixing for Visual Place Recognition, WACV, 2023.
GeM: Filip Radenovic, et al, Finetuning CNN image retrieval with no human annotation, TPAMI, 2018,
convAP: Amar Ali-bey, et al, GSV-Cities: Toward Appropriate Supervised Visual Place Recognition, Neurocomputing, 2022.
'''


import os
import sys
import configparser

import torch
import torch.nn as nn
import numpy as np

from PIL import Image
import torchvision.transforms.functional as TF

import torchvision.transforms as tvf

from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
sys.path.append('./cosplace')
from cosplace.cosplace_model import cosplace_network, layers
sys.path.append('./mixvpr')
from mixvpr.main import VPRModel

config = configparser.ConfigParser()
config['global_params'] = {
    'pooling' : 'patchnetvlad',
    'resumepath' : './pretrained_models/mapillary_WPCA',
    'threads' : 0,
    'num_pcs' : 512,
    'ngpu' : 1,
    'patch_sizes' : 5,
    'strides' : 1
}

netvlad_pretrained_dir = './pretrained_models/mapillary_WPCA512.pth.tar'
cosplace_pretrained_dir = './pretrained_models/cosplace_resnet152_512.pth'
mixvpr_pretrained_dir = './pretrained_models/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'
test_image_dir= 'test.png'

def patch_netvlad_essential(image_tensor, device):
        # patch netvlad start
    encoder_dim, encoder = get_backend() # 512, nn.Sequential(*layers)

    checkpoint = torch.load(netvlad_pretrained_dir)

    config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

    model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=True)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    # feature extract
    pool_size = int(config['global_params']['num_pcs'])

    model.eval()

    with torch.no_grad():
        features = np.empty((1, pool_size), dtype=np.float32)

        image_tensor = image_tensor.to(device)

        image_encoding = model.encoder(image_tensor)

        vlad_local, vlad_global = model.pool(image_encoding)

        vlad_global_pca = get_pca_encoding(model, vlad_global)
        vlad_global_pca = vlad_global_pca.cpu().numpy()

        #patch
        local_feats = []
        for this_iter, this_local in enumerate(vlad_local):
            this_local_feats = get_pca_encoding(model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))). \
                reshape(this_local.size(2), this_local.size(0), pool_size).permute(1, 2, 0)
            local_feats.append(torch.transpose(this_local_feats[0, :, :], 0, 1))

        local_feats = local_feats[0].cpu().numpy()
    # patch net vlad end
    torch.cuda.empty_cache()

def cosplace_essential(image_tensor, device):
     # cosplace start
    model = cosplace_network.GeoLocalizationNet('ResNet152', 512)
    model_state_dict = torch.load(cosplace_pretrained_dir)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    
    descriptors = []
    with torch.no_grad():
        descriptors = model(image_tensor.to(device))
        descriptors = descriptors.cpu().numpy()

    # PATCH COSPLACE
    # for this_iter, this_local in enumerate(vlad_local):
    #         this_local_feats = get_pca_encoding(model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))). \
    #             reshape(this_local.size(2), this_local.size(0), pool_size).permute(1, 2, 0)
    #         local_feats.append(torch.transpose(this_local_feats[0, :, :], 0, 1))

    # local_feats = local_feats.cpu().numpy()

    # cosplace end
    torch.cuda.empty_cache()

def mixvpr_essential(image_tensor, device):
    
    model = VPRModel(backbone_arch='resnet50',
                     layers_to_crop=[4],
                     agg_arch='MixVPR',
                     agg_config={
                         'in_channels' : 1024,
                         'in_h' : 20,
                         'in_w' : 20,
                         'out_channels' : 1024,
                         'mix_depth' : 4,
                         'mlp_ratio': 1,
                         'out_rows' : 4 # the output dim will be (out_rows * out_channels)
                     })
    
    #---- Aggregator
        # agg_arch='CosPlace',
        # agg_config={'in_dim': 2048,
        #             'out_dim': 2048},
        # agg_arch='GeM',
        # agg_config={'p': 3},
        
        # agg_arch='ConvAP',
        # agg_config={'in_channels': 2048,
        #             'out_channels': 2048},
    
    state_dict = torch.load(mixvpr_pretrained_dir)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    des = []
    with torch.no_grad():
        des = model(image_tensor.to(device))
        des = des.detach().cpu().numpy()

    torch.cuda.empty_cache()

def gem_essential(image_tensor, device):
    model = VPRModel(
        agg_arch='GeM',
        agg_config={'p': 3})
    
    model.to(device)
    model.eval()

    des = []
    with torch.no_grad():
        des = model(image_tensor.to(device))
        des = des.detach().cpu().numpy()

    torch.cuda.empty_cache()

def covAP_essential(image_tensor, device):
    model = VPRModel(
        agg_arch='ConvAP',
        agg_config={'in_channels': 2048,
                     'out_channels': 2048})
    
    model.to(device)
    model.eval()

    des = []
    with torch.no_grad():
        des = model(image_tensor.to(device))
        des = des.detach().cpu().numpy()

    torch.cuda.empty_cache()


def main():

    # test_image = Image.open(test_image_dir)
    test_image = Image.open(test_image_dir).convert('RGB')

    # image_tensor = TF.to_tensor(test_image)
    # image_tensor.unsqueeze_(0)

    transforms = tvf.Compose(
        [tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC),
         tvf.ToTensor(),
         tvf.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])]
    )

    # if don't .unsqueeze(0) -> ValueError: expected 4D input (got 3D input)
    image_tensor = transforms(test_image).unsqueeze(0)

    if not torch.cuda.is_available():
        raise Exception('No GPU found')
    
    device = torch.device('cuda')

    # patch_netvlad_essential(image_tensor, device)
    # cosplace_essential(image_tensor, device)
    # mixvpr_essential(image_tensor, device)
    # gem_essential(image_tensor, device)
    covAP_essential(image_tensor, device)

if __name__ == '__main__':
    main()