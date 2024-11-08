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


import os
import sys
import configparser

import torch
import torch.nn as nn
import numpy as np

from math import sqrt
from time import time

from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as tvf
import torch.nn.functional as F

from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
sys.path.append('./cosplace')
from cosplace.cosplace_model import cosplace_network, layers
sys.path.append('./mixvpr')
from mixvpr.main import VPRModel
sys.path.append('./transvpr')
from transvpr.feature_extractor import Extractor_base
from transvpr.blocks import POOL

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

netvlad_pretrained_dir = './pretrained_models/mapillary_WPCA512.pth.tar'
cosplace_pretrained_dir = './pretrained_models/cosplace_resnet152_512.pth'
mixvpr_pretrained_dir = './pretrained_models/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'
transvpr_pretrained_dir = './pretrained_models/TransVPR_MSLS.pth'
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


def transVPR_essential(image_tensor, device):
    
    model = Extractor_base()
    pool = POOL(model.embedding_dim)
    model.add_module('pool', pool)

    checkpoint = torch.load(transvpr_pretrained_dir)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        patch_feat = model(image_tensor.to(device))
        global_feat, attention_mask = model.pool(patch_feat)

        global_feat.detach().cpu().numpy()
        attention_mask.detach().cpu().numpy()
        # patch size : 16 x 16

    torch.cuda.empty_cache()


def netvlad_essential(image_tensor, device):
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

    torch.cuda.empty_cache()

    return vlad_global_pca


# TRANS-VLAD
def z_normal_map(image_tensor, device):

    trans_model = Extractor_base()
    pool = POOL(trans_model.embedding_dim)
    trans_model.add_module('pool', pool)

    trans_checkpoint = torch.load(transvpr_pretrained_dir)
    trans_model.load_state_dict(trans_checkpoint)
    trans_model.to(device)
    trans_model.eval()

    attention_mask = []

    with torch.no_grad():
        patch_feat = trans_model(image_tensor.to(device))
        global_feat, attention_mask = trans_model.pool(patch_feat)

        global_feat.detach().cpu().numpy()
        attention_mask = attention_mask.detach().cpu().numpy()

    # sum mask for z-score normalization 
    z_normalized_mask = []
    
    for i in range(attention_mask.shape[2]):
        z_normalized_mask.append(sum(attention_mask[0].T[i]))

    z_normalized_mask = np.array(z_normalized_mask)
    # z-score normalization
    # z_normalized_mask = (z_normalized_mask - np.mean(z_normalized_mask)) / np.std(z_normalized_mask)

    # 0 to 1 normalization
    z_normalized_mask = (z_normalized_mask - np.min(z_normalized_mask)) / (np.max(z_normalized_mask) - np.min(z_normalized_mask))
    z_normalized_mask = z_normalized_mask / sum(z_normalized_mask)

    return z_normalized_mask


def transvlad(image_tensor, device, mask_len):

    s1 = time()
    
    if mask_len is not int:
        mask_len = len(mask_len)

    # PIL Image.size -> W, H
    # W = image.size[0]
    # H = image.size[1]

    # torch.size([1,3,H,W])    
    H = image_tensor.size(dim=2)
    W = image_tensor.size(dim=3)
    patch_size = int(sqrt(H * W / mask_len))

    transforms = tvf.Compose(
        # 80 * 80: minmum size of netvlad
        [tvf.Resize((80, 80), interpolation=tvf.InterpolationMode.BICUBIC),
         tvf.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])]
    )

    s2 = time()
    print(f'1 : {s2 - s1}')

    encoder_dim, encoder = get_backend() # 512, nn.Sequential(*layers)

    checkpoint = torch.load(netvlad_pretrained_dir)

    config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

    model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=True)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    model.eval()
    
    vlad_matrix = np.empty((mask_len, encoder_dim))
    B = 0
    idx = 0

    s3 = time()
    print(f'2 : {s3 - s2}')

    with torch.no_grad():

        patch_batch = np.empty((mask_len, 3, 16, 16), dtype=np.float32)

        for w in range(0, W, patch_size):
            for h in range(0, H, patch_size):
                patch = image_tensor[B, :, w : w + patch_size, h : h + patch_size]
                patch_batch[idx, :] = patch.unsqueeze(0)
                idx += 1

                # patch = transforms(patch).unsqueeze(0)
                # patch_tensor = patch.to(device)
                # patch_encoding = model.encoder(patch_tensor)
                # local_vlad = model.pool(patch_encoding)
                # local_vlad_pca = get_pca_encoding(model, local_vlad)

                # vlad_matrix[idx, :] = local_vlad_pca.detach().cpu().numpy()
                # idx += 1

        s4 = time()
        print(f'3 : {s4 - s3}')

        patch_batch = transforms(torch.from_numpy(patch_batch))
        patch_tensor = patch_batch.to(device)
        patch_encoding = model.encoder(patch_tensor)
        local_vlad = model.pool(patch_encoding)
        local_vlad_pca = get_pca_encoding(model, local_vlad)

        vlad_matrix = local_vlad_pca.detach().cpu().numpy()
                
    torch.cuda.empty_cache()

    s5 = time()
    print(f'4 : {s5 - s4}')

    return vlad_matrix


def attention_map(image_tensor, device):

    trans_model = Extractor_base()
    pool = POOL(trans_model.embedding_dim)
    trans_model.add_module('pool', pool)

    trans_checkpoint = torch.load(transvpr_pretrained_dir)
    trans_model.load_state_dict(trans_checkpoint)
    trans_model.to(device)
    trans_model.eval()

    with torch.no_grad():
        patch_feat = trans_model(image_tensor.to(device))
        global_feat, attention_mask = trans_model.pool(patch_feat)

    attention_map = attention_mask.view(1,3,20,20)
    attention_map = attention_map.repeat_interleave(16, dim=2).repeat_interleave(16, dim=3)


    softmax_attention_map = F.softmax(attention_map, dim = 1)
            
    softmax_attention_map = (softmax_attention_map * attention_map).sum(dim = 1, keepdim = True)
    
    softmax_attention_map_extended = softmax_attention_map.expand_as(image_tensor)

    return (image_tensor * softmax_attention_map_extended).detach().cpu().numpy()


def main():

    # test_image = Image.open(test_image_dir)
    test_image = Image.open(test_image_dir).convert('RGB')
    # test_image = test_image.resize((100,100))

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
    # covAP_essential(image_tensor, device)
    transVPR_essential(image_tensor, device)
    # transvlad(image_tensor, device)

    torch.cuda.empty_cache()


def netvlad_minimum_test():
    test_image = Image.open(test_image_dir).convert('RGB')
    patch_size = 32
    
    while True:
        try:
            test_image = test_image.resize((patch_size,patch_size))
            image_tensor = TF.to_tensor(test_image)
            image_tensor.unsqueeze_(0)

            if not torch.cuda.is_available():
                raise Exception('No GPU found')
    
            device = torch.device('cuda')

            patch_netvlad_essential(image_tensor, device)

            break

        except:
            print(f'patch size at {patch_size} is failed')
            patch_size += 1

    print(patch_size)


def transvlad_main():

    test_image = Image.open(test_image_dir).convert('RGB')

    transforms = tvf.Compose(
        [tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC),
         tvf.ToTensor(),
         tvf.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])]
    )

    image_tensor = transforms(test_image).unsqueeze(0)

    if not torch.cuda.is_available():
        raise Exception('No GPU found')
    
    device = torch.device('cuda')

    z_normal_mask = z_normal_map(image_tensor, device)

    vlad_matrix = transvlad(image_tensor, device, z_normal_mask)
    # vlad_matrix = transmixvpr(image_tensor, device, z_normal_mask)

    trans_vlad_vector = z_normal_mask @ vlad_matrix

def trans_mix_poc():
    test_image = Image.open(test_image_dir).convert('RGB')

    transforms = tvf.Compose(
        [tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC),
         tvf.ToTensor(),
         tvf.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])]
    )

    image_tensor = transforms(test_image).unsqueeze(0)

    if not torch.cuda.is_available():
        raise Exception('No GPU found')
    
    device = torch.device('cuda')

    a = attention_map(image_tensor, device)
    mixvpr_essential(a, device)


if __name__ == '__main__':
    main()
    # netvlad_minimum_test()
    # transvlad_main()
    # trans_mix_poc()