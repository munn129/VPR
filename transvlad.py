import sys
import configparser
import torchvision.transforms.functional as TF
import torchvision.transforms as tvf
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from math import sqrt
from pathlib import Path
from custom_dataset import CustomDataset

from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
sys.path.append('./cosplace')
from cosplace.cosplace_model import cosplace_network, layers
sys.path.append('./mixvpr')
from mixvpr.main import VPRModel
sys.path.append('./transvpr')
from transvpr.feature_extractor import Extractor_base
from transvpr.blocks import POOL

WEIGHTS = {
    'netvlad' : './pretrained_models/mapillary_WPCA512.pth.tar',
    'cosplace' : './pretrained_models/cosplace_resnet152_512.pth',
    'mixpvr' : './pretrained_models/resnet50_MixVPR_512_channels(256)_rows(2).ckpt',
    'tranvpr' : './pretrained_models/TransVPR_MSLS.pth'
}

DIM = 512

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

class TransVLAD:
    def __init__(self, loader: DataLoader):

        if not torch.cuda.is_available():
            raise Exception('CUDA must need')
        
        self.device = torch.device('cuda')
        
        if loader.batch_size > 1:
            # TODO
            # batch size > 1
            raise Exception('batch size must be 1')
        
        self.loader = loader
        
        # trans vpr
        trans_model = Extractor_base()
        pool = POOL(trans_model.embedding_dim)
        trans_model.add_module('pool', pool)
        trans_checkpoint = torch.load(WEIGHTS['tranvpr'])
        trans_model.load_state_dict(trans_checkpoint)
        trans_model = trans_model.to(self.device)
        trans_model.eval()

        self.trans_model = trans_model

        # netvlad
        encoder_dim, encoder = get_backend()
        vald_checkpoint = torch.load(WEIGHTS['netvlad'])
        config['global_params']['num_clusters'] = str(vald_checkpoint['state_dict']['pool.centroids'].shape[0])
        vlad_model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=True)
        vlad_model.load_state_dict(vald_checkpoint['state_dict'])
        vlad_model = vlad_model.to(self.device)
        vlad_model.eval()

        self.vlad_model = vlad_model

        # mixvpr
        mixvpr_model = VPRModel(backbone_arch='resnet50',
            layers_to_crop=[4],
            agg_arch='MixVPR',
            agg_config={
                'in_channels' : 1024,
                'in_h' : 20,
                'in_w' : 20,
                'out_channels' : 256,
                'mix_depth' : 4,
                'mlp_ratio': 1,
                'out_rows' : 2 # the output dim will be (out_rows * out_channels)
            })
        
        state_dict = torch.load(WEIGHTS['mixpvr'])
        mixvpr_model.load_state_dict(state_dict)

        mixvpr_model = mixvpr_model.to(self.device)
        mixvpr_model.eval()
        self.mixvpr_model = mixvpr_model

        # cosplace
        cos_model = cosplace_network.GeoLocalizationNet('ResNet152', 512)
        cos_model_state_dict = torch.load(WEIGHTS['cosplace'])
        cos_model.load_state_dict(cos_model_state_dict)
        cos_model = cos_model.to(self.device)
        cos_model.eval()
        self.cos_model = cos_model 

        # var
        self.z_normalized_mask = []
        self.vlad_matrix = []
        self.matrix = np.empty((loader.__len__(), DIM))

        self.transforms = tvf.Compose(
            # 80 * 80: minmum size of netvlad
            [tvf.Resize((80, 80), interpolation=tvf.InterpolationMode.BICUBIC),
            tvf.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])]
        )

    def z_normal(self, image_tensor):
        attention_mask = []
        self.z_normalized_mask = []

        with torch.no_grad():
            patch_feat = self.trans_model(image_tensor.to(self.device))
            attention_mask = self.trans_model.pool(patch_feat)[1].detach().cpu().numpy()
            # attention mask (1,3,400)

        torch.cuda.empty_cache()

        # attention_mask.shape: 1,3,400

        for i in range(attention_mask.shape[2]):
            self.z_normalized_mask.append(sum(attention_mask[0].T[i]))

        self.z_normalized_mask = np.array(self.z_normalized_mask)

        # # z-score normalization
        # self.z_normalized_mask = (self.z_normalized_mask - np.mean(self.z_normalized_mask)) / np.std(self.z_normalized_mask)

        # # 0 to 1 normalization
        self.z_normalized_mask = (self.z_normalized_mask - np.min(self.z_normalized_mask)) / (np.max(self.z_normalized_mask) - np.min(self.z_normalized_mask))
        self.z_normalized_mask = self.z_normalized_mask / sum(self.z_normalized_mask)

        # mask filtering
        threshold = np.percentile(self.z_normalized_mask, 80)
        self.z_normalized_mask = np.where(self.z_normalized_mask >= threshold, 1, 0)

        # mid attention mask test
        # self.z_normalized_mask = attention_mask[0, 1, :]


    def local_vlad(self, image_tensor):
        
        mask_len = len(self.z_normalized_mask)

        H = image_tensor.size(dim=2)
        W = image_tensor.size(dim=3)
        patch_size = int(sqrt(H * W / mask_len))

        self.vlad_matrix = np.empty((mask_len, DIM))
        B = 0
        idx = 0

        with torch.no_grad():

            patch_batch = np.empty((mask_len, 3, 16, 16), dtype=np.float32)

            for w in range(0, W, patch_size):
                for h in range(0, H, patch_size):
                    patch = image_tensor[B, :, w : w + patch_size, h : h + patch_size]
                    patch_batch[idx, :] = patch.unsqueeze(0)
                    idx += 1

            patch_batch = self.transforms(torch.from_numpy(patch_batch))
            patch_tensor = patch_batch.to(self.device)
            patch_encoding = self.vlad_model.encoder(patch_tensor)
            local_vlad = self.vlad_model.pool(patch_encoding)
            local_vlad_pca = get_pca_encoding(self.vlad_model, local_vlad)

            self.vlad_matrix = local_vlad_pca.detach().cpu().numpy()
                    
        torch.cuda.empty_cache()

    
    def something(self, image_tensor):

        with torch.no_grad():
            patch_feat = self.trans_model(image_tensor.to(self.device))
            global_feat, attention_mask = self.trans_model.pool(patch_feat)

            attention_map = attention_mask.view(1,3,20,20)
            attention_map = attention_map.repeat_interleave(16, dim=2).repeat_interleave(16, dim=3)

            softmax_attention_map = F.softmax(attention_map, dim = 1)
            
            softmax_attention_map = (softmax_attention_map * attention_map).sum(dim = 1, keepdim = True)

            softmax_attention_map_extended = softmax_attention_map.expand_as(image_tensor)
            
            prob_map =  (image_tensor.to(self.device) * softmax_attention_map_extended)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            normalized_prob_map = (prob_map - mean.to(self.device)) /std.to(self.device)

            normalized_prob_map = self.mixvpr_model(normalized_prob_map).detach().cpu().numpy()

        torch.cuda.empty_cache()

        return normalized_prob_map
            # return self.mixvpr_model(attention_map.to(self.device)).detach().cpu().numpy()


    def something2(self , image_tensor):
        with torch.no_grad():
            patch_feat = self.trans_model(image_tensor.to(self.device))
            global_feat, attention_mask = self.trans_model.pool(patch_feat)

            # attention_mask.shape: (1,3, 400)
            attention_mask = attention_mask.view(1,3,20,20)

            attention_mask = attention_mask.sum(dim = 1, keepdim = True)

            attention_mask = attention_mask.view(1,1,2,10,2,10).sum(dim=(3,5))

            attention_mask = attention_mask.view(1,1,4)

            attention_mask /= 3

            W = int(image_tensor.shape[2]/2)
            H = int(image_tensor.shape[3]/2)

            top_left = image_tensor[:, :, :W, :H]
            top_right = image_tensor[:, :, :W, H:]
            bottom_left = image_tensor[:, :, W:, :H]
            bottom_right = image_tensor[:, :, W:, H:]

            top_left_des = self.cos_model(top_left.to(self.device))
            top_right_des = self.cos_model(top_right.to(self.device))
            bottom_left_des = self.cos_model(bottom_left.to(self.device))
            bottom_right_des = self.cos_model(bottom_right.to(self.device))

            concatenated_descriptor = torch.cat((top_left_des, top_right_des, bottom_left_des, bottom_right_des), dim = 0)

        torch.cuda.empty_cache()

        return (attention_mask @ concatenated_descriptor).detach().cpu().numpy()


    def feature_extract(self):
        
        for image_tensor, indices in tqdm(self.loader):

            indices_np = indices.detach().numpy()
            
            # self.z_normal(image_tensor)
            # self.z_normalized_mask = np.ones((400,1))
            # self.local_vlad(image_tensor)

            self.matrix[indices_np, :] = self.something2(image_tensor)


    def get_matrix(self):
        return self.matrix
    
def main():

    dir = '/media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/front'

    loader = DataLoader(CustomDataset(Path(dir), 0),
                        batch_size = 1,
                        num_workers = 0)
    
    extractor = TransVLAD(loader)

    for image_tensor, id in tqdm(loader):

        extractor.something2(image_tensor)

if __name__ == '__main__':
    main()