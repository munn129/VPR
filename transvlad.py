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

weight_prefix = '/media/moon/moon_ssd/pretrained_models/'

WEIGHTS = {
    'netvlad' : './pretrained_models/mapillary_WPCA512.pth.tar',
    # 'cosplace' : './pretrained_models/cosplace_resnet152_512.pth',
    'cosplace' : f'{weight_prefix}cosplace/resnet152_256.pth',
    'mixvpr' : './pretrained_models/resnet50_MixVPR_512_channels(256)_rows(2).ckpt',
    # 'mixvpr' : f'{weight_prefix}mixvpr/resnet50_MixVPR_128_channels(64)_rows(2).ckpt',
    'tranvpr' : './pretrained_models/TransVPR_MSLS.pth'
}

# DIM = 512

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
    def __init__(self, loader: DataLoader, dim=512):

        DIM = dim

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
                'out_rows' : 2, # the output dim will be (out_rows * out_channels)
                'is_mix' : True
            })
        
        # resnet50_MixVPR_128_channels(64)_rows(2).ckpt
        # mixvpr_model = VPRModel(backbone_arch='resnet50',
        #     layers_to_crop=[4],
        #     agg_arch='MixVPR',
        #     agg_config={
        #         'in_channels' : 1024,
        #         'in_h' : 20,
        #         'in_w' : 20,
        #         'out_channels' : 64,
        #         'mix_depth' : 4,
        #         'mlp_ratio': 1,
        #         'out_rows' : 2, # the output dim will be (out_rows * out_channels)
        #         'is_mix' : True
        #     })
        
        mixvpr_state_dict = torch.load(WEIGHTS['mixvpr'])
        mixvpr_model.load_state_dict(mixvpr_state_dict)

        mixvpr_model = mixvpr_model.to(self.device)
        mixvpr_model.eval()
        self.mixvpr_model = mixvpr_model

        # cosplace
        # cos_model = cosplace_network.GeoLocalizationNet('ResNet152', 512)
        # cos_model = cosplace_network.GeoLocalizationNet('ResNet152', 2048)
        cos_model = cosplace_network.GeoLocalizationNet('ResNet152', dim)
        # cos_model_state_dict = torch.load(WEIGHTS['cosplace'])
        cos_model_state_dict = torch.load(f'{weight_prefix}cosplace/resnet152_{dim}.pth')
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

        self.normalize = tvf.Compose(
            [tvf.Normalize([0.485, 0.456, 0.406],
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
    
    def something3(self, image_tensor):
        with torch.no_grad():
            
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

            # 4 * 512
            concatenated_descriptor = torch.cat((top_left_des, top_right_des, bottom_left_des, bottom_right_des), dim = 0)

        torch.cuda.empty_cache()

        return concatenated_descriptor.sum(dim = 0, keepdim = True).detach().cpu().numpy()

    def something4(self, image_tensor):

        with torch.no_grad():

            mixvpr_des, attention_mask = self.mixvpr_model(image_tensor.to(self.device))

            # attention_mask.shape: (1, 400)
            attention_mask = attention_mask.view(1,20,20)

            attention_mask = attention_mask.sum(dim = 0, keepdim = True)

            attention_mask = attention_mask.view(1,1,2,10,2,10).sum(dim=(3,5))

            attention_mask = attention_mask.view(1,4)
            
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

            # 4 * 512
            concatenated_descriptor = torch.cat((top_left_des, top_right_des, bottom_left_des, bottom_right_des), dim = 0)

        torch.cuda.empty_cache()

        return (attention_mask @ concatenated_descriptor).detach().cpu().numpy()


    def something5(self, image_tensor):

        with torch.no_grad():

            mix = self.mixvpr_model(image_tensor.to(self.device))
            
            # print(mix.shape)
            # x = torch.randn(1,2048,10,10)

            # mix: (1, 1024, 400)
            # (1, 1024, 400) -> (1, 2048, 200)
            mix = F.interpolate(mix, size=200, mode='linear', align_corners=False)
            mix = mix.repeat(1,2,1)

            # (1, 2048, 200) -> (1, 2048, 100)
            mix = F.interpolate(mix, size=100, mode='linear', align_corners=False)

            # (1, 2048, 100) -> (1, 2048, 10, 10)
            mix = mix.view(1, 2048, 10, 10)

            tmp = self.cos_model(mix.to(self.device))

            return tmp.detach().cpu().numpy()

        torch.cuda.empty_cache()

    def something6(self, image_tensor):

        with torch.no_grad():
            
            mix_vpr = self.mixvpr_model(image_tensor.to(self.device)) # 1 * 1024 * 400
            
            patch_feat = self.trans_model(image_tensor.to(self.device))
            _, mask = self.trans_model.pool(patch_feat) # 1 * 3 * 400

            mask = mask.sum(dim = 1, keepdim = True)
            mask /= 3 # 1 * 1 * 400

            extened_mask = mask.repeat(1, 1024, 1) # 1 * 1024 * 400

            # mixmix = torch.cat((mix_vpr, extened_mask), dim = 1) # 1* 2048 * 400

            mixmix = []
            id_1 = 0
            id_2 = 0

            for i in range(mix_vpr.shape[1] * 2):
                if i%2 :
                    mixmix.append(mix_vpr[:,id_1,:])
                    id_1 += 1
                else:
                    mixmix.append(extened_mask[:,id_2,:])
                    id_2 += 1

            mixmix = torch.cat(mixmix).unsqueeze(0)

            mixmix = F.interpolate(mixmix, size=100, mode='linear', align_corners=False) # 1 * 2048 * 100

            mixmix = mixmix.view(1, 2048, 10, 10)

            tmp = self.cos_model(mixmix.to(self.device))

        torch.cuda.empty_cache()

        return tmp.detach().cpu().numpy()
    
    def something7(self, image_tensor):

        with torch.no_grad():
            _, mask = self.trans_model.pool(self.trans_model(image_tensor.to(self.device))) # 1 * 3 * 400
            mask = mask.view(1,3, 20, 20)
            mask = mask.repeat_interleave(16, dim = 2).repeat_interleave(16, dim = 3) # 1,3,320,320
            # mask = self.normalize(mask)
            
            trans_mixvpr_des = self.mixvpr_model(mask.to(self.device)) # 1 * 1024 * 400
            mixvpr_des = self.mixvpr_model(image_tensor.to(self.device)) # 1 * 1024 * 400

            concat_des = torch.cat((trans_mixvpr_des, mixvpr_des), dim = 1) # 1*2048*400
            concat_des = F.interpolate(concat_des, size=100, mode='linear', align_corners=False) # 1 * 2048 * 100

            concat_des = concat_des.view(1, 2048, 10, 10)

            descriptor = self.cos_model(concat_des.to(self.device))

        torch.cuda.empty_cache()

        return descriptor.detach().cpu().numpy()

    
    def something8(self, image_tensor):
        
        # image_tensor: 1,3,320,320

        with torch.no_grad():
            W = int(image_tensor.shape[2]/2)
            H = int(image_tensor.shape[3]/2)

            top_left = F.interpolate(image_tensor[:, :, :W, :H], size=(320,320), mode='bilinear', align_corners=False)
            top_right = F.interpolate(image_tensor[:, :, :W, H:], size=(320,320), mode='bilinear', align_corners=False)
            bottom_left = F.interpolate(image_tensor[:, :, W:, :H], size=(320,320), mode='bilinear', align_corners=False)
            bottom_right = F.interpolate(image_tensor[:, :, W:, H:], size=(320,320), mode='bilinear', align_corners=False)
            # size must 1,3,320,320

            top_left_mix = self.mixvpr_model(top_left.to(self.device))
            top_right_mix = self.mixvpr_model(top_right.to(self.device))
            bottom_left_mix = self.mixvpr_model(bottom_left.to(self.device))
            bottom_right_mix = self.mixvpr_model(bottom_right.to(self.device))
            # 1, 1024, 400

            combined_mix = torch.cat([top_left_mix, top_right_mix, bottom_left_mix, bottom_right_mix], dim=1)
            # 1, 4096, 400

            # conv_layer = torch.nn.Conv1d(in_channels=4096, out_channels=2048, kernel_size=1)
            # reduced_tensor = conv_layer(combined_mix)
            reduced_tensor = combined_mix.view(1, 2048, 2, 400).mean(dim=2)
            # 1, 2048, 400

            interpolated = F.interpolate(reduced_tensor, size=100, mode='linear', align_corners=False)
            # 1, 2048, 100

            interpolated = interpolated.view(1, 2048, 10, 10)

            des = self.cos_model(interpolated.to(self.device))

        torch.cuda.empty_cache()

        return des.detach().cpu().numpy()
    
    def something9(self, image_tensor):
        # cross concatenation
        
        with torch.no_grad():
            mixed_tensor = self.mixvpr_model(image_tensor.to(self.device))
            # 1, 1024, 400

            mixed_tensor = mixed_tensor.view(1,1024,20,20)

            TH = int(mixed_tensor.shape[2]/2)

            top_left = mixed_tensor[:,:,:TH,:TH].view(1, 512, 2, TH, TH).sum(dim=2)
            top_right = mixed_tensor[:,:,:TH, TH:].view(1, 512, 2, TH, TH).sum(dim=2)
            bottom_left = mixed_tensor[:,:,TH:, :TH].view(1, 512, 2, TH, TH).sum(dim=2)
            bottom_right = mixed_tensor[:,:,TH:,TH:].view(1, 512, 2, TH, TH).sum(dim=2)
            # 1, 1024, 10, 10 -> 1, 512, 10, 10

            combined_mix = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=1)
            # 1, 2048, 10, 10

            des = self.cos_model(combined_mix.to(self.device))

        torch.cuda.empty_cache()

        return des.detach().cpu().numpy()
    
    def something10(self, image_tensor):
        # horizontal concatenation
        # 66

        with torch.no_grad():
            mixed_tensor = self.mixvpr_model(image_tensor.to(self.device))
            # 1, 1024, 400

            mixed_tensor = mixed_tensor.view(1,1024,20,20)

            TH = int(mixed_tensor.shape[2]/4)

            far_left = mixed_tensor[:,:,:TH,:].view(1, 512, 2, TH, TH * 4).sum(dim=2)
            middle_left = mixed_tensor[:,:,TH:TH*2,:].view(1, 512, 2, TH, TH * 4).sum(dim=2)
            middle_right = mixed_tensor[:,:,TH*2:TH*3,:].view(1, 512, 2, TH, TH * 4).sum(dim=2)
            far_right = mixed_tensor[:,:,TH*3:TH*4,:].view(1, 512, 2, TH, TH * 4).sum(dim=2)
            # 1, 1024, 10, 10 -> 1, 512, 10, 10

            combined_mix = torch.cat([far_left, middle_left, middle_right, far_right], dim=1)
            # 1, 2048, 10, 10

            des = self.cos_model(combined_mix.to(self.device))

        torch.cuda.empty_cache()

        return des.detach().cpu().numpy()
    
    def something11(self, image_tensor):
        # vertical concatenation
        # 

        with torch.no_grad():
            mixed_tensor = self.mixvpr_model(image_tensor.to(self.device))
            # 1, 1024, 400

            mixed_tensor = mixed_tensor.view(1,1024,20,20)

            TH = int(mixed_tensor.shape[2]/4)

            far_left = mixed_tensor[:,:,:,:TH].view(1, 512, 2, TH * 4, TH).sum(dim=2)
            middle_left = mixed_tensor[:,:,:,TH:TH*2].view(1, 512, 2, TH * 4, TH).sum(dim=2)
            middle_right = mixed_tensor[:,:,:,TH*2:TH*3].view(1, 512, 2, TH * 4, TH).sum(dim=2)
            far_right = mixed_tensor[:,:,:,TH*3:TH*4].view(1, 512, 2, TH * 4, TH).sum(dim=2)
            # 1, 1024, 10, 10 -> 1, 512, 10, 10

            combined_mix = torch.cat([far_left, middle_left, middle_right, far_right], dim=1)
            # 1, 2048, 10, 10

            des = self.cos_model(combined_mix.to(self.device))

        torch.cuda.empty_cache()

        return des.detach().cpu().numpy()


    def feature_extract(self):
        
        for image_tensor, indices in tqdm(self.loader):

            indices_np = indices.detach().numpy()
            
            # self.z_normal(image_tensor)
            # self.z_normalized_mask = np.ones((400,1))
            # self.local_vlad(image_tensor)

            self.matrix[indices_np, :] = self.something11(image_tensor)


    def get_matrix(self):
        return self.matrix
    
def main():

    dir = '/media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/concat'

    loader = DataLoader(CustomDataset(Path(dir), 0),
                        batch_size = 1,
                        num_workers = 0)
    
    extractor = TransVLAD(loader)

    for image_tensor, id in tqdm(loader):

        extractor.something11(image_tensor)

if __name__ == '__main__':
    main()