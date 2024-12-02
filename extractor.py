import sys
import configparser
import torchvision.transforms.functional as TF
import torchvision.transforms as tvf
import torch
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA

weight_prefix = '/media/moon/moon_ssd/pretrained_models'

WEIGHTS = {
    'netvlad' : './pretrained_models/mapillary_WPCA512.pth.tar',
    # 'netvlad' : f'{weight_prefix}/netvlad/mapillary_WPCA128.pth.tar',
    # 'cosplace' : './pretrained_models/cosplace_resnet152_512.pth',
    'cosplace' : f'{weight_prefix}/cosplace/',
    'mixvpr' : './pretrained_models/resnet50_MixVPR_512_channels(256)_rows(2).ckpt',
    'transvpr' : './pretrained_models/TransVPR_MSLS.pth'
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

def pca_decomposition(tensor):
    pass

class Extractor:
    def __init__(self, method: str, loader: DataLoader, dim=512):
        self.dim = dim
        DIM = dim

        if not torch.cuda.is_available():
            raise Exception("CUDA must need")

        self.device = torch.device('cuda')

        self.method = method
        
        if method == 'netvlad':
            from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding

            encoder_dim, encoder = get_backend() # 512, nn.Sequential(*layers)
            checkpoint = torch.load(WEIGHTS[method])
            config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])
            model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=True)
            model.load_state_dict(checkpoint['state_dict'])

            self.get_pca_encoding = get_pca_encoding

        elif method == 'cosplace':
            sys.path.append('./cosplace')
            from cosplace.cosplace_model import cosplace_network, layers

            model = cosplace_network.GeoLocalizationNet('ResNet152', self.dim)

            model_state_dict = torch.load(f'{WEIGHTS[method]}/resnet152_{self.dim}.pth')
            model.load_state_dict(model_state_dict)

        elif method == 'mixvpr' or method == 'gem' or method == 'convap':
            sys.path.append('./mixvpr')
            from mixvpr.main import VPRModel

            if method == 'mixvpr':
                model = VPRModel(backbone_arch='resnet50',
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
                
                state_dict = torch.load(WEIGHTS[method])
                model.load_state_dict(state_dict)

            elif method == 'gem':
                model = VPRModel(
                    agg_arch='GeM',
                    agg_config={'p': 3})
                
                # TODO
                DIM = 2048
                self.method = 'gem'
                
            elif method == 'convap':
                DIM = self.dim
                model = VPRModel(
                    agg_arch='ConvAP',
                    agg_config={'in_channels': 2048,
                                'out_channels': int(DIM/4)})
                
                # TODO
                self.method = 'convap'

        elif method == 'transvpr':
            sys.path.append('./transvpr')
            from transvpr.feature_extractor import Extractor_base
            from transvpr.blocks import POOL

            # TODO
            DIM = 256

            model = Extractor_base()
            pool = POOL(model.embedding_dim)
            model.add_module('pool', pool)

            checkpoint = torch.load(WEIGHTS[method])
            model.load_state_dict(checkpoint)

        else:
            raise Exception('Input method is not supported.')
        
        model = model.to(self.device)
        model.eval()
        self.model = model

        self.loader = loader
        self.matrix = np.empty((loader.__len__() * loader.batch_size, DIM))

    def feature_extract(self):

        if self.method == 'netvlad':
            with torch.no_grad():

                for image_tensor, indices in tqdm(self.loader):
                    
                    indices_np = indices.detach().numpy()

                    vlad = self.model.pool(self.model.encoder(image_tensor.to(self.device)))
                    vlad_pca = self.get_pca_encoding(self.model, vlad)

                    self.matrix[indices_np, :] = vlad_pca.detach().cpu().numpy()

            torch.cuda.empty_cache()

        elif self.method == 'transvpr':
            with torch.no_grad():

                for image_tensor, indices in tqdm(self.loader):

                    indices_np = indices.detach().numpy()

                    patch_feat = self.model(image_tensor.to(self.device))
                    global_feat, _ = self.model.pool(patch_feat)

                    self.matrix[indices_np, :] = global_feat.detach().cpu().numpy()

            torch.cuda.empty_cache()

        else: # cosplace, mixvpr, gem, convap
            with torch.no_grad():

                for image_tensor, indices in tqdm(self.loader):

                    indices_np = indices.detach().numpy()

                    descriptor = self.model(image_tensor.to(self.device))

                    self.matrix[indices_np, :] = descriptor.detach().cpu().numpy()

            torch.cuda.empty_cache()

    def get_matrix(self):
        return self.matrix
    
def main():
    pass

if __name__ == '__main__':
    main()