import sys
import configparser
import torchvision.transforms.functional as TF
import torchvision.transforms as tvf
import torch

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

class Extractor:
    def __init__(self, method: str, save_dir: str, loader: DataLoader):

        if not torch.cuda.is_available():
            raise Exception("CUDA must need")

        self.device = torch.device('cuda')
        
        if method == 'netvlad':
            from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding

            encoder_dim, encoder = get_backend() # 512, nn.Sequential(*layers)
            checkpoint = torch.load(WEIGHTS[method])
            config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])
            pool_size = int(config['global_params']['num_pcs'])
            model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=True)
            model.load_state_dict(checkpoint['state_dict'])

        elif method == 'cosplace':
            sys.path.append('./cosplace')
            from cosplace.cosplace_model import cosplace_network, layers

            model = cosplace_network.GeoLocalizationNet('ResNet152', 512)
            model_state_dict = torch.load(WEIGHTS[method])
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
                
            elif method == 'convap':
                model = VPRModel(
                    agg_arch='ConvAP',
                    agg_config={'in_channels': 2048,
                                'out_channels': 2048})

        elif method == 'transvpr':
            sys.path.append('./transvpr')
            from transvpr.feature_extractor import Extractor_base
            from transvpr.blocks import POOL

            model = Extractor_base()
            pool = POOL(model.embedding_dim)
            model.add_module('pool', pool)

            checkpoint = torch.load(WEIGHTS[method])

        else:
            raise Exception('Input method is not supported.')
        
        model = model.to(self.device)
        model.eval()
        self.model = model

        self.loader = loader

    def netvlad(self):
        pass

    def cosplace(self):
        pass

    def mixvpr(self):
        pass

    def gem(self):
        pass

    def convap(self):
        pass

    def transvpr(self):
        pass
