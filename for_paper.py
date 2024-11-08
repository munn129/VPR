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
from PIL import Image

sys.path.append('./mixvpr')
from mixvpr.main import VPRModel

WEIGHT = './pretrained_models/resnet50_MixVPR_512_channels(256)_rows(2).ckpt'
IMAGES = '/media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/front'
IDX = 7010
TH_MIN = 0
TH_MAX = 50
ALPHA = 1

def overlay(image, att_map, alpha = 0.5):
    image_np = np.array(image).astype(np.float32) / 255.0

    att_map = np.array(Image.fromarray(att_map).resize(image.size, resample=Image.BICUBIC))

    att_map_colored = np.zeros((att_map.shape[0], att_map.shape[1], 3), dtype=np.float32) 
    att_map_colored[..., 2] = att_map / 255.0

    att_map_colored = att_map_colored * alpha

    att_map_colored_img = (np.clip(att_map_colored * 255, 0, 255)).astype(np.uint8) 

    combined = image_np + att_map_colored
    combined = np.clip(combined * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(combined), Image.fromarray(att_map_colored_img)

def main():

    image_dir = sorted(list(Path(IMAGES).glob('*.png')))[IDX]
    image = Image.open(image_dir).convert('RGB')
    # image.save('target.png')

    device = torch.device('cuda')

    mixvpr = VPRModel(
        backbone_arch='resnet50',
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
            'is_mix' : False
        }
    )

    transforms = tvf.Compose(
        [tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC),
         tvf.ToTensor(),
         tvf.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])]
    )

    state_dict = torch.load(WEIGHT)
    mixvpr.load_state_dict(state_dict)
    mixvpr = mixvpr.to(device)
    mixvpr.eval()

    image_tensor = transforms(image).unsqueeze(0)

    with torch.no_grad():
        _, attention_mask = mixvpr(image_tensor.to(device))

        attention_mask = attention_mask.view(20,20)

    torch.cuda.empty_cache()

    attention_mask = attention_mask.detach().cpu().numpy()

    att_map = (attention_mask - attention_mask.min()) / (attention_mask.max() - attention_mask.min())
    att_map = (att_map * 255).astype(np.uint8)

    # th = 200
    att_map[att_map < TH_MIN] = 0
    # att_map[att_map > TH_MAX] = 0

    # att_map = Image.fromarray(att_map)

    # att_map.save('map.png')
    
    # print(attention_mask)

    # Overlay the attention map on top of the image
    overlayed_map, att = overlay(image, att_map, alpha=ALPHA)
    
    # Save the final overlayed image
    overlayed_map.save('overlayed.png')
    att.save('att.png')
    

if __name__ == '__main__':
    main()