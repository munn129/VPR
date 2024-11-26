import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 is_mix=False) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)
        self.is_mix = is_mix

    def forward(self, x):


        # x: (1,1024,20,20)
        x = x.flatten(2)

        # x: (1,1024,400)
        # x = self.mix(x)
        # x: (1,1024,400)

        if self.is_mix:
            return x
        
        # attention_mask = F.softmax(x, dim = -1)
        # attention_mask = attention_mask.sum(dim=1)
        # attention_mask = x.sum(dim=1)

        
        x = x.permute(0, 2, 1)
        # x: (1,400,1024)
        x = self.channel_proj(x)
        # x: (1,400,1024)
        x = x.permute(0, 2, 1)
        # x: (1,1024,400)

        # attention_mask = F.softmax(x, dim = -1)
        # # attention_mask: (1, 1024,400)
        # attention_mask = attention_mask.sum(dim=1)
        # attention_mask /= attention_mask.shape[1]
        # # # attention_mask: (1,400)

        x = self.row_proj(x)
        # x: (1,1024,4)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        # x: (1, 4096)
        return x



# -------------------------------------------------------------------------------

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    # x = torch.randn(1, 2048, 10, 10)
    # agg = MixVPR(
    #     in_channels=2048,
    #     in_h=10,
    #     in_w=10,
    #     out_channels=1024,
    #     mix_depth=4,
    #     mlp_ratio=1,
    #     out_rows=4)
    
    # y = agg.mix(x.view(1,2048,100))
    # print(y.shape)
    
    # x = torch.randn(1, 1024, 20, 20)
    # agg = MixVPR(
    #     in_channels=1024,
    #     in_h=20,
    #     in_w=20,
    #     out_channels=1024,
    #     mix_depth=4,
    #     mlp_ratio=1,
    #     out_rows=4)

    agg = MixVPR(
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=256,
        mix_depth=4,
        mlp_ratio=1,
        out_rows=2,
        is_mix=True)
    
    x = torch.randn(1,1024,400)
    y = agg(x)
    print(y.shape)

    # print_nb_params(agg)
    # output, att = agg(x)
    # print(output.shape, att.shape)


if __name__ == '__main__':
    main()
