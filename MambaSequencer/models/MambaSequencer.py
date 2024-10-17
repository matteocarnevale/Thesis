import math
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.models.layers import lecun_normal_, Mlp
from timm.models.registry import register_model
from torch import nn

from models.layers import Sequencer2DBlock, PatchEmbed, Downsample2D, MAMBA2_2D, MAMBA2D


import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.helpers import named_apply, build_model_with_cfg
from timm.models.layers import DropPath, lecun_normal_

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': DEFAULT_CROP_PCT, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }

# Function to initialize weights in the model
def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            lecun_normal_(module.weight) if flax else nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
        stdv = 1.0 / math.sqrt(module.hidden_size)
        for weight in module.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

# Helper function to create a stage with multiple blocks
def get_stage(index, layers, patch_sizes, embed_dims, state_expansion, block_expansion, conv_dim, headdim, mlp_ratios, block_layer, rnn_layer, mlp_layer,
              norm_layer, act_layer, num_layers, bidirectional, union, with_fc, drop=0., drop_path_rate=0., **kwargs):

    assert rnn_layer in [MAMBA2D, MAMBA2_2D], "Invalid rnn_layer"
    assert len(state_expansion) == len(embed_dims) == len(mlp_ratios) == len(layers), "Lengths of parameters must match"

    blocks = []
    num_layers_up_to_index = sum(layers[:index])
    total_layers = sum(layers) - 1

    for block_idx in range(layers[index]):
        drop_path = drop_path_rate * (block_idx + num_layers_up_to_index) / total_layers
        block_kwargs = {
            'dim': embed_dims[index],
            'state_expansion': state_expansion[index],
            'block_expansion': block_expansion[index],
            'conv_dim': conv_dim[index],
            'mlp_ratio': mlp_ratios[index],
            'rnn_layer': rnn_layer,
            'mlp_layer': mlp_layer,
            'norm_layer': norm_layer,
            'act_layer': act_layer,
            'num_layers': num_layers,
            'bidirectional': bidirectional,
            'union': union,
            'with_fc': with_fc,
            'drop': drop,
            'drop_path': drop_path,
        }
        if rnn_layer == MAMBA2_2D:
            block_kwargs['headdim'] = headdim[index]
        
        blocks.append(block_layer(**block_kwargs))

    # Downsample layer between stages
    if index < len(embed_dims) - 1:
        blocks.append(Downsample2D(embed_dims[index], embed_dims[index + 1], patch_sizes[index + 1]))

    return nn.Sequential(*blocks)

# The main Sequencer2D model class
class Sequencer2D(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            layers=[4, 3, 8, 3],
            patch_sizes=[7, 2, 1, 1],
            embed_dims=[192, 384, 384, 384],
            mlp_ratios=[3.0, 3.0, 3.0, 3.0],
            block_layer=Sequencer2DBlock,
            rnn_layer=MAMBA2_2D,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            num_rnn_layers=1,
            bidirectional=True,
            union="cat",
            with_fc=True,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
            state_expansion=[64, 64, 64, 64],   
            block_expansion=[2, 2, 2, 2],   
            conv_dim=[4, 4, 4, 4],
            headdim=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dims[0]
        self.embed_dims = embed_dims

        # Patch Embedding using Timm's implementation
        self.stem = PatchEmbed(
            img_size=img_size, patch_size=patch_sizes[0], in_chans=in_chans,
            embed_dim=embed_dims[0], norm_layer=norm_layer if stem_norm else None,
            flatten=False)

        # Build all stages (blocks)
        self.blocks = nn.Sequential(*[
            get_stage(
                i, layers, patch_sizes, embed_dims, state_expansion, block_expansion,
                conv_dim, headdim, mlp_ratios, block_layer=block_layer, rnn_layer=rnn_layer,
                mlp_layer=mlp_layer, norm_layer=norm_layer, act_layer=act_layer,
                num_layers=num_rnn_layers, bidirectional=bidirectional,
                union=union, with_fc=with_fc, drop=drop_rate, drop_path_rate=drop_path_rate,
            ) for i in range(len(embed_dims))
        ])

        # Normalization and classification head
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], self.num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self.init_weights(nlhb=nlhb)

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=(1, 2))  # Global Average Pooling
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    return state_dict


default_cfgs = {
    'sequencer2d_s': _cfg(
        url='',  # URL for pretrained weights (empty for now)
        num_classes=1000,
        input_size=(3, 224, 224),
        crop_pct=DEFAULT_CROP_PCT,
        interpolation='bicubic',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        first_conv='stem.proj',
        classifier='head'
    ),
}


def _create_sequencer2d(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Sequencer2D models.')

    model = build_model_with_cfg(
        Sequencer2D, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def sequencer2d_s(pretrained=False, **kwargs):
    model_args = dict(
        layers=[1, 1, 1, 1],
        patch_sizes=[7, 4, 1, 1],
        embed_dims=[192, 384, 384, 384],
        mlp_ratios=[2.0, 2.0, 2.0, 2.0],
        rnn_layer=MAMBA2D,
        bidirectional=True,
        union="cat",
        with_fc=True,
        state_expansion=[16, 32, 64, 64],
        block_expansion=[2, 2, 2, 2],   
        conv_dim=[4, 4, 4, 4],
        **kwargs
    )
    model = _create_sequencer2d('sequencer2d_s', pretrained=pretrained, **model_args)
    return model
