from functools import partial
from typing import Tuple

import torch
from timm.models.layers import DropPath, Mlp, PatchEmbed as TimmPatchEmbed

from torch import nn, _assert, Tensor

from utils.helpers import to_2tuple

from mamba_ssm import Mamba
from mamba_ssm import Mamba2


import torch
import torch.nn as nn

class MAMBA2D(nn.Module):
    def __init__(self, input_size: int, state_expansion: int, block_expansion: int, conv_dim: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__()

        self.input_size = input_size
        self.state_expansion = state_expansion
        self.output_size = 2 * input_size if bidirectional else input_size
        self.bidirectional = bidirectional
        self.union = union
        self.num_layers = num_layers
        
        # Create a chain of Mamba blocks based on num_layers
        self.mamba_hf = self.create_mamba(input_size, state_expansion, block_expansion, conv_dim, num_layers)
        self.mamba_hb = self.create_mamba(input_size, state_expansion, block_expansion, conv_dim, num_layers)
        
        self.mamba_vf = self.create_mamba(input_size, state_expansion, block_expansion, conv_dim, num_layers)
        self.mamba_vb = self.create_mamba(input_size, state_expansion, block_expansion, conv_dim, num_layers)

        self.with_vertical = True
        self.with_horizontal = True
        self.with_fc = with_fc

        if with_fc:
            self.fc = self.configure_fc(union, input_size)

    def create_mamba(self, input_size, state_expansion, block_expansion, conv_dim, num_layers):
        """ Helper function to create a chain of Mamba layers. """
        layers = []
        for _ in range(num_layers):
            layers.append(Mamba(
                d_model=input_size,
                d_state=state_expansion,
                d_conv=conv_dim,
                expand=block_expansion,
            ))
        
        return nn.Sequential(*layers).to('cuda' if torch.cuda.is_available() else 'cpu')

    def configure_fc(self, union, input_size):
        """ Helper function to configure fully connected layer. """
        if union == "cat":
            return nn.Linear(2 * self.output_size, input_size)
        elif union in ["add", "vertical", "horizontal"]:
            return nn.Linear(self.output_size, input_size)
        else:
            raise ValueError(f"Unrecognized union: {union}")

    def forward(self, x):
        B, H, W, C = x.shape

        # Horizontal and Vertical Processing
        h, v = None, None
        
        if self.with_horizontal:
            x_h = x.reshape(-1, W, C)
            h1 = self.mamba_hf(x_h)
            if self.bidirectional:
                h2 = self.mamba_hb(torch.flip(x_h, dims=[1]))
                h2 = torch.flip(h2, dims=[1])
                h = torch.cat((h1, h2), dim=-1)
            else:
                h = h1
            h = h.reshape(B, H, W, -1)

        if self.with_vertical:
            x_v = x.permute(0, 2, 1, 3).reshape(-1, H, C)
            v1 = self.mamba_vf(x_v)
            if self.bidirectional:
                v2 = self.mamba_vb(torch.flip(x_v, dims=[1]))
                v2 = torch.flip(v2, dims=[1])
                v = torch.cat((v1, v2), dim=-1)
            else:
                v = v1
            v = v.reshape(B, W, H, -1).permute(0, 2, 1, 3)

        # Final Processing
        if self.with_vertical and self.with_horizontal:
            x = torch.cat([v, h], dim=-1) if self.union == "cat" else (v + h)
        elif self.with_vertical:
            x = v
        elif self.with_horizontal:
            x = h

        if self.with_fc:
            x = self.fc(x)

        return x


class MAMBA2_2D(nn.Module):
    def __init__(self, input_size: int, state_expansion: int, block_expansion: int, conv_dim: int, headdim: int,
                num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                union="cat", with_fc=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = state_expansion
        self.output_size = 2 * input_size if bidirectional else input_size
        self.bidirectional = bidirectional
        self.union = union
        
        self.mamba_hf = self.create_mamba(input_size, state_expansion, block_expansion, conv_dim, num_layers)
        self.mamba_hb = self.create_mamba(input_size, state_expansion, block_expansion, conv_dim, num_layers)
        
        self.mamba_vf = self.create_mamba(input_size, state_expansion, block_expansion, conv_dim, num_layers)
        self.mamba_vb = self.create_mamba(input_size, state_expansion, block_expansion, conv_dim, num_layers)

        self.with_vertical = True
        self.with_horizontal = True
        self.with_fc = with_fc

        if with_fc:
            self.fc = self.configure_fc(union, input_size)

    def create_mamba(self, input_size, state_expansion, block_expansion, conv_dim, headdim):
        """ Helper function to create Mamba2 layer. """
        return Mamba2(
            d_model=input_size,
            d_state=state_expansion,
            d_conv=conv_dim,
            expand=block_expansion,
            headdim=headdim,
        ).to('cuda' if torch.cuda.is_available() else 'cpu')

    def configure_fc(self, union, input_size):
        """ Helper function to configure fully connected layer. """
        if union == "cat":
            return nn.Linear(2 * self.output_size, input_size)
        elif union in ["add", "vertical", "horizontal"]:
            return nn.Linear(self.output_size, input_size)
        else:
            raise ValueError(f"Unrecognized union: {union}")

    def forward(self, x):
        B, H, W, C = x.shape

        # Horizontal and Vertical Processing
        h, v = None, None
        
        if self.with_horizontal:
            x_h = x.reshape(-1, W, C)
            h1 = self.mamba_hf(x_h)
            if self.bidirectional:
                h2 = self.mamba_hb(torch.flip(x_h, dims=[1]))
                h2 = torch.flip(h2, dims=[1])
                h = torch.cat((h1, h2), dim=-1)
            else:
                h = h1
            h = h.reshape(B, H, W, -1)

        if self.with_vertical:
            x_v = x.permute(0, 2, 1, 3).reshape(-1, H, C)
            v1 = self.mamba_vf(x_v)
            if self.bidirectional:
                v2 = self.mamba_vb(torch.flip(x_v, dims=[1]))
                v2 = torch.flip(v2, dims=[1])
                v = torch.cat((v1, v2), dim=-1)
            else:
                v = v1
            v = v.reshape(B, W, H, -1).permute(0, 2, 1, 3)

        # Final Processing
        if self.with_vertical and self.with_horizontal:
            x = torch.cat([v, h], dim=-1) if self.union == "cat" else (v + h)
        elif self.with_vertical:
            x = v
        elif self.with_horizontal:
            x = h

        if self.with_fc:
            x = self.fc(x)

        return x


class Sequencer2DBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3.0, rnn_layer=MAMBA2D, mlp_layer=Mlp,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 num_layers=1, bidirectional=True, union="cat", with_fc=True,
                 drop=0., drop_path=0., state_expansion=None, block_expansion=None, conv_dim=None, headdim=None):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        
        # Normalization
        self.norm1 = norm_layer(dim)

        # Create RNN Layer using a helper method
        self.rnn_tokens = self.create_rnn_layer(
            rnn_layer, dim, state_expansion, block_expansion, conv_dim, headdim,
            num_layers, bidirectional, union, with_fc
        )

        # Drop Path to manage stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Normalization and MLP
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def create_rnn_layer(self, rnn_layer, dim, state_expansion, block_expansion, conv_dim, headdim,
                         num_layers, bidirectional, union, with_fc):
        """ Helper method to create RNN layer with common parameters. """
        assert state_expansion is not None and block_expansion is not None and conv_dim is not None, \
            f"{rnn_layer.__name__} requires 'state_expansion', 'block_expansion', and 'conv_dim'"

        if rnn_layer == MAMBA2D:
            return rnn_layer(dim, state_expansion, block_expansion, conv_dim,
                             num_layers=num_layers, bidirectional=bidirectional,
                             union=union, with_fc=with_fc)
        elif rnn_layer == MAMBA2_2D:
            return rnn_layer(dim, state_expansion, block_expansion, conv_dim, headdim,
                             num_layers=num_layers, bidirectional=bidirectional,
                             union=union, with_fc=with_fc)
        else:
            raise ValueError(f"Unrecognized RNN layer: {rnn_layer}")

    def forward(self, x):
        # Forward pass through RNN tokens and MLP channels with DropPath
        x = x + self.drop_path(self.rnn_tokens(self.norm1(x)))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class PatchEmbed(TimmPatchEmbed):
   
    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.norm(x)
        return x

class Shuffle(nn.Module):
    """ Optimized Shuffle class """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            B, H, W, C = x.shape
            # Shuffle elements by reshaping and using torch.randperm
            r = torch.randperm(H * W, device=x.device)
            x = x.reshape(B, -1, C)[:, r, :].reshape(B, H, W, C)
        return x


class Downsample2D(nn.Module):
    """ Optimized Downsampling with a simplified reshape """
    def __init__(self, input_dim, output_dim, patch_size):
        super().__init__()
        self.down = nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Directly permute before and after the downsampling operation
        x = x.permute(0, 3, 1, 2)  # Change format from BHWC to BCHW for Conv2D
        x = self.down(x)
        x = x.permute(0, 2, 3, 1)  # Back to BHWC format
        return x
    
    