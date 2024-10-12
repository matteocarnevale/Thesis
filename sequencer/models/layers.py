#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

from functools import partial
from typing import Tuple

import torch
from timm.models.layers import DropPath, Mlp, PatchEmbed as TimmPatchEmbed

from torch import nn, _assert, Tensor

from utils.helpers import to_2tuple

from mamba_ssm import Mamba
from mamba_ssm import Mamba2


class RNNIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RNNIdentity, self).__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        return x, None


class RNNBase(nn.Module):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True):
        super().__init__()
        self.rnn = RNNIdentity()

    def forward(self, x):
        B, H, W, C = x.shape
        x, _ = self.rnn(x.view(B, -1, C))
        return x.view(B, H, W, -1)


class RNN(RNNBase):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,
                          bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)


class GRU(RNNBase):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          bias=bias, bidirectional=bidirectional)


class LSTM(RNNBase):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           bias=bias, bidirectional=bidirectional)


class RNN2DBase(nn.Module):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2 * hidden_size if bidirectional else hidden_size
        self.union = union

        self.with_vertical = True
        self.with_horizontal = True
        self.with_fc = with_fc

        if with_fc:
            if union == "cat":
                self.fc = nn.Linear(2 * self.output_size, input_size)
            elif union == "add":
                self.fc = nn.Linear(self.output_size, input_size)
            elif union == "vertical":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_horizontal = False
            elif union == "horizontal":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_vertical = False
            else:
                raise ValueError("Unrecognized union: " + union)
        elif union == "cat":
            pass
            if 2 * self.output_size != input_size:
                raise ValueError(f"The output channel {2 * self.output_size} is different from the input channel {input_size}.")
        elif union == "add":
            pass
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
        elif union == "vertical":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_horizontal = False
        elif union == "horizontal":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_vertical = False
        else:
            raise ValueError("Unrecognized union: " + union)

        self.rnn_v = RNNIdentity()
        self.rnn_h = RNNIdentity()

    def forward(self, x):
        B, H, W, C = x.shape

        if self.with_vertical:
            v = x.permute(0, 2, 1, 3)
            v = v.reshape(-1, H, C)
            v, _ = self.rnn_v(v)
            v = v.reshape(B, W, H, -1)
            v = v.permute(0, 2, 1, 3)

        if self.with_horizontal:
            h = x.reshape(-1, W, C)
            h, _ = self.rnn_h(h)
            h = h.reshape(B, H, W, -1)

        if self.with_vertical and self.with_horizontal:
            if self.union == "cat":
                x = torch.cat([v, h], dim=-1)
            else:
                x = v + h
        elif self.with_vertical:
            x = v
        elif self.with_horizontal:
            x = h

        if self.with_fc:
            x = self.fc(x)

        return x


class RNN2D(RNN2DBase):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True, nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)
        if self.with_horizontal:
            self.rnn_h = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)


class LSTM2D(RNN2DBase):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_horizontal:
            self.rnn_h = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)


class GRU2D(RNN2DBase):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_horizontal:
            self.rnn_h = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)

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
        
        
        self.mamba_h = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=input_size, # Model dimension d_model
            d_state=state_expansion,  # SSM state expansion factor
            d_conv=conv_dim,    # Local convolution width
            expand=block_expansion,    # Block expansion factor
        )

        self.mamba_v = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=input_size, # Model dimension d_model
            d_state=state_expansion,  # SSM state expansion factor
            d_conv=conv_dim,    # Local convolution width
            expand=block_expansion,    # Block expansion factor
        )
        

        self.with_vertical = True
        self.with_horizontal = True
        self.with_fc = with_fc

        if with_fc:
            if union == "cat":
                self.fc = nn.Linear(2 * self.output_size, input_size)
            elif union == "add":
                self.fc = nn.Linear(self.output_size, input_size)
            elif union == "vertical":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_horizontal = False
            elif union == "horizontal":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_vertical = False
            else:
                raise ValueError("Unrecognized union: " + union)
        elif union == "cat":
            pass
            if 2 * self.output_size != input_size:
                raise ValueError(f"The output channel {2 * self.output_size} is different from the input channel {input_size}.")
        elif union == "add":
            pass
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
        elif union == "vertical":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_horizontal = False
        elif union == "horizontal":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_vertical = False
        else:
            raise ValueError("Unrecognized union: " + union)

    def forward(self, x):
        
        B, H, W, C = x.shape

        #Horizontal Processing
        if self.with_horizontal:
            x_h = x.reshape(-1, W, C)
            
            if (self.bidirectional):
                h1 = self.mamba_h(x_h)
                reverse_xh = torch.flip(x_h, dims=[1]) #Flipping along the sequence dimension H for vertical, W for horizontal
                h2 = self.mamba_h(reverse_xh)
                h2 = torch.flip(h2, dims=[1])
                h = torch.cat((h1, h2), dim=-1)
                h = h.reshape(B, H, W, -1)

            else:
                h = self.mamba_h(x_h)
                h = h.reshape(B, H, W, -1)

        #Vertical Processing
        if self.with_vertical:
            x_v = x.permute(0, 2, 1, 3)
            x_v = x_v.reshape(-1, H, C)
            
            if (self.bidirectional):
                v1 = self.mamba_v(x_v)
                reverse_xv = torch.flip(x_v, dims=[1]) #Flipping along the sequence dimension H for vertical, W for horizontal
                v2 = self.mamba_v(reverse_xv)
                v2 = torch.flip(v2, dims=[1])
                v = torch.cat((v1, v2), dim=-1)
                v = v.reshape(B, W, H, -1)
                v = v.permute(0, 2, 1, 3)

            else:
                v = self.mamba_v(x_v)
                v = v.reshape(B, W, H, -1)
                v = v.permute(0, 2, 1, 3)

        #Final Processing
        if self.with_vertical and self.with_horizontal:
            if self.union == "cat":
                x = torch.cat([v, h], dim=-1)
            else:
                x = v + h
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
        
        self.mamba_h = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=input_size, # Model dimension d_model
            d_state=state_expansion,  # SSM state expansion factor
            d_conv=conv_dim,    # Local convolution width
            expand=block_expansion,    # Block expansion factor
            headdim=headdim,
        )

        self.mamba_v = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=input_size, # Model dimension d_model
            d_state=state_expansion,  # SSM state expansion factor
            d_conv=conv_dim,    # Local convolution width
            expand=block_expansion,    # Block expansion factor
            headdim=headdim,
        )
        

        self.with_vertical = True
        self.with_horizontal = True
        self.with_fc = with_fc

        if with_fc:
            if union == "cat":
                self.fc = nn.Linear(2 * self.output_size, input_size)
            elif union == "add":
                self.fc = nn.Linear(self.output_size, input_size)
            elif union == "vertical":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_horizontal = False
            elif union == "horizontal":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_vertical = False
            else:
                raise ValueError("Unrecognized union: " + union)
        elif union == "cat":
            pass
            if 2 * self.output_size != input_size:
                raise ValueError(f"The output channel {2 * self.output_size} is different from the input channel {input_size}.")
        elif union == "add":
            pass
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
        elif union == "vertical":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_horizontal = False
        elif union == "horizontal":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_vertical = False
        else:
            raise ValueError("Unrecognized union: " + union)


    def forward(self, x):
        
        B, H, W, C = x.shape

        #Horizontal Processing
        if self.with_horizontal:
            x_h = x.reshape(-1, W, C)
            
            if (self.bidirectional):
                h1 = self.mamba_h(x_h)
                reverse_xh = torch.flip(x_h, dims=[0])
                h2 = self.mamba_h(reverse_xh)
                h2 = torch.flip(h2, dims=[0])
                h = torch.cat((h1, h2), dim=-1)
                h = h.reshape(B, H, W, -1)

            else:
                h = self.mamba_h(x_h)
                h = h.reshape(B, H, W, -1)

        #Vertical Processing
        if self.with_vertical:
            x_v = x.permute(0, 2, 1, 3)
            x_v = x_v.reshape(-1, H, C)
            
            if (self.bidirectional):
                v1 = self.mamba_v(x_v)
                reverse_xv = torch.flip(x_v, dims=[0])
                v2 = self.mamba_v(reverse_xv)
                v2 = torch.flip(v2, dims=[0])
                v = torch.cat((v1, v2), dim=-1)
                v = v.reshape(B, W, H, -1)
                v = v.permute(0, 2, 1, 3)

            else:
                v = self.mamba_v(x_v)
                v = v.reshape(B, W, H, -1)
                v = v.permute(0, 2, 1, 3)

        #Final Processing
        if self.with_vertical and self.with_horizontal:
            if self.union == "cat":
                x = torch.cat([v, h], dim=-1)
            else:
                x = v + h
        elif self.with_vertical:
            x = v
        elif self.with_horizontal:
            x = h

        if self.with_fc:
            x = self.fc(x)

        return x



class VanillaSequencerBlock(nn.Module):
    def __init__(self, dim, hidden_size, mlp_ratio=3.0, rnn_layer=LSTM, mlp_layer=Mlp,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 num_layers=1, bidirectional=True, drop=0., drop_path=0.):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm1 = norm_layer(dim)
        self.rnn_tokens = rnn_layer(dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.rnn_tokens(self.norm1(x)))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Sequencer2DBlock(nn.Module):
    def __init__(self, dim, hidden_size=None, mlp_ratio=3.0, rnn_layer=MAMBA2D, mlp_layer=Mlp,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 num_layers=1, bidirectional=True, union="cat", with_fc=True,
                 drop=0., drop_path=0., state_expansion=None, block_expansion=None, conv_dim=None, headdim=None):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm1 = norm_layer(dim)
        if rnn_layer == MAMBA2D:
            assert state_expansion is not None and block_expansion is not None and conv_dim is not None, "MAMBA2D requires 'state_expansion', 'block_expansion', and 'conv_dim'"
            self.rnn_tokens = rnn_layer(dim, state_expansion, block_expansion, conv_dim, 
                                        num_layers=num_layers, bidirectional=bidirectional,
                                        union=union, with_fc=with_fc)
        elif rnn_layer == MAMBA2_2D:
            assert state_expansion is not None and block_expansion is not None and conv_dim is not None, "MAMBA2D requires 'state_expansion', 'block_expansion', and 'conv_dim'"
            self.rnn_tokens = rnn_layer(dim, state_expansion, block_expansion, conv_dim, headdim, 
                                        num_layers=num_layers, bidirectional=bidirectional,
                                        union=union, with_fc=with_fc)
        else:
            self.rnn_tokens = rnn_layer(dim, hidden_size, num_layers=num_layers, 
                                        bidirectional=bidirectional, union=union, with_fc=with_fc)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            B, H, W, C = x.shape
            r = torch.randperm(H * W)
            x = x.reshape(B, -1, C)
            x = x[:, r, :].reshape(B, H, W, -1)
        return x


class Downsample2D(nn.Module):
    def __init__(self, input_dim, output_dim, patch_size):
        super().__init__()
        self.down = nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.down(x)
        x = x.permute(0, 2, 3, 1)
        return x
