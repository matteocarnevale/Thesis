import torch
from torch import nn
from mamba_ssm import Mamba2



class MAMBA2D(nn.Module):
    def __init__(self, input_size: int,
                 num_layers: int = 1, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__()

        self.input_size = input_size
        self.output_size = 2 * input_size if bidirectional else input_size
        self.bidirectional = bidirectional
        self.union = union
        
        self.mamba_h = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=input_size, # Model dimension d_model
            d_state=128,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.mamba_v = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=input_size, # Model dimension d_model
            d_state=128,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
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

    
B, L, W, C = 2, 64, 64, 512
x = torch.randn(B, L, W, C).to("cuda")

model = MAMBA2D(
    input_size = C,
    num_layers = 1, 
    bidirectional = True,
    union = "cat", 
    with_fc = True
).to("cuda")

y = model(x)
print(y)