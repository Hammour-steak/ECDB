import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms as transforms

import functools

from .module_util import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,
    LinearAttention, Attention,
    PreNorm, Residual,
    zero_module)

from .DenoisingUNet_arch import ConditionalUNet

class ControlNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, depth, upscale=1, hint_channels=3):
        super().__init__()

        self.depth = depth
        self.upscale = upscale # not used

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        # zero module
        self.zero_convs = nn.ModuleList()

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc*2, nf, 7)

        self.init_zero_conv = self.make_zero_conv(nf)
        
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False


        t_values = torch.linspace(0, 1, 101)
        t_values = t_values.to('cuda')
        self.con_fuse_schedule = self.function(t_values)

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_hint_block = nn.Sequential(
            nn.Conv2d(hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1),
            nn.SiLU(),
            zero_module(nn.Conv2d(256, nf, 3, padding=1))
        )

        self.input_feature_block = nn.Sequential(
            nn.Conv2d(hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1),
            nn.SiLU(),
            zero_module(nn.Conv2d(256, nf, 3, padding=1))
        )

        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i+1))
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth-1) else default_conv(dim_in, dim_out)
            ]))

            self.zero_convs.append(nn.ModuleList([self.make_zero_conv(dim_in), self.make_zero_conv(dim_in)]))


        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.mid_zero_conv = self.make_zero_conv(mid_dim)


    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def make_zero_conv(self, channels):
        return zero_module(nn.Conv2d(channels, channels, 1, padding=0))
    
    def function(self, t):
        return torch.exp(-5 * t) - torch.exp(torch.tensor(-5, device='cuda')) * t

    def forward(self, xt, cond, time, hint):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)
        
        x = xt - cond
        feature = x
        x = torch.cat([x, cond], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        hint = self.check_image_size(hint, H, W)
        feature = self.check_image_size(feature, H, W)
        outs = []

        guided_hint = self.input_hint_block(hint)
        guided_feature = self.input_feature_block(feature)
        x = self.init_conv(x)

        feature_weight = self.con_fuse_schedule[time].unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # 保存图像
        # transform = transforms.ToPILImage()
        # t = str(time[0])
        # fe = guided_feature
        # fe = fe[:, 61:64, :, :]
        # img = transform(torch.squeeze(fe, 0) * 255)
        # img.save(f"/home/yueconghan/pzw/GOUB/figs/image_{t}.png")

        x = x + guided_hint + feature_weight * guided_feature
        outs.append(self.init_zero_conv(x))
 
        t = self.time_mlp(time)

        for (b1, b2, attn, downsample), (zero_conv1, zero_conv2) in zip(self.downs, self.zero_convs):
            x = b1(x, t)
            outs.append(zero_conv1(x))

            x = b2(x, t)
            x = attn(x)
            outs.append(zero_conv2(x))

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        outs.append(self.mid_zero_conv(x))
        return outs

class ControlledConditionalUNet(ConditionalUNet):
    def forward(self, xt, cond, time, control=None):
        hs = []
        with torch.no_grad():
            if isinstance(time, int) or isinstance(time, float):
                time = torch.tensor([time]).to(xt.device)
            
            x = xt - cond
            x = torch.cat([x, cond], dim=1)

            H, W = x.shape[2:]
            x = self.check_image_size(x, H, W)

            x = self.init_conv(x)
            x_ = x.clone()

            t = self.time_mlp(time)


            for b1, b2, attn, downsample in self.downs:
                x = b1(x, t)
                hs.append(x)

                x = b2(x, t)
                x = attn(x)
                hs.append(x)

                x = downsample(x)

            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)
        
        if control is not None:
            x += control.pop()

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, hs.pop() + control.pop()], dim=1)
            x = b1(x, t)
            
            x = torch.cat([x, hs.pop() + control.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W]
        return x

class ControlConditional(nn.Module):
    def __init__(self, in_nc, out_nc, nf, depth, upscale=1, hint_channels=3):
        super().__init__()

        self.control_model = ControlNet(in_nc=in_nc, out_nc=out_nc, nf=nf, depth=depth, upscale=upscale, hint_channels=hint_channels)
        self.unet_model = ControlledConditionalUNet(in_nc=in_nc, out_nc=out_nc, nf=nf, depth=depth, upscale=upscale)

    def load_ckpt(self, ckpt_path, mode='train'):
        assert mode in ['train', 'test']

        if mode == 'train':
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            self.unet_model.load_state_dict(ckpt)
            self.control_model.load_state_dict(ckpt, strict=False)
        else:
            pass

    def forward(self, xt, cond, time):
        control = self.control_model(xt, cond, time, cond)
        eps = self.unet_model(xt, cond, time, control)
        return eps

if __name__ == '__main__':
    device = 'cuda:0'
    xt = torch.randn(4, 3, 128, 128).to(device)
    cond = torch.randn(4, 3, 128, 128).to(device)
    time=torch.randint(100, (4,)).to(device)
    hint = torch.randn(4, 3, 128, 128).to(device)
    model = ControlConditional(3, 3, 64)
    model.load_ckpt('/home/yueconghan/datasets/SR(goub).pth')
    model.to(device)
    out = model(xt, cond, time, hint)
    print(out.shape)

