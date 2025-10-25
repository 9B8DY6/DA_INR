import torch
import torch.nn as nn
import torch.nn.functional as F
from torchngp.nerf.renderer import NeRFRenderer
import math

class NeRFNetwork(NeRFRenderer):
    def __init__(self,

                num_layers=2,
                hidden_dim=64,
                geo_feat_dim=15,
                num_layers_color=3,
                hidden_dim_color=64,


                num_levels=16,
                level_dim=2,
                base_resolution=16,
                log2_hashmap_size=19,
                per_level_scale=2,

                num_layers_bg=2,
                hidden_dim_bg=64,
                num_layers_deform=5, # a deeper MLP is very necessary for performance.
                hidden_dim_deform=128,
                bound=1,

                **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # deformation network
        self.num_layers_deform = num_layers_deform
        self.hidden_dim_deform = hidden_dim_deform

        from torchngp.gridencoder import GridEncoder
        from torchngp.freqencoder import FreqEncoder

        self.encoder_deform = FreqEncoder(input_dim=2, degree=10)
        self.in_dim_deform = self.encoder_deform.output_dim
        self.encoder_time = FreqEncoder(input_dim=1, degree=6)
        self.in_dim_time = self.encoder_time.output_dim    

        deform_net = []
        for l in range(num_layers_deform):
            if l == 0:
                in_dim = self.in_dim_deform + self.in_dim_time # grid dim + time
            else:
                in_dim = hidden_dim_deform
            
            if l == num_layers_deform - 1:
                out_dim = 2 # deformation for xy
            else:
                out_dim = hidden_dim_deform
            
            deform_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.deform_net = nn.ModuleList(deform_net)


        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        
        input_dimension = 2
        self.encoder = GridEncoder(input_dim=input_dimension, num_levels=num_levels, level_dim=level_dim, per_level_scale=per_level_scale,base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=2048*self.bound, gridtype='tiled', align_corners=False)
        self.in_dim = self.encoder.output_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim + self.in_dim_time + self.in_dim_deform + 128 # concat everything
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = hidden_dim
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = hidden_dim_color
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 2 # real, imag
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
    
    def test(self, train_x, eval_x, t, cell=None, features=None):
        # deform
        enc_ori_x = self.encoder_deform(train_x, bound=self.bound) # [N, C]
        enc_t = self.encoder_time(t) # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(train_x.shape[0], 1) # [1, C'] --> [N, C']
        deform = torch.cat([enc_ori_x, enc_t], dim=1) # [N, C + C']

        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)
        
        x = train_x + deform
        
        lrres, timedim = enc_t.shape
        enc_t = enc_t[0].clone().reshape(1, timedim).repeat(eval_x.shape[0], 1)

        LR_res = int(math.sqrt(x.shape[0]))
        HR_res = int(math.sqrt(eval_x.shape[0]))
        encoded_x_feat_shape = enc_ori_x.shape[-1]

        interp_mode = 'nearest'
        hr_coords = F.grid_sample(x.reshape(1, LR_res, LR_res, -1).permute(0,3,1,2), eval_x.reshape(1,HR_res, HR_res,-1), mode=interp_mode, align_corners=False) 
        hr_coords = hr_coords.permute(0,2,3,1)
        interpolated_features = F.grid_sample(features, eval_x.reshape(1, HR_res, HR_res,-1), mode='bilinear', align_corners=False) 
        interpolated_features = interpolated_features.permute(0,2,3,1)
        enc_ori_x = F.grid_sample(enc_ori_x.reshape(1, LR_res, LR_res, -1).permute(0,3,1,2), eval_x.reshape(1,HR_res, HR_res,-1), mode=interp_mode, align_corners=False)[0].permute(2,1,0).reshape(-1, encoded_x_feat_shape)
        x = hr_coords.permute(0,2,1,3).reshape(eval_x.shape[0], -1)
        interpolated_features = interpolated_features.permute(0,1,2,3).reshape(eval_x.shape[0], -1)

        # sigma
        x = self.encoder(x, bound=self.bound)
        h = torch.cat([x, enc_ori_x, enc_t, interpolated_features], dim=-1)

        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        real_imag = h

        return real_imag

    def forward(self, x, t, cell=None):
        enc_ori_x = self.encoder_deform(x, bound=self.bound) # [N, C]
        enc_t = self.encoder_time(t) # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1) # [1, C'] --> [N, C']
        deform = torch.cat([enc_ori_x, enc_t], dim=1) # [N, C + C']

        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)
        
        x = x + deform

        # sigma
        x = self.encoder(x, bound=self.bound)
        h = torch.cat([x, enc_ori_x, enc_t], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        real_imag = h

        return real_imag

    # optimizer utils
    def get_params(self, lr, lr_net):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_deform.parameters(), 'lr': lr},
            {'params': self.encoder_time.parameters(), 'lr': lr},
            {'params': self.deform_net.parameters(), 'lr': lr_net},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr_net})
        
        return params
    