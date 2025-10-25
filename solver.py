import os
import json
import time
import scipy.io as sio
import numpy as np
import itertools

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

from utils.common_utils import *
from Mypnufft_mc_func_cardiac import *

import random
from models.mdsr import make_mdsr
import models.models as ms

class Solver():
    def __init__(self, module, opt):        
        self.opt = opt
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
             
        self.prepare_gt()        
        
        self.step = 0
        self.t1, self.t2 = None, None
        self.best_psnr, self.best_psnr_step = 0, 0
        self.best_ssim, self.best_ssim_step = 0, 0

        self.feat_extractor = make_mdsr().to(self.dev)

        self.net = module.NeRFNetwork(
                num_levels=opt.num_levels,
                level_dim=opt.level_dim,
                base_resolution=opt.base_resolution,
                log2_hashmap_size=opt.log2_hashmap_size
        ).to(self.dev)

        p = get_params('net', self.net, None) 
        s  = sum([np.prod(list(pnb.size())) for pnb in p])
        print ('# params: %d' % s)

        x, y = np.meshgrid(np.linspace(-1,1,self.gt_img_size), np.linspace(-1,1,self.gt_img_size))
        coords = torch.tensor(np.concatenate([x[...,None], y[...,None]], -1)).to(torch.float32)
        self.coords_eval_spatial_int = torch.reshape(coords, [-1, coords.shape[-1]]).to(self.dev)
        time = np.linspace(0, 1, self.Nfr).astype(np.float32)
        self.t = time

        p = itertools.chain(filter(lambda p: p.requires_grad, self.net.parameters()))

        self.optimizer = torch.optim.AdamW(p, opt.lr)  
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                    step_size=opt.step_size, 
                                                    gamma=opt.gamma)
        if opt.isresume is not None:
            self.load(opt.isresume)

    def model_freeze(self, ):
        for i, (name, param) in enumerate(self.feat_extractor.named_parameters()):
            param.requires_grad = False

    def feature_extract(self, index_frame):
        self.syn_radial_img_torch = torch.tensor((self.minmaxnorm(self.syn_radial_img)-0.5)/0.5).to(torch.float32).to(self.dev)
        return self.feat_extractor(self.syn_radial_img_torch[...,index_frame][None][None].repeat(1,3,1,1))

    def dataconsitencyloss(self, x, y):
        self.loss_fn = nn.L1Loss()
        return self.loss_fn(x, y)

    def fit(self):
        opt = self.opt
        self.writer = SummaryWriter(opt.ckpt_root)   
        self.t1 = time.time()    
        step = self.step

        while step < opt.max_steps:
            # randomly pick frames to train (batch, default = 1)
            if opt.temporal_interpolation:
                idx_fr = random.randrange(1, self.Nfr, opt.down_ratio) # only odd
                diff = int(opt.down_ratio)
            else:
                idx_fr = np.random.randint(0, self.Nfr)
                diff = 1                
            
            
            if idx_fr - diff < 0:
                idx_fr_4_time_v1 = idx_fr+diff
                idx_fr_4_time_v2 = idx_fr+diff
            elif idx_fr + diff >= self.Nfr:
                idx_fr_4_time_v1 = idx_fr-diff
                idx_fr_4_time_v2 = idx_fr-diff
            else:
                idx_fr_4_time_v1 = idx_fr-diff
                idx_fr_4_time_v2 = idx_fr+diff                                

            idx_fr_4_time = idx_fr % self.Nfr               
            frame_index = self.t[idx_fr_4_time]


            sr_ratio = self.opt.sr_ratio
            
            self.prepare_dataset(sr_ratio)

            x, y = np.meshgrid(np.linspace(-1,1, self.img_size), np.linspace(-1,1, self.img_size))
            coords = torch.tensor(np.concatenate([x[...,None], y[...,None]], -1)).to(torch.float32)
            self.coords_sr = torch.reshape(coords, [-1, coords.shape[-1]]).to(self.dev)
            
            if isinstance(frame_index, (float,np.float32)):
                time_idx = torch.tensor(frame_index).unsqueeze(0).unsqueeze(1).to(self.dev)
            elif torch.is_tensor(frame_index):
                time_idx = frame_index.unsqueeze(0).unsqueeze(1).to(self.dev)
            
            
            self.model_freeze()
            features_v1 = self.feature_extract(int(idx_fr_4_time_v1))
            features_v2 = self.feature_extract(int(idx_fr_4_time_v2))
            features = torch.cat([features_v1,features_v2], dim=1)
            out_sp = self.net.test(self.coords_sr, self.coords_sr, time_idx, cell=None, features=features).reshape(self.img_size, self.img_size, -1)

            angle = self.set_ang[idx_fr* self.Nfibo * self.Nvec:(idx_fr+1) * self.Nfibo * self.Nvec]
            self.mynufft.X = out_sp
            out_kt = self.mynufft(angle, angle.shape[0]//self.Nvec, self.Nvec, self.Nc, self.coil, self.denc[:,:angle.shape[0]//self.Nvec,:]).reshape(-1,2)
            gt_kt = self.syn_radial_ri_ts[0,:, idx_fr* self.Nfibo:(idx_fr+1)*self.Nfibo].reshape(-1,2)
            
            total_loss = self.dataconsitencyloss(out_kt[...,0],gt_kt[...,0]) \
                + self.dataconsitencyloss(out_kt[...,1],gt_kt[...,1])
   
            self.optimizer.zero_grad()
            total_loss.backward()   
            self.optimizer.step()
            if not opt.scheduler_off:
                self.scheduler.step()
            self.writer.add_scalar('total loss/training_loss', total_loss, step)          
             
            if step % opt.save_period == 0 and opt.save_period > 0 and step > 0:
                self.evaluate()

            step += 1
            self.step = step
        
        self.writer.close()   
        
    @torch.no_grad()
    def evaluate(self):
        step = self.step
        max_steps = self.opt.max_steps

        # For visualizations
        # Get Average PSNR and SSIM values for entire frames
        psnr_val_list = []
        ssim_val_list = []
        ims = []
        out= []
        images = []

        with torch.no_grad():
            self.net.eval()
            print("...start...eval...")
            for idx_fr in range(self.Nfr):
                if self.opt.temporal_interpolation:
                    diff = int(self.opt.down_ratio)
                    if (idx_fr - 1) % diff == 0:
                        if (idx_fr - diff) < 0:
                            idx_fr_4_time_v1 = idx_fr+diff
                            idx_fr_4_time_v2 = idx_fr+diff
                        elif idx_fr + diff >= self.Nfr:
                            idx_fr_4_time_v1 = idx_fr-diff
                            idx_fr_4_time_v2 = idx_fr-diff
                        else:
                            idx_fr_4_time_v1 = idx_fr-diff
                            idx_fr_4_time_v2 = idx_fr+diff
                    else:
                        if idx_fr == 0 or idx_fr == self.Nfr-1:
                            idx_fr_4_time_v1 = idx_fr-1
                            idx_fr_4_time_v2 = idx_fr-1
                        else:
                            for cntt in range(diff-1):
                                if (idx_fr-cntt-1)%diff == 0:
                                    idx_fr_4_time_v1 = idx_fr-cntt
                                if (idx_fr+cntt-1)%diff == 0:
                                    idx_fr_4_time_v2 = idx_fr+cntt                              
                           
                
                if self.opt.spatial_interpolation or not self.opt.temporal_intepolation:
                    diff = 1
                if idx_fr - diff < 0:
                    idx_fr_4_time_v1 = idx_fr+diff
                    idx_fr_4_time_v2 = idx_fr+diff
                elif idx_fr + diff >= self.Nfr:
                    idx_fr_4_time_v1 = idx_fr-diff
                    idx_fr_4_time_v2 = idx_fr-diff
                else:
                    idx_fr_4_time_v1 = idx_fr-diff
                    idx_fr_4_time_v2 = idx_fr+diff 

                idx_fr_4_time = idx_fr % self.Nfr
                frame_index = self.t[idx_fr_4_time]                    

                time_idx = torch.tensor(frame_index).unsqueeze(0).unsqueeze(1).to(self.dev)      

                self.model_freeze()

                features_v1 = self.feature_extract(int(idx_fr_4_time_v1))
                features_v2 = self.feature_extract(int(idx_fr_4_time_v2))
                features = torch.cat([features_v1,features_v2], dim=1)
                out_sp = self.net.test(self.coords_sr, self.coords_eval_spatial_int, time_idx, cell=None, features=features).reshape(self.gt_img_size, self.gt_img_size, -1)

                
                out_sp_ = (out_sp[...,0]**2+out_sp[...,1]**2)**0.5
                tmp_ims = out_sp_.detach().cpu().numpy()
                tmp_ims = self.minmaxnorm(tmp_ims)
                images.append(tmp_ims)
                gt_cartesian_img = self.gt_cartesian_img[:,:,idx_fr_4_time]
                
                psnr_val_list += [psnr(gt_cartesian_img, tmp_ims, data_range=1)]
                ssim_val_list += [ssim(gt_cartesian_img, tmp_ims, data_range=1)]

                images_grid = np.concatenate([tmp_ims[None], gt_cartesian_img[None]], axis=2)
                ims.append(images_grid)
                out.append(out_sp.detach().cpu().numpy())

        
        psnr_val = np.array(psnr_val_list).sum()/self.Nfr
        ssim_val = np.array(ssim_val_list).sum()/self.Nfr

        self.writer.add_scalar('metrics/psnr', psnr_val, step)
        self.writer.add_scalar('metrics/ssim', ssim_val, step)
        self.writer.add_image('recon_image', ims[0], step)

        self.t2 = time.time()
        
        is_save=False
        
        if (psnr_val >= self.best_psnr):
            self.best_psnr, self.best_psnr_step = psnr_val, step
            B_T = time.time()
            self.best_time = B_T - self.t1
            is_save=True

        if (ssim_val >= self.best_ssim):
            self.best_ssim, self.best_ssim_step = ssim_val, step
            B_T = time.time()
            self.best_time = B_T - self.t1
            is_save=True                
        
        if is_save:
            self.save(step)
            self.save_frames(images)
            is_save=False

        curr_lr = self.scheduler.get_lr()[0]
        eta = (self.t2-self.t1) * (max_steps-step) /self.opt.save_period / 3600
        print("[{}/{}] {:.2f} {:.4f} (Best PSNR: {:.2f} SSIM {:.4f} @ {} step) LR: {}, ETA: {:.1f} hours"
            .format(step, max_steps, psnr_val, ssim_val, self.best_psnr, self.best_ssim, self.best_psnr_step,
                curr_lr, eta))

    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['net_state_dict']) 
        self.net.to(self.dev)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler = checkpoint['scheduler']
        self.step = checkpoint['step']
        self.best_psnr, self.best_psnr_step = checkpoint['best_psnr'], checkpoint['best_psnr_step']
        self.best_ssim, self.best_ssim_step = checkpoint['best_ssim'], checkpoint['best_ssim_step']
        self.step = checkpoint['step']+1        
        
    def save(self, step):
        print('saving ... ')
        save_path = os.path.join(self.opt.ckpt_root, str(step)+".pt")
        ckptdict = {
                'step': step,
                'time': self.best_time,
                'net_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler': self.scheduler,
                }
        best_scores = {
                'best_psnr': self.best_psnr,
                'best_ssim': self.best_ssim,
                'best_psnr_step': self.best_psnr_step,
                'best_ssim_step': self.best_ssim_step,
                }
        ckptdict = {**ckptdict, **best_scores}
        
                
        torch.save(ckptdict, save_path)
        with open(os.path.join(self.opt.ckpt_root, 'best_scores.json'), 'w') as f:
            json.dump(best_scores, f)

    def save_frames(self, images):
        output_frames_path = os.path.join(self.opt.ckpt_root, f'output_frames_{self.step}')
        os.makedirs(output_frames_path, exist_ok=True)

        for idx_fr in range(self.Nfr):
            tmp_ims = images[idx_fr].squeeze()
            plt.imsave(os.path.join(output_frames_path,'{:04}.png'.format(idx_fr)), tmp_ims, cmap = 'gray', vmax=0.5)


    def minmaxnorm(self, x):
        return (x-x.min())/(x.max() + 1e-6)
    
    def prepare_gt(self):
        # For visualization: GT full sampled images
        data = sio.loadmat('cardiac32ch_b1.mat')
        seqq = data['data']
        ccoil = data['b1']
        gt_cartesian_img = np.zeros(seqq.shape[:3])
        for i in range(gt_cartesian_img.shape[-1]):
            gt_cartesian_img[...,i] = self.minmaxnorm(np.sqrt((np.abs(seqq[:,:,i]*ccoil)**2).mean(-1)))

        self.gt_cartesian_img = gt_cartesian_img
        self.Nfr = gt_cartesian_img.shape[-1]
        self.gt_img_size = gt_cartesian_img.shape[0]

    def prepare_dataset(self, sr_ratio):
        num_cycle = self.opt.num_cycle
        Nfibo = self.opt.Nfibo
        scale = self.opt.scaling_data

        data_raw_fname = 'syn_radial_data_cycle%s_Nfibo%s_scale%s_spatialdown%s.mat'%(num_cycle, Nfibo, scale, sr_ratio)
        syn_radial_img_fname = 'syn_radial_img_cycle%s_Nfibo%s_scale%s_spatialdown%s.mat'%(num_cycle, Nfibo, scale, sr_ratio)
              
        fname = self.opt.fname
        seq= np.squeeze(sio.loadmat(fname)['data'])*scale # numpy array (128, 128, 23, 32), complex128, kt-space data
        coil = sio.loadmat(fname)['b1'].astype(np.complex64) #  numpy array, coil sensitivity
        coil = np.transpose(coil,(2,0,1)) # (32, 128, 128)
        
        Nfr = np.shape(seq)[2]*num_cycle # 23 number of frames * num_cycle (13)
        Nc=np.shape(seq)[-1] # 32 number of coils
        Nvec=np.shape(seq)[0]*2 # 256 radial sampling number (virtual k-space)
        img_size=np.shape(seq)[0] # 128        
        
        tmp_imgsize = int(img_size/sr_ratio)
        if tmp_imgsize % 2 != 0:
            tmp_imgsize += 1
        offset = int((img_size-tmp_imgsize)//2)
        down_seq = np.zeros((tmp_imgsize, tmp_imgsize, Nfr, Nc)).astype(np.complex64)
        
        start_of_crop = offset
        end_of_crop = offset+tmp_imgsize
        for frame_num in range(Nfr):
            for coil_num in range(Nc):
                onecoil_seq = seq[:,:,frame_num,coil_num]
                fourier_onecoilseq = np.fft.fftshift(np.fft.fft2(np.fft.fftshift((onecoil_seq))))
                kcentercrop = fourier_onecoilseq[start_of_crop:end_of_crop,start_of_crop:end_of_crop]
                centercrop = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kcentercrop)))
                down_seq[:,:,frame_num,coil_num] = centercrop
        
        tmp_coil = np.zeros((Nc, tmp_imgsize, tmp_imgsize)).astype(np.complex64)
        for coil_num in range(Nc):
            onecoil = coil[coil_num]
            konecoil = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(onecoil)))
            kcrop = konecoil[start_of_crop:end_of_crop,start_of_crop:end_of_crop]
            crop = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kcrop)))
            tmp_coil[coil_num]=crop
        
        coil = tmp_coil
        Nvec = np.shape(down_seq)[0]*2
        img_size=np.shape(down_seq)[0]
        seq = down_seq

        gt_cartesian_kt = seq[...,np.newaxis].astype(np.complex64) 
        gt_cartesian_kt_ri = np.concatenate((np.real(gt_cartesian_kt),np.imag(gt_cartesian_kt)),axis=-1)  
        gt_cartesian_kt_ri = np.transpose(gt_cartesian_kt_ri,(3,2,0,1,4)) 
        gt_cartesian_kt_ri = np.concatenate([gt_cartesian_kt_ri]*num_cycle,axis = 1)
        gt_cartesian_kt = np.concatenate([gt_cartesian_kt]*num_cycle,axis=2) 

        w1=np.linspace(1,0,Nvec//2) 
        w2=np.linspace(0,1,Nvec//2) 
        w=np.concatenate((w1,w2),axis=0)[np.newaxis,np.newaxis] 
        wr=np.tile(w,(Nc,Nfibo,1)) 
        denc = wr.astype(np.complex64)

        # GT image sequence
        self.prepare_gt()

        # 111.246 degree - golden angle | 23.63 degree - tiny golden angle
        gA=111.246
        one_vec_y=np.linspace(-3.1293208599090576,3.1293208599090576,num=Nvec)[...,np.newaxis]
        one_vec_x=np.zeros((Nvec,1))
        one_vec=np.concatenate((one_vec_y,one_vec_x),axis=1) 

        Nang=Nfibo*Nfr
        set_ang=np.zeros((Nang*Nvec,2),np.double)
        for i in range(Nang):
            theta=gA*(i+1)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c,-s), (s, c)))
            for j in range(Nvec):
                tmp=np.matmul(R,one_vec[j,:])
                set_ang[i*Nvec+j,0]=tmp[0]
                set_ang[i*Nvec+j,1]=tmp[1]
        
        #### SAVE KT
        data_raw=np.zeros((Nc,Nfibo*Nfr,Nvec)).astype(np.complex64)
        # Generate down-sampled data 
        for idx_fr in range(Nfr): # Fourier transform per each frame
            print('%s/%s'%(idx_fr,Nfr), '\r', end='')
            angle=set_ang[idx_fr*Nfibo*Nvec:(idx_fr+1)*Nfibo*Nvec,:]
            mynufft_test = Mypnufft_cardiac_test(img_size,angle,Nfibo,Nvec,Nc,coil,denc)
            
            tmp=mynufft_test.forward(gt_cartesian_kt_ri[:,idx_fr,:,:,:])
            tmp_c=tmp[...,0]+1j*tmp[...,1]
            tmp_disp=tmp_c.reshape(Nc,Nfibo,Nvec)

            data_raw[:,idx_fr*Nfibo:(idx_fr+1)*Nfibo,:] = tmp_disp
        data_raw=np.transpose(data_raw,(2,1,0))
        if not os.path.isfile(data_raw_fname):
            sio.savemat(data_raw_fname,{'data_raw':data_raw, 'coil':coil, 'gt_cartesian_img': self.gt_cartesian_img, 'denc': denc, 'set_ang':set_ang})
            print('file saved: %s' % data_raw_fname)

        ### SAVE IMAGE   
        # Generate down-sampled image
        syn_radial_ri = np.concatenate((np.real(data_raw[...,np.newaxis]),np.imag(data_raw[...,np.newaxis])),axis=3)
        syn_radial_ri = np.transpose(syn_radial_ri,(2,1,0,3)) 
        syn_radial_ri_ts = np_to_torch(syn_radial_ri.astype(np.float32)).to(self.dev).detach()
        
        # Just for visualization: naive inverse Fourier of undersampled data
        syn_radial_img=np.zeros((img_size,img_size,Nfr))
        print('Get images of the synthetic radial (down-sampled) data')
        MAX_VALUE = 0
        for idx_fr in range(Nfr):
            print('%s/%s'%(idx_fr,Nfr), '\r', end='')
            angle=set_ang[idx_fr*Nfibo*Nvec:(idx_fr+1)*Nfibo*Nvec,:] 
            inp= torch_to_np(syn_radial_ri_ts[:,:,idx_fr*Nfibo:(idx_fr+1)*Nfibo,:,:]) 

            mynufft_test = Mypnufft_cardiac_test(img_size,angle,Nfibo,Nvec,Nc,coil,denc)
            gt_re_np=mynufft_test.backward(inp.reshape((-1,2))) 

            tmp = np.sqrt(gt_re_np[:,:,0]**2+gt_re_np[:,:,1]**2) 
            if tmp.max() > MAX_VALUE:
                MAX_VALUE = tmp.max()
            syn_radial_img[:,:,idx_fr] = self.minmaxnorm(tmp)

        if not os.path.isfile(syn_radial_img_fname):
            sio.savemat(syn_radial_img_fname,{'syn_radial_img':syn_radial_img})
            print('file saved: %s' % syn_radial_img_fname)
            

        self.mynufft = Mypnufft_cardiac(img_size, Nc)
        self.set_ang = set_ang
        self.img_size = img_size
        self.Nfibo = self.opt.Nfibo
        self.Nvec = Nvec
        self.coil = coil
        self.denc = denc
        self.syn_radial_img = syn_radial_img
        self.Nc = Nc
        self.syn_radial_ri_ts = syn_radial_ri_ts
        