import os
from glob import glob
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import legacy
import dnnlib

from PIL import Image
from torch.utils.data import Dataset
from model import Generator_feat
from util import bilinear_interp_sampling 

def img_np2pt(img):
    return torch.FloatTensor(np.array(img) / 255)

def sample_fpyr(G, latent):
    with torch.no_grad():
        fpyr, _ = G.synthesis(latent, noise_mode='const')

    return [f.cpu() for f in fpyr]

class StyleGANLightDataset(Dataset):
    def __init__(self, latent_path, mask_path=None):
        self.latent_path = latent_path
        self.mask_path = mask_path
        self.device = torch.device('cuda:0')

        self.res = 1024
        self.network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl"

        self.id_list, self.latent_dict = self.set_latent_dict(latent_path)
        self.fpyrs_dict = self.set_fpyrs_dict(self.latent_dict)

        self.mask_dict = self.load_mask_dict(mask_path)
        self.normal_idx_dict = self.get_valid_normal_index(self.mask_dict)

        self.num_id = len(self.id_list)
        print(f'length: {len(self.id_list)}')

    def load_mask_dict(self, mask_path):
        mask_dict = {}
        for id in self.id_list:
            mask_dict[id] = torch.load(f'{mask_path}/{id}_0.pt').detach().cpu()

        return mask_dict

    def set_latent_dict(self, latent_path):
        latent_dict = {}
        id_list = []

        flist = glob(os.path.join(latent_path, '*.pt'))
        for f in flist:
            fname = os.path.basename(f).split('.')[0]
            id = fname.split('_')[0]
            if not id in latent_dict.keys():
                latent_dict[id] = []
                id_list.append(id)

            if f.endswith('_0.pt') or f.endswith('_1.pt') or f.endswith('_2.pt') or f.endswith('_3.pt') or f.endswith('_4.pt'):
                latent_dict[id].append(f) 

        return id_list, latent_dict

    def set_fpyrs_dict(self, latent_dict):
        fpyrs = {}

        with dnnlib.util.open_url(self.network_pkl) as f:
            data = legacy.load_network_pkl(f)
            G = data['G_ema']

        G_feat = Generator_feat(G.z_dim, G.c_dim, G.w_dim, G.img_resolution, G.img_channels).to(self.device)
        G_feat.load_state_dict(G.state_dict())

        for id in self.id_list:
            fpyrs[id] = []
            for f in latent_dict[id]:
                latent = torch.load(f).to(self.device)
                fpyrs[id].append(sample_fpyr(G_feat, latent))

        return fpyrs
    
    def get_valid_normal_index(self, mask_dict):
        normal_idx_dict = {}

        for id in self.id_list:
            valid_norm = (mask_dict[id] != 0)
            print(f'valid_norm.shape: {valid_norm.shape}')
            normal_idx_dict[id] = valid_norm.nonzero()

        return normal_idx_dict

    def __len__(self):
        return len(self.id_list) * self.res * self.res

    def __getitem__(self, index):
        index = index % self.num_id
        id = self.id_list[index]
        bool_norm = self.normal_idx_dict[id]
        pos_idx = np.random.randint(bool_norm.shape[0])
        h, w = bool_norm[pos_idx]
        h = h.item()
        w = w.item()

        fvecs = []
        for fpyr in self.fpyrs_dict[id]:
            fvec = []
            for i, feat in enumerate(fpyr):
                res = 2 ** (2 + int(i/2))
                v = bilinear_interp_sampling(feat, h, w, self.res, res)
                fvec.append(v.squeeze(0))

            fvec = torch.cat(fvec, dim=0)
            fvecs.append(fvec)

        return index, fvecs[0], fvecs[1], fvecs[2], fvecs[3], fvecs[4]

