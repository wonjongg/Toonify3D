import os
import gc
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

import legacy
import dnnlib
from model import Generator_feat 
from util import bilinear_interp_sampling 


def img_np2pt(img):
    return torch.FloatTensor(np.array(img) / 255)

def sample_fpyr(G, latent):
    with torch.no_grad():
        fpyr, _ = G.synthesis(latent, noise_mode='const')

    return [m.cpu() for m in fpyr]

def id_replace(s):
    s = s.replace('env1', 'env0')
    s = s.replace('env2', 'env0')
    s = s.replace('env3', 'env0')
    s = s.replace('env4', 'env0')
    s = s.replace('env5', 'env0')

    return s

class StyleGANNormalDataset(Dataset):
    def __init__(self, latent_path, normal_path, weight_path, weight_name, mask_path=None):
        self.latent_path = latent_path
        self.normal_path = normal_path
        self.weight_path = weight_path
        self.weight_name = weight_name
        self.mask_path = mask_path
        self.device = torch.device('cuda:0')

        self.res = 1024
        self.network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl"

        self.id_list = sorted(os.listdir(latent_path))
        self.id_list = [e for e in self.id_list]

        self.latent_dict = self.set_latent_dict(latent_path)
        self.normal_dict = self.set_normal_dict(normal_path)
        self.weight_dict = self.set_weight_dict(weight_path)
        self.fpyrs_dict = self.set_fpyrs_dict(self.latent_dict, self.weight_dict)

        self.mask_dict = self.load_mask_dict(mask_path)

        self.normal_idx_dict = self.get_valid_normal_index(self.normal_dict, self.mask_dict)

        self.num_id = len(self.id_list)
        print(f'length: {self.num_id}')

        gc.collect()
        torch.cuda.empty_cache()

    def load_mask_dict(self, mask_path):
        mask_dict = {}
        for id in self.id_list:
            id_base = id_replace(id)
            mask_dict[id] = torch.load(f'{mask_path}/{id_base}_mask.pt').detach().cpu()

        return mask_dict

    def set_latent_dict(self, latent_path):
        latent_dict = {}
        for id in self.id_list:
            latent_dict[id] = os.path.join(os.path.join(latent_path, id), '0.pt')

        return latent_dict

    def set_normal_dict(self, normal_path):
        normal_dict = {}
        for id in self.id_list:
            id_base = id_replace(id)
            id_split = id_base.split('_')
            id_base = id_split[0] + '_' + id_split[1]
            if id.endswith('f'):
                path = os.path.join(normal_path, f'{id_base}_normal_f.png')
            else:
                path = os.path.join(normal_path, f'{id_base}_normal.png')

            normal_dict[id] = img_np2pt(Image.open(path).convert('RGB'))

        return normal_dict

    def set_weight_dict(self, weight_path):
        weight_dict = {}
        for id in self.id_list:
            weight_dict[id] = os.path.join(weight_path, f'{self.weight_name}_{id}.pt')

        return weight_dict

    def set_fpyrs_dict(self, latent_dict, weight_dict):
        fpyrs = {}

        with dnnlib.util.open_url(self.network_pkl) as f:
            data = legacy.load_network_pkl(f)
            G = data['G_ema']

        G_feat = Generator_feat(G.z_dim, G.c_dim, G.w_dim, G.img_resolution, G.img_channels).to(self.device)

        for id in self.id_list:
            G_feat.load_state_dict(torch.load(weight_dict[id]).state_dict())
            latent = torch.load(latent_dict[id]).to(self.device)
            fpyrs[id] = sample_fpyr(G_feat, latent)
            print(id)

        return fpyrs
    
    def get_valid_normal_index(self, normal_dict, mask_dict):
        normal_idx_dict = {}

        for id in self.id_list:
            bool_norm = normal_dict[id] != 0
            bool_norm = bool_norm[:, :, 0] | bool_norm[:, :, 1] | bool_norm[:, :, 2]
            valid_norm = bool_norm & (mask_dict[id] != 0)

            normal_idx_dict[id] = valid_norm.nonzero()

        return normal_idx_dict

    def __len__(self):
        return len(self.id_list) * self.res * self.res

    def __getitem__(self, index):
        index = index % self.num_id
        id = self.id_list[index]
        normal = self.normal_dict[id]
        bool_norm = self.normal_idx_dict[id]
        pos_idx = np.random.randint(bool_norm.shape[0])
        h, w = bool_norm[pos_idx]
        h = h.item()
        w = w.item()

        fpyr = self.fpyrs_dict[id]

        fvec = []
        
        for i, feat in enumerate(fpyr):
            res = 2 ** (2 + int(i/2))
            v = bilinear_interp_sampling(feat, h, w, self.res, res)
            fvec.append(v.squeeze(0))

        fvec = torch.cat(fvec, dim=0)
        normal = normal[h, w]

        return index, fvec, normal 
