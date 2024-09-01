import os 
import gc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import StyleGANNormalDataset
from dataset_light import StyleGANLightDataset
from torch.utils import data
from model import StyleNormal 

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)
    
def main(args):
    device = 'cuda:0'

    dataset_env = StyleGANLightDataset(args.env_latent_path, args.env_mask_path)

    dataset = StyleGANNormalDataset(args.latent_path, args.normal_path, args.weight_path, args.weight_name, args.mask_path)

    train_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(dataset, shuffle=True),
        drop_last=True,
    )
    
    train_loader_env = data.DataLoader(
        dataset_env,
        batch_size=args.batch_size,
        sampler=data_sampler(dataset_env, shuffle=True),
        drop_last=True,
    )
    
    os.makedirs(args.model_path, exist_ok=True)

    classifier = StyleNormal(numpy_class=args.num_class, dim=args.dim).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    classifier.train()

    iteration = 0

    for epoch in range(100):
        for data1, data2 in zip(train_loader, train_loader_env):
            fvecs, normals = data1[1], data1[2]
            fvecs_e0, fvecs_e1, fvecs_e2, fvecs_e3, fvecs_e4 = data2[1], data2[2], data2[3], data2[4], data2[5]

            fvecs, normals = fvecs.to(device), normals.to(device)
            fvecs_e0, fvecs_e1, fvecs_e2, fvecs_e3, fvecs_e4 = fvecs_e0.to(device), fvecs_e1.to(device), fvecs_e2.to(device), fvecs_e3.to(device), fvecs_e4.to(device)

            optimizer.zero_grad()
            normals_pred = 2 * classifier(fvecs) - 1
            normals_pred = normals_pred / normals_pred.norm(p=2, dim=1).unsqueeze(1)
            normals_pred = (normals_pred + 1) / 2

            normals_pred_e0 = 2 * classifier(fvecs_e0) - 1
            normals_pred_e1 = 2 * classifier(fvecs_e1) - 1
            normals_pred_e2 = 2 * classifier(fvecs_e2) - 1
            normals_pred_e3 = 2 * classifier(fvecs_e3) - 1
            normals_pred_e4 = 2 * classifier(fvecs_e4) - 1

            normals_pred_e0 = normals_pred_e0 / normals_pred_e0.norm(p=2, dim=1).unsqueeze(1)
            normals_pred_e1 = normals_pred_e1 / normals_pred_e1.norm(p=2, dim=1).unsqueeze(1)
            normals_pred_e2 = normals_pred_e2 / normals_pred_e2.norm(p=2, dim=1).unsqueeze(1)
            normals_pred_e3 = normals_pred_e3 / normals_pred_e3.norm(p=2, dim=1).unsqueeze(1)
            normals_pred_e4 = normals_pred_e4 / normals_pred_e4.norm(p=2, dim=1).unsqueeze(1)

            normals_pred_mean = ((normals_pred_e0 +\
                                normals_pred_e1 +\
                                normals_pred_e2 +\
                                normals_pred_e3 +\
                                normals_pred_e4) / 5).detach()

            data_loss = criterion(normals_pred, normals)

            reg_loss = criterion(normals_pred_e0, normals_pred_mean) +\
                        criterion(normals_pred_e1, normals_pred_mean) +\
                        criterion(normals_pred_e2, normals_pred_mean) +\
                        criterion(normals_pred_e3, normals_pred_mean) +\
                        criterion(normals_pred_e4, normals_pred_mean)

            loss = data_loss + 1e-3 * reg_loss
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 10 == 0:
                print(f"Epoch: {epoch}, iteration: {iteration}, loss: {loss.item():.5f}, data_loss: {data_loss.item():.5f}, reg_loss: {reg_loss.item():.5f}")
                gc.collect()

            if iteration % 1000 == 0:
                model_path = os.path.join(args.model_path, f'model_iter{iteration:08}.pth')
                print(f"Save checkpoint: {model_path}")
                torch.save(classifier.state_dict(), model_path)


    gc.collect()
    model_path = os.path.join(args.model_path, f'model_final.pth')
    torch.save(classifier.state_dict(), model_path)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    your_PTI_path = ''
    exp_name = ''
    weight_name_from_PTI = ''
    save_path = ''

    parser = argparse.ArgumentParser()

    parser.add_argument('--latent_path', type=str, default=f'{your_PTI_path}/PTI/embeddings/{exp_name}/PTI')
    parser.add_argument('--normal_path', type=str, defaPTIult=f'{your_PTI_path}/PTI/inputs/{exp_name}/aligned_normals')
    parser.add_argument('--weight_path', type=str, default=f'{your_PTI_path}/PTI/checkpoints')
    parser.add_argument('--weight_name', type=str, default=f'{weight_name_from_PTI}')
    parser.add_argument('--mask_path', type=str, default=f'{your_PTI_path}/PTI/inputs/{exp_name}/aligned_valid')
    parser.add_argument('--model_path', type=str, default=f'{save_path}')
    parser.add_argument('--env_latent_path', type=str, default='./styleflow_results')
    parser.add_argument('--env_mask_path', type=str, default='./styleflow_results/mask')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('--dim', type=int, default=6080)
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    main(args)
