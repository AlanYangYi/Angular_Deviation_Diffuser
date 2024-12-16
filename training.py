import argparse
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from Angular_Deviation_Diffuser.model import FoldingDiff
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset
from pathlib import Path
import util
import random
import math
import numpy as np
import os

class FoldingDiffDataset(Dataset):
    def __init__(self, meta, data_dir, T, tensor_file, mu=None, s=8e-3, max_len=147):
        self.meta = meta
        self.records = meta.to_records()
        self.data_dir = Path(data_dir)
        self.T = T
        self.max_len = max_len


        if Path(tensor_file).exists():
            print(f"Loading data from {tensor_file}")
            self.data_cache = torch.load(tensor_file)
        else:
            self.data_cache = {}
            # 预先加载所有数据到内存中
            for r in self.records:
                self.data_cache[r.id] = torch.tensor(np.load(self.data_dir / f'{r.id}.npy')).float()
            # 保存数据到 tensor 文件
            torch.save(self.data_cache, tensor_file)
            print(f"Saved data to {tensor_file}")

        # Cosine variance schedule
        t = torch.arange(T + 1)
        f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2.0).square()
        self.alpha_bar = f_t / f_t[0]
        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha_bar_sqrt = torch.sqrt(1 - self.alpha_bar)

        if mu is None:
            feats = []
            for r in self.records:
                angles = self.data_cache[r.id]
                angles = np.nan_to_num(angles.numpy())
                feats.append(angles)
            feats = np.concatenate(feats, axis=0)
            self.mu = torch.tensor(feats.mean(axis=0)).float()
        else:
            self.mu = torch.tensor(mu).float()

    def get_mu(self):
        return self.mu

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        r = self.records[idx]
        x0 = self.data_cache[r.id]  # 从缓存中读取数据
        loss_mask = torch.isfinite(x0).float()

        x0.nan_to_num_(0.0)
        x0 = util.wrap(x0 - self.mu)

        n_residues = len(x0)
        if n_residues < self.max_len:
            x0 = torch.cat([x0, torch.zeros([self.max_len - n_residues, 6])], axis=0)
            loss_mask = torch.cat([loss_mask, torch.zeros([self.max_len - n_residues, 6])], axis=0)
        elif n_residues > self.max_len:
            start_idx = random.randint(0, n_residues - self.max_len)
            x0 = x0[start_idx: start_idx + self.max_len]
            loss_mask = loss_mask[start_idx: start_idx + self.max_len]

        t = torch.randint(0, self.T, (1,)).long()
        eps = torch.randn(x0.shape)
        x = x0 * self.alpha_bar_sqrt[t] + eps * self.one_minus_alpha_bar_sqrt[t]
        x = util.wrap(x)

        return {"x": x, "t": t, "eps": eps, "loss_mask": loss_mask}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['data_dir'] = str(state['data_dir'])
        return state

    def __setstate__(self, state):
        state['data_dir'] = Path(state['data_dir'])
        self.__dict__.update(state)


def get_meta_data(dir_='npys'):
    if not os.path.exists(dir_):

        raise FileNotFoundError(f"Error: Directory {dir_} does not exist for the dataset.")

    files = os.listdir(dir_)
    print(len(files))

    meta = pd.DataFrame({
        'id': [f.split('.')[0] for f in files]
    })
    meta['num_residues'] = [len(np.load(f'{dir_}/{id}.npy')) for id in meta['id'].values]
    meta.to_csv('meta.csv', index=False)






def parse_arguments(args_list):
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="meta.csv")
    parser.add_argument("--data_dir", type=str, default="npys")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=1000)
    if args_list:
        return parser.parse_args(args_list)
    else:
        return parser.parse_args()




def training(args_list:list):


    args = parse_arguments(args_list)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('meta.csv'):
        get_meta_data(dir_=args.data_dir)


    meta = pd.read_csv(args.meta).sample(frac=1.0, random_state=42)
    N = len(meta)
    train_meta, val_meta = meta.iloc[: int(0.9 * N)], meta.iloc[int(0.9 * N):]
    train_set = FoldingDiffDataset(meta=train_meta, data_dir=args.data_dir, T=args.timesteps,tensor_file='train_set_data.pt')
    mu = train_set.get_mu()
    print(f'mu is :{mu}')
    val_set = FoldingDiffDataset(meta=val_meta, data_dir=args.data_dir, T=args.timesteps, mu=mu,tensor_file='val_set_data.pt')

    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )



    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='foldingdiff-{epoch:04d}-{val_loss:.4f}',
        save_top_k=-1,  # Save all models
        every_n_epochs=250,  # Save every 500 epochs
        save_weights_only=True
    )



    trainer = pl.Trainer(
        accelerator="auto",
        devices='auto',
        max_epochs=5000,
        callbacks=[checkpoint_callback]
    )
    model = FoldingDiff()
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
    )
if __name__ == "__main__":
    training(None)
