from argparse import ArgumentParser
from typing import List, Any
import json
import os
from os.path import join as pjoin

import cv2
import nibabel as nib
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_msssim import MSSSIM
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from concatdataset import ConcatDataset
from unet import UNet
from utils import CONFIGURATION, DATA_DIRS, NORMALIZATION


class Normalize:
    def __call__(_, sample):
        slist = [sample['in'], sample['out']]
        proj_norm = NORMALIZATION['projections_99']
        slist[0] = slist[0]/proj_norm
        slist[1] = slist[1]/proj_norm
        return {
            'in': slist[0],
            'out': slist[1],
        }


class ToTensor(object):
    def __call__(_, sample):
        slist = [sample['in'], sample['out']]
        slist[1] = slist[1][None, ...]
        for i in range(len(slist)):
            slist[i] = torch.from_numpy(slist[i]).float()
        return {
            'in': slist[0],
            'out': slist[1],
        }


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, filename, neighbor_diff, num_neighbors=2, transform=None):
        self.projections = nib.load(pjoin(data_dir, filename)).get_fdata().transpose(1, 0, 2)
        self.transform = transform
        self.neighbor_diff = neighbor_diff
        assert num_neighbors > 1 and num_neighbors % 2 == 0
        self.num_neighbors = num_neighbors

    def __len__(self):
        return self.projections.shape[-1]

    def __getitem__(self, idx):
        projection = self.projections[..., idx]
        neighbor_diff = self.neighbor_diff
        half_num_neighbors = self.num_neighbors//2
        projection_plus = [self.projections[..., (idx + (k + 1)*neighbor_diff) % len(self)] for k in range(half_num_neighbors)]
        projection_minus = [self.projections[..., (idx - (k + 1)*neighbor_diff) % len(self)] for k in range(half_num_neighbors)][::-1]

        sample = {
            'in': np.concatenate([projection_minus, projection_plus], axis=0),
            'out': projection,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class TrajectoryUpsampling(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.current_lr = self.hparams.lr
        self.valid_dir = self.hparams.valid_dir
        self.network = UNet(self.hparams.num_neighbors, depth=5)
        if self.hparams.pretrained:
            loss_l1 = torch.nn.L1Loss()
            loss_msssim = MSSSIM()
            alpha = 0.5
            self.loss = lambda x, y: alpha*loss_l1(x, y) + \
                (1-alpha)*(1-loss_msssim(x, y))
        else:
            self.loss = torch.nn.MSELoss()
        self.trafo = transforms.Compose([Normalize(), ToTensor()])
        self.example_input_array = torch.empty(1, self.hparams.num_neighbors, 256, 256)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.8,
            min_lr=self.hparams.end_lr,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, _):
        neighbors, inter_proj = batch['in'], batch['out']
        prediction = self(neighbors)
        loss = self.loss(prediction, inter_proj)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        neighbors, inter_proj = batch['in'], batch['out']
        prediction = self(neighbors)
        loss = self.loss(prediction, inter_proj)

        if self.current_epoch % 5 == 0 and batch_idx < 100:
            os.makedirs(
                pjoin(self.valid_dir, f'{self.current_epoch}'),
                exist_ok=True,
            )
            inter_proj = inter_proj.cpu().numpy()[0, 0]
            cv2.imwrite(
                pjoin(self.valid_dir, f'{self.current_epoch}/{batch_idx}_out_gt.png'),
                inter_proj/inter_proj.max()*255)

            prediction = prediction.cpu().float().numpy()[0, 0]
            cv2.imwrite(
                pjoin(self.valid_dir,
                      f'{self.current_epoch}/{batch_idx}_out_pred.png'),
                prediction/inter_proj.max()*255,
            )

        return {'val_loss': loss}

    def create_dataset(self, filename: str) -> MyDataset:
        return MyDataset(
            self.hparams.data_dir,
            filename,
            self.hparams.neighbor_diff,
            self.hparams.num_neighbors,
            transform=self.trafo,
        )

    def train_dataloader(self) -> DataLoader:
        full_dataset = ConcatDataset(*[
            self.create_dataset(f)
            for f in self.hparams.train_files
        ])

        return DataLoader(full_dataset,
                          shuffle=True,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        full_dataset = ConcatDataset(*[
            self.create_dataset(f)
            for f in self.hparams.valid_files
        ])

        return DataLoader(full_dataset,
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('training', avg_loss)
        self.log('lr', self.current_lr)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        self.current_lr = optimizer.param_groups[0]['lr']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str)
        parser.add_argument('--valid_dir', type=str)
        parser.add_argument('--num_neighbors', type=int)
        parser.add_argument('--neighbor_diff', type=int)
        parser.add_argument('--pretrained', type=bool, default=False)
        parser.add_argument('--end_lr', type=float, default=1e-6)
        parser.add_argument('--train_files', type=list)
        parser.add_argument('--valid_files', type=list)
        return parser


def main():
    parser = ArgumentParser()
    parser = TrajectoryUpsampling.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.lr = 1e-2
    hparams.end_lr = 1e-6
    hparams.max_epochs = 300
    hparams.batch_size = CONFIGURATION['batch_size']
    hparams.neighbor_diff = 1
    hparams.num_neighbors = 8
    hparams.pretrained = False
    hparams.valid_dir = f'valid_nd{hparams.neighbor_diff}_nn{hparams.num_neighbors}' + ('_pre' if hparams.pretrained else '_mse')
    hparams.data_dir = DATA_DIRS['projections']
    with open('train_valid.json') as json_file:
        json_dict = json.load(json_file)
        hparams.train_files = json_dict['train_files']
        hparams.valid_files = json_dict['valid_files']

    if hparams.pretrained:
        # initialize weights from pretrained model
        pre_dir = f'valid_nd{hparams.neighbor_diff}_nn{hparams.num_neighbors}_mse'
        pre_path = [
            x for x in sorted(os.listdir(pre_dir))
            if x.endswith('.ckpt') and x.startswith('epoch')
        ][-1]
        model = TrajectoryUpsampling.load_from_checkpoint(
            pjoin(pre_dir, pre_path),
            **vars(hparams),
        )
        model.valid_dir = hparams.valid_dir
    else:
        model = TrajectoryUpsampling(**vars(hparams))

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.valid_dir,
        monitor='val_loss',
        save_last=True,
    )
    trainer = Trainer(
        precision=CONFIGURATION['precision'],
        progress_bar_refresh_rate=CONFIGURATION['progress_bar_refresh_rate'],
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        max_epochs=hparams.max_epochs,
        terminate_on_nan=True,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
