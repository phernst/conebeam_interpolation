from argparse import ArgumentParser
from typing import List, Any
import json
import os
from os.path import join as pjoin

import cv2
import numpy as np
from piq import multi_scale_ssim
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from concatdataset import ConcatDataset
from metrics import rmse
from scatternet import ScatterNet
from utils import NORMALIZATION, load_h5


class Normalize:
    def __call__(self, sample):
        slist = [sample['in'], sample['out']]
        proj_norm = NORMALIZATION['projections_99']
        slist[0] = slist[0]/proj_norm
        slist[1] = slist[1]/proj_norm
        return {
            'in': slist[0],
            'out': slist[1],
        }


class ToTensor(object):
    def __call__(self, sample):
        slist = [sample['in'], sample['out']]
        slist[1] = slist[1][None, ...]
        slist = [torch.from_numpy(elem).float() for elem in slist]
        return {
            'in': slist[0],
            'out': slist[1],
        }


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, filename, neighbor_diff, num_neighbors=2, transform=None):
        self.projections = load_h5(pjoin(data_dir, filename))
        self.transform = transform
        self.neighbor_diff = neighbor_diff
        assert num_neighbors > 1 and num_neighbors % 2 == 0
        self.num_neighbors = num_neighbors

    def __len__(self):
        return self.projections.shape[0]

    def __getitem__(self, idx):
        projection = self.projections[idx]
        neighbor_diff = self.neighbor_diff
        half_num_neighbors = self.num_neighbors//2
        projection_plus = [self.projections[(idx + (k + 1)*neighbor_diff) % len(self)][:] for k in range(half_num_neighbors)]
        projection_minus = [self.projections[(idx - (k + 1)*neighbor_diff) % len(self)][:] for k in range(half_num_neighbors)][::-1]

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
        self.valid_dir = self.hparams.valid_dir
        self.network = ScatterNet(
            self.hparams.num_neighbors,
            [8, 16, 32, 64, 128, 256],
            activation=torch.nn.LeakyReLU,
        )

        def perc_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            clamp_x = torch.clamp(x, min=0, max=1)
            clamp_y = torch.clamp(y, min=0, max=1)
            return F.mse_loss(x, y) + 1. - multi_scale_ssim(clamp_x, clamp_y)
        self.loss = F.l1_loss
        self.accuracy = rmse
        self.trafo = transforms.Compose([Normalize(), ToTensor()])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            epochs=self.hparams.max_epochs,
            steps_per_epoch=len(self.train_dataloader()),
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
            'monitor': 'val_loss',
        }

    def forward(self, *args, **kwargs):
        net_in = args[0]
        return self.network(net_in)

    def training_step(self, *args, **kwargs):
        batch = args[0]
        neighbors, inter_proj = batch['in'], batch['out']
        prediction = self(neighbors)
        loss = self.loss(prediction, inter_proj)
        return {'loss': loss}

    def validation_step(self, *args, **kwargs):
        batch, batch_idx = args[0], args[1]
        neighbors, inter_proj = batch['in'], batch['out']
        prediction = self(neighbors)
        loss = self.loss(prediction, inter_proj)
        accuracy = self.accuracy(prediction, inter_proj)

        if batch_idx < 20:
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

        return {'val_loss': loss, 'val_acc': accuracy}

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

        return DataLoader(
            full_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        full_dataset = ConcatDataset(*[
            self.create_dataset(f)
            for f in self.hparams.valid_files
        ])

        return DataLoader(
            full_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
        )

    def predict_dataloader(self):
        ...

    def test_dataloader(self):
        ...

    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('training', avg_loss)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str)
        parser.add_argument('--valid_dir', type=str)
        parser.add_argument('--num_neighbors', type=int)
        parser.add_argument('--neighbor_diff', type=int)
        parser.add_argument('--train_files', type=list)
        parser.add_argument('--valid_files', type=list)
        return parser


def main():
    parser = ArgumentParser()
    parser = TrajectoryUpsampling.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.use_amp = True
    hparams.lr = 3e-3
    hparams.max_epochs = 150
    hparams.batch_size = 32
    hparams.neighbor_diff = 4
    hparams.num_neighbors = 2
    hparams.valid_dir = f'valid/nd{hparams.neighbor_diff}_nn{hparams.num_neighbors}_l1'
    hparams.data_dir = 'projections'
    with open('train_valid.json', 'r', encoding='utf-8') as json_file:
        json_dict = json.load(json_file)
        hparams.train_files = json_dict['train_files']
        hparams.valid_files = json_dict['valid_files']

    model = TrajectoryUpsampling(**vars(hparams))

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.valid_dir,
        monitor='val_loss',
        save_last=True,
    )
    lr_callback = LearningRateMonitor()

    trainer = Trainer(
        precision=16 if hparams.use_amp else 32,
        gpus=1,
        callbacks=[checkpoint_callback, lr_callback],
        max_epochs=hparams.max_epochs,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
