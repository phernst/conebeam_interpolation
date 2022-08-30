from argparse import ArgumentParser
from typing import List, Any
import json
import os
from os.path import join as pjoin

import cv2
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Trainer
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from concatdataset import ConcatDataset
from metrics import rmse
from reco_loss import RecoLoss
from unet import UNet
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


class ToTensor:
    def __call__(self, sample):
        slist = [sample['in'], sample['out']]
        slist = [torch.from_numpy(elem).float() for elem in slist]
        return {
            'in': slist[0],
            'out': slist[1],
        }


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, interpolations_dir, filename, neighbor_diff,
                 num_outputs, num_neighbors=2, transform=None):
        assert num_neighbors in (1, 2, 4)
        assert self._is_num_outputs_valid(neighbor_diff, num_outputs)
        self.num_outputs = num_outputs
        self.projections = load_h5(pjoin(data_dir, filename))
        self.interpolations = load_h5(pjoin(interpolations_dir, filename))
        self.neighbor_diff = neighbor_diff
        self.transform = transform
        self.num_neighbors = num_neighbors

    def __len__(self):
        return self.projections.shape[0]

    @staticmethod
    def _is_num_outputs_valid(neighbor_diff: int, num_outputs: int) -> bool:
        return (neighbor_diff == 1 and num_outputs == 1) or \
            (neighbor_diff == 2 and num_outputs in (1, 3)) or \
            (neighbor_diff == 4 and num_outputs in (1, 3, 7))

    def __getitem__(self, idx):
        interpolation = self.interpolations[idx][None]
        neighbor_diff = self.neighbor_diff

        half_num_neighbors = self.num_neighbors//2
        projection_plus = [self.projections[(idx + (k + 1)*neighbor_diff) % len(self)][:] for k in range(half_num_neighbors)]
        projection_minus = [self.projections[(idx - (k + 1)*neighbor_diff) % len(self)][:] for k in range(half_num_neighbors)][::-1]

        projections_out = [self.projections[idx]] if self.num_outputs == 1 else \
            [
                self.projections[(idx - neighbor_diff//2) % len(self)],
                self.projections[idx],
                self.projections[(idx + neighbor_diff//2) % len(self)],
            ] if self.num_outputs == 3 else \
            [
                self.projections[(idx - 3*neighbor_diff//4) % len(self)],
                self.projections[(idx - 2*neighbor_diff//4) % len(self)],
                self.projections[(idx - 1*neighbor_diff//4) % len(self)],
                self.projections[idx],
                self.projections[(idx + 1*neighbor_diff//4) % len(self)],
                self.projections[(idx + 2*neighbor_diff//4) % len(self)],
                self.projections[(idx + 3*neighbor_diff//4) % len(self)],
            ]

        sample = {
            'in': np.concatenate([projection_minus, interpolation, projection_plus], axis=0),
            'out': np.stack(projections_out, axis=0),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class TrajectoryUpsampling(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.name = f'inter_nd{self.hparams.neighbor_diff}' \
            f'_nn{self.hparams.num_neighbors}' \
            f'_out{self.hparams.num_outputs}_reco'
        self.valid_dir = self.hparams.valid_dir
        self.network = UNet(
            self.hparams.num_neighbors + 1,
            n_classes=self.hparams.num_outputs,
        )

        self.loss = RecoLoss()
        self.accuracy = rmse
        self.trafo = transforms.Compose([Normalize(), ToTensor()])
        self.example_input_array = torch.zeros(1, self.hparams.num_neighbors + 1, 256, 256)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.hparams.lr,
            epochs=self.hparams.max_epochs,
            steps_per_epoch=len(self.train_dataloader())//2,
        )
        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": 'step',
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
        in_proj, inter_proj = batch['in'], batch['out']
        prediction = self(in_proj)
        loss = self.loss(prediction, inter_proj)
        accuracy = self.accuracy(prediction, inter_proj)

        if batch_idx < 20:
            os.makedirs(
                pjoin(self.valid_dir, self.name, f'{self.current_epoch}'),
                exist_ok=True,
            )

            inter_proj = inter_proj.cpu().numpy()[0, 0]
            cv2.imwrite(
                pjoin(self.valid_dir, self.name,
                      f'{self.current_epoch}', f'{batch_idx}_out_gt.png'),
                inter_proj/inter_proj.max()*255)

            prediction = prediction.cpu().float().numpy()[0, 0]
            cv2.imwrite(
                pjoin(self.valid_dir, self.name,
                      f'{self.current_epoch}', f'{batch_idx}_out_pred.png'),
                prediction/inter_proj.max()*255,
            )

        return {'val_loss': loss, 'val_acc': accuracy}

    def create_dataset(self, filename: str) -> MyDataset:
        return MyDataset(
            self.hparams.data_dir,
            f'interpolations_nd{self.hparams.neighbor_diff}',
            filename,
            self.hparams.neighbor_diff,
            transform=self.trafo,
            num_outputs=self.hparams.num_outputs,
            num_neighbors=self.hparams.num_neighbors,
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
        parser.add_argument('--neighbor_diff', type=int)
        parser.add_argument('--num_neighbors', type=int)
        parser.add_argument('--num_outputs', type=int)
        parser.add_argument('--train_files', type=list)
        parser.add_argument('--valid_files', type=list)
        return parser


def main():
    torch.set_num_threads(2)
    seed_everything(seed=42)

    parser = ArgumentParser()
    parser = TrajectoryUpsampling.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.use_amp = True
    hparams.lr = 2e-2
    hparams.max_epochs = 150
    hparams.batch_size = 16
    hparams.neighbor_diff = 4
    hparams.num_neighbors = 2
    hparams.num_outputs = 7
    hparams.valid_dir = 'valid'
    hparams.data_dir = 'projections'
    with open('train_valid.json', 'r', encoding='utf-8') as json_file:
        json_dict = json.load(json_file)
        hparams.train_files = json_dict['train_files']
        hparams.valid_files = json_dict['valid_files']

    model = TrajectoryUpsampling(**vars(hparams))

    checkpoint_callback = ModelCheckpoint(
        dirpath=pjoin(hparams.valid_dir, model.name),
        monitor='val_loss',
        save_last=True,
    )
    lr_callback = LearningRateMonitor()
    logger = TensorBoardLogger('lightning_logs', name=model.name)

    trainer = Trainer(
        logger=logger,
        precision=16 if hparams.use_amp else 32,
        gpus=1,
        callbacks=[checkpoint_callback, lr_callback],
        max_epochs=hparams.max_epochs,
        accumulate_grad_batches=2,
    )
    trainer.fit(model)
    # lr_finder = trainer.tuner.lr_find(model)

    # # # Results can be found in
    # with open('lr_finder.json', 'w', encoding='utf-8') as lrfile:
    #     json.dump(lr_finder.results, lrfile)

    # # Plot with
    # fig = lr_finder.plot(suggest=True, show=True)
    # # fig.show()

    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print(new_lr)


if __name__ == '__main__':
    main()
