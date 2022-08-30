import json
import os
from os.path import join as pjoin

import nibabel as nib
import numpy as np
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training_reco_interpolation import TrajectoryUpsampling


def main(run_name: str):
    with open('train_valid.json', 'r', encoding='utf-8') as json_file:
        test_subjects = json.load(json_file)['test_files']

    checkpoint_dir = pjoin('valid', run_name)
    checkpoint_path = sorted([x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    out_dir = pjoin('test', run_name)
    os.makedirs(out_dir, exist_ok=True)

    visual_dir = pjoin('visual', run_name)
    os.makedirs(visual_dir, exist_ok=True)

    model = TrajectoryUpsampling.load_from_checkpoint(
        pjoin(checkpoint_dir, checkpoint_path))
    model.eval()
    model.cuda()

    test_dataset = model.create_dataset(test_subjects[0])

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=model.hparams.batch_size,
        pin_memory=True,
    )

    with torch.inference_mode():
        full_predictions = torch.cat(
            [model(batch['in'].cuda()) for batch in tqdm(dataloader_test)])

        full_gt = torch.cat(
            [batch['out'][:, 3].cuda() for batch in tqdm(dataloader_test)])

        print(f'{full_predictions.shape=}')
        print(f'{full_gt.shape=}')

        full_predictions[:, 0] = full_predictions[:, 0].roll(-3)
        full_predictions[:, 1] = full_predictions[:, 1].roll(-2)
        full_predictions[:, 2] = full_predictions[:, 2].roll(-1)
        full_predictions[:, 3] = full_predictions[:, 3].roll(0)
        full_predictions[:, 4] = full_predictions[:, 4].roll(1)
        full_predictions[:, 5] = full_predictions[:, 5].roll(2)
        full_predictions[:, 6] = full_predictions[:, 6].roll(3)

        img = nib.Nifti1Image(full_predictions[0].cpu().numpy().transpose(), np.eye(4))
        nib.save(img, 'pred.nii.gz')
        img = nib.Nifti1Image(full_gt[0].cpu().numpy().transpose(), np.eye(4))
        nib.save(img, 'gt.nii.gz')

    # df = pd.DataFrame.from_dict(metrics)
    # df.to_csv(pjoin(out_dir, f"Results{f'_mis{misalign}' if misalign else ''}{f'_noise{photon_flux}' if photon_flux is not None else ''}.csv"))


if __name__ == '__main__':
    seed_everything(42)
    main('inter_nd4_nn2_out7_reco')
