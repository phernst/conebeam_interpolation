import os
from os.path import join as pjoin

import h5py
import nibabel as nib
import torch
from torch_radon import ConeBeam
from torch_radon.volumes import Volume3D
from tqdm import tqdm

from ct_utils import hu2mu, create_radon


def create_projections(radon: ConeBeam,
                       path_to_nifti: str,
                       projections_path: str):
    nib_volume = nib.load(path_to_nifti)
    nib_dims = tuple(float(f) for f in nib_volume.header['pixdim'][1:4])
    nib_volume = hu2mu(nib_volume.get_fdata())
    nib_volume[nib_volume < 0] = 0
    volume = torch.from_numpy(nib_volume.transpose()).float().cuda()

    radon.volume = Volume3D(
        depth=volume.shape[0],
        height=volume.shape[1],
        width=volume.shape[2],
        voxel_size=nib_dims,
    )

    projections = radon.forward(volume[None, None])
    if projections.isnan().any():
        print(path_to_nifti)
        print(volume.isnan().any())
        projections.nan_to_num_(0)

    os.makedirs(projections_path, exist_ok=True)
    file_name = path_to_nifti[path_to_nifti.rfind(os.sep)+1:].split('.', 1)[0]
    h5 = h5py.File(pjoin(projections_path, f'{file_name}.h5'), "w")
    h5.create_dataset('projections', data=projections[0, 0].cpu().numpy())


def create_all_projections():
    data_dir = '/mnt/nvme2/lungs/lungs3d/'
    for data_file in tqdm([f for f in sorted(os.listdir(data_dir)) if f.endswith('.nii.gz')]):
        create_projections(
            create_radon(),
            pjoin(data_dir, data_file),
            'projections',
        )


if __name__ == "__main__":
    create_all_projections()
