import os
from os.path import join as pjoin

import h5py
import nibabel as nib
import torch
import torch.nn.functional as F
from torch_radon import ConeBeam
from torch_radon.volumes import Volume3D
from tqdm import tqdm

from ct_utils import fdk_reconstruction, hu2mu


def create_interpolations(path_to_volume: str, neighbor_diff: int):
    nib_vol = nib.load(path_to_volume)
    nib_dims = tuple(float(f) for f in nib_vol.header['pixdim'][1:4])

    volume = torch.from_numpy(nib_vol.get_fdata()).permute(2, 1, 0)
    volume = hu2mu(volume[None, None].float().cuda())
    volume[volume < 0] = 0

    num_views = 360
    det_pixels = (512, 512)
    det_spacings = (1.0, 1.0)
    radon = ConeBeam(
        det_count_u=det_pixels[0],
        angles=torch.deg2rad(torch.arange(360)),
        src_dist=750,
        det_dist=250,
        det_count_v=det_pixels[1],
        det_spacing_u=det_spacings[0],
        det_spacing_v=det_spacings[1],
    )
    radon.volume = Volume3D(
        depth=volume.shape[2],
        height=volume.shape[3],
        width=volume.shape[4],
        voxel_size=nib_dims,
    )
    projections = radon.forward(volume)
    if projections.isnan().any():
        print(f'full: {path_to_volume}, {neighbor_diff=}')
        projections.nan_to_num_(0)

    interpolation_volume = torch.zeros(
        *((num_views,) + det_pixels),
        dtype=torch.float,
        device='cuda',
    )

    for idx in range(2*neighbor_diff):
        sparse_idx = ((torch.arange(360) - neighbor_diff + idx) % 360)[::(2*neighbor_diff)]
        sparse_projections = projections[:, :, sparse_idx]
        sparse_angles = torch.deg2rad(sparse_idx)

        radon.angles = sparse_angles
        sparse_reco = fdk_reconstruction(sparse_projections, radon, 'ramp')
        sparse_reco[sparse_reco < 0] = 0

        inter_idx = (sparse_idx + neighbor_diff) % 360
        inter_angles = torch.deg2rad(inter_idx)
        radon.angles = inter_angles
        inter_projections = radon.forward(sparse_reco)
        if inter_projections.isnan().any():
            print(f'inter: {path_to_volume}, {neighbor_diff=}')
            inter_projections.nan_to_num_(0)

        interpolation_volume[inter_idx] = inter_projections[0, 0]

    interpolation_volume = F.interpolate(
        interpolation_volume[None, None],
        size=(360, 256, 256),
        mode='trilinear',
        align_corners=True,
    )[0, 0]

    interpolations_path = f'interpolations_nd{neighbor_diff}'
    os.makedirs(interpolations_path, exist_ok=True)
    file_name = path_to_volume[path_to_volume.rfind(os.sep)+1:].split('.', 1)[0]
    h5 = h5py.File(pjoin(interpolations_path, f'{file_name}.h5'), "w")
    h5.create_dataset('projections', data=interpolation_volume.cpu().numpy())


def create_all_interpolations():
    data_dir = '/mnt/nvme2/lungs/lungs3d/'
    itrt = [
        (f, nd)
        for f in sorted(os.listdir(data_dir))
        for nd in [1, 2, 4]
        if f.endswith('.nii.gz')
    ]
    for data_file, neighbor_diff in tqdm(itrt):
        create_interpolations(
            pjoin(data_dir, data_file),
            neighbor_diff,
        )


if __name__ == '__main__':
    create_all_interpolations()
