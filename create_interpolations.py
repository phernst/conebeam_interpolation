import argparse
import os
from os.path import join as pjoin

import ctl
import ctl.gui
import nibabel as nib
import numpy as np
import torch

from create_projections import create_system
from mathutils import interpolate_projections
from utils import DATA_DIRS


def create_interpolations(
        system: ctl.SimpleCTSystem,
        neighbor_diff: int,
        path_to_nifti: str,
        interpolations_path: str):
    nib_volume = nib.load(path_to_nifti)
    nib_dims = tuple([float(f) for f in nib_volume.header['pixdim'][1:4]])
    nib_volume = nib_volume.get_fdata()
    volume = ctl.VoxelVolumeF.from_numpy(nib_volume.transpose())
    volume.set_voxel_size(nib_dims)

    num_views = 360

    setup = ctl.AcquisitionSetup(system, num_views)
    setup.apply_preparation_protocol(ctl.protocols.AxialScanTrajectory())

    projection_matrices = ctl.GeometryEncoder.encode_full_geometry(setup)
    projection_matrices = [p[0] for p in projection_matrices]

    projector = ctl.ocl.RayCasterProjector()
    projections = projector.configure_and_project(setup, volume).numpy()

    det_pixels = system.detector().nb_pixels()
    uv_grid = torch.stack(torch.meshgrid(torch.arange(det_pixels[0]), torch.arange(det_pixels[1])), dim=-1)
    uv_grid = uv_grid.double()

    interpolation_volume = torch.zeros(*(det_pixels + (num_views,)), dtype=torch.float, device='cuda')

    for idx in range(num_views):
        print(f'calculating interpolation {idx+1}/{num_views}...')
        g_approx_0, g_approx_1 = interpolate_projections(
            torch.from_numpy(projections[(idx - neighbor_diff) % num_views]).cuda(),
            torch.from_numpy(projections[(idx + neighbor_diff) % num_views]).cuda(),
            projection_matrices[(idx - neighbor_diff) % num_views],
            projection_matrices[(idx + neighbor_diff) % num_views],
            projection_matrices[idx],
            uv_grid,
        )
        interpolation_volume[:, :, idx] = (g_approx_0 + g_approx_1)/2

    os.makedirs(interpolations_path, exist_ok=True)
    file_name = path_to_nifti[path_to_nifti.rfind(os.sep)+1:]
    img = nib.Nifti1Image(interpolation_volume.cpu().numpy(), np.eye(4))
    nib.save(img, f"{interpolations_path}/{file_name}")


def create_all_interpolations(neighbor_diff: int):
    assert neighbor_diff in [1, 2, 4]
    data_dir = DATA_DIRS['datasets']
    for data_file in sorted(os.listdir(data_dir)):
        print(f'processing: {data_file}')
        create_interpolations(
            create_system(),
            neighbor_diff,
            pjoin(data_dir, data_file),
            f'interpolations_nd{neighbor_diff}',
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('neighbor_diff', type=int,
                        help='angular difference between interpolated '
                        'projection and next neighbor in degrees (1, 2 or 4)')
    args = parser.parse_args()
    create_all_interpolations(neighbor_diff=args.neighbor_diff)
