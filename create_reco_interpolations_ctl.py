import os
from os.path import join as pjoin

import ctl
from ctl import ocl
import h5py
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ct_utils import hu2mu


def create_interpolations(path_to_volume: str, neighbor_diff: int):
    nib_vol = nib.load(path_to_volume)
    nib_dims = tuple(float(f) for f in nib_vol.header['pixdim'][1:4])
    nib_vol = hu2mu(nib_vol.get_fdata())
    nib_vol[nib_vol < 0] = 0
    volume = ctl.VoxelVolumeF.from_numpy(nib_vol.transpose())
    volume.set_voxel_size(nib_dims)

    det_pixels = (512, 512)
    det_spacings = (1.0, 1.0)
    system = ctl.SimpleCTSystem(
        detector=ctl.FlatPanelDetector(
            nb_pixels=det_pixels,
            pixel_dimensions=det_spacings,
        ),
        gantry=ctl.TubularGantry(
            source_to_detector_distance=1000,
            source_to_iso_center_distance=750,
        ),
        source=ctl.XrayTube(),
    )
    num_views = 360
    setup = ctl.AcquisitionSetup(system, num_views)
    setup.apply_preparation_protocol(ctl.protocols.AxialScanTrajectory())

    projector = ctl.ocl.RayCasterProjector()
    projections = projector.configure_and_project(setup, volume)

    reconstructor = ctl.ocl.FDKReconstructor(ctl.ocl.ApodizationFilter(
        ctl.ocl.ApodizationFilter.FilterType.Hann, 1.))

    interpolation_volume = torch.zeros(
        *((num_views,) + det_pixels),
        dtype=torch.float,
        device='cuda',
    )

    for idx in range(2*neighbor_diff):
        sparse_idx = ((np.arange(360) - neighbor_diff + idx) % 360)[::(2*neighbor_diff)]
        sparse_projections = ctl.ProjectionDataView(projections, list(sparse_idx)).data_copy()

        sparse_setup = ctl.AcquisitionSetup(system, len(sparse_idx))
        sparse_setup.apply_preparation_protocol(ctl.protocols.AxialScanTrajectory(
            start_angle=np.deg2rad(sparse_idx[0]),
        ))

        sparse_reco = ctl.VoxelVolumeF(volume.dimensions(), volume.voxel_size())
        reconstructor.configure_and_reconstruct_to(
            sparse_setup,
            sparse_projections,
            sparse_reco,
        )

        inter_idx = (sparse_idx + neighbor_diff) % 360
        inter_setup = ctl.AcquisitionSetup(system, len(sparse_idx))
        inter_setup.apply_preparation_protocol(ctl.protocols.AxialScanTrajectory(
            start_angle=np.deg2rad(inter_idx[0]),
        ))
        inter_projections = projector.configure_and_project(inter_setup, sparse_reco)
        inter_projections = inter_projections.numpy()

        interpolation_volume[inter_idx] = torch.from_numpy(inter_projections[:, 0]).float().cuda()

    interpolation_volume = F.interpolate(
        interpolation_volume[None, None],
        size=(360, 256, 256),
        mode='trilinear',
        align_corners=True,
    )[0, 0]

    interpolations_path = f'interpolations_nd{neighbor_diff}_ctl'
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
