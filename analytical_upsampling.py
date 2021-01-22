import argparse
import json
from os.path import join as pjoin

import ctl
import nibabel as nib
import numpy as np
import torch

from mathutils import interpolate_projections
from utils import DATA_DIRS


# upsampling in [2, 4, 8]
# TODO: remove redundant code
def main(test_subject: int, upsampling: int):
    with open('train_valid.json', 'r') as json_file:
        json_dict = json.load(json_file)
        test_files = json_dict['test_files']
    if test_files is None:
        print("no test files")
        return

    nib_volume = nib.load(pjoin(DATA_DIRS['datasets'], f'{test_files[test_subject]}'))
    nib_dims = tuple([float(f) for f in nib_volume.header['pixdim'][1:4]])
    nib_volume = nib_volume.get_fdata()
    volume = ctl.VoxelVolumeF.from_numpy(nib_volume.transpose())
    volume.set_voxel_size(nib_dims)

    system = ctl.SimpleCTSystem(
        detector=ctl.FlatPanelDetector((256, 256), (4.0, 4.0)),
        gantry=ctl.TubularGantry(1000, 750),
        source=ctl.XrayTube(),
    )
    assert system.is_valid()
    setup = ctl.AcquisitionSetup(system, nb_views=360)
    setup.apply_preparation_protocol(ctl.protocols.AxialScanTrajectory())

    projection_matrices = ctl.GeometryEncoder.encode_full_geometry(setup)
    projection_matrices = [p[0] for p in projection_matrices]

    projector = ctl.ocl.RayCasterProjector()
    projections = projector.configure_and_project(setup, volume).numpy()
    print(f'projections: {projections.shape}')

    det_pixels = system.detector().nb_pixels()
    uv_grid = torch.stack(torch.meshgrid(
        torch.arange(det_pixels[0]),
        torch.arange(det_pixels[1])), dim=-1)
    uv_grid = uv_grid.double()

    interpolation_volume = np.zeros_like(projections)
    interpolation_volume[::upsampling] = projections[::upsampling]
    for idx in range(upsampling//2, 360, upsampling):
        print(f'calculating interpolation {idx}...')
        g_approx_0, g_approx_1 = interpolate_projections(
            torch.from_numpy(interpolation_volume[(idx - upsampling//2) % 360]).cuda(),
            torch.from_numpy(interpolation_volume[(idx + upsampling//2) % 360]).cuda(),
            projection_matrices[(idx - upsampling//2) % 360],
            projection_matrices[(idx + upsampling//2) % 360],
            projection_matrices[idx],
            uv_grid,
        )
        interpolation_volume[idx, 0] = ((g_approx_0 + g_approx_1)/2).cpu().numpy().transpose()

    if upsampling > 2:
        for idx in range(0, 360, upsampling):
            print(f'calculating interpolation {idx + upsampling//4}...')
            g_approx_0, g_approx_1 = interpolate_projections(
                torch.from_numpy(interpolation_volume[(idx) % 360]).cuda(),
                torch.from_numpy(interpolation_volume[(idx + upsampling) % 360]).cuda(),
                projection_matrices[(idx) % 360],
                projection_matrices[(idx + upsampling) % 360],
                projection_matrices[idx + upsampling//4],
                uv_grid,
            )
            interpolation_volume[idx + upsampling//4, 0] = (0.75*g_approx_0 + 0.25*g_approx_1).cpu().numpy().transpose()
            print(f'calculating interpolation {idx + upsampling//4 + upsampling//2}...')
            g_approx_0, g_approx_1 = interpolate_projections(
                torch.from_numpy(interpolation_volume[(idx) % 360]).cuda(),
                torch.from_numpy(interpolation_volume[(idx + upsampling) % 360]).cuda(),
                projection_matrices[(idx) % 360],
                projection_matrices[(idx + upsampling) % 360],
                projection_matrices[idx + upsampling//4 + upsampling//2],
                uv_grid,
            )
            interpolation_volume[idx + upsampling//4 + upsampling//2, 0] = (0.25*g_approx_0 + 0.75*g_approx_1).cpu().numpy().transpose()

    if upsampling > 4:
        for idx in range(0, 360, upsampling):
            print(f'calculating interpolation {idx + upsampling//8}...')
            g_approx_0, g_approx_1 = interpolate_projections(
                torch.from_numpy(interpolation_volume[(idx) % 360]).cuda(),
                torch.from_numpy(interpolation_volume[(idx + upsampling) % 360]).cuda(),
                projection_matrices[(idx) % 360],
                projection_matrices[(idx + upsampling) % 360],
                projection_matrices[idx + upsampling//8],
                uv_grid,
            )
            interpolation_volume[idx + upsampling//8, 0] = (0.875*g_approx_0 + 0.125*g_approx_1).cpu().numpy().transpose()
            print(f'calculating interpolation {idx + upsampling//8 + upsampling//4}...')
            g_approx_0, g_approx_1 = interpolate_projections(
                torch.from_numpy(interpolation_volume[(idx) % 360]).cuda(),
                torch.from_numpy(interpolation_volume[(idx + upsampling) % 360]).cuda(),
                projection_matrices[(idx) % 360],
                projection_matrices[(idx + upsampling) % 360],
                projection_matrices[idx + upsampling//8 + upsampling//4],
                uv_grid,
            )
            interpolation_volume[idx + upsampling//8 + upsampling//4, 0] = (0.625*g_approx_0 + 0.375*g_approx_1).cpu().numpy().transpose()
            print(f'calculating interpolation {idx + upsampling//8 + 2*upsampling//4}...')
            g_approx_0, g_approx_1 = interpolate_projections(
                torch.from_numpy(interpolation_volume[(idx) % 360]).cuda(),
                torch.from_numpy(interpolation_volume[(idx + upsampling) % 360]).cuda(),
                projection_matrices[(idx) % 360],
                projection_matrices[(idx + upsampling) % 360],
                projection_matrices[idx + upsampling//8 + 2*upsampling//4],
                uv_grid,
            )
            interpolation_volume[idx + upsampling//8 + 2*upsampling//4, 0] = (0.375*g_approx_0 + 0.625*g_approx_1).cpu().numpy().transpose()
            print(f'calculating interpolation {idx + upsampling//8 + 3*upsampling//4}...')
            g_approx_0, g_approx_1 = interpolate_projections(
                torch.from_numpy(interpolation_volume[(idx) % 360]).cuda(),
                torch.from_numpy(interpolation_volume[(idx + upsampling) % 360]).cuda(),
                projection_matrices[(idx) % 360],
                projection_matrices[(idx + upsampling) % 360],
                projection_matrices[idx + upsampling//8 + 3*upsampling//4],
                uv_grid,
            )
            interpolation_volume[idx + upsampling//8 + 3*upsampling//4, 0] = (0.125*g_approx_0 + 0.875*g_approx_1).cpu().numpy().transpose()

    img = nib.Nifti1Image(interpolation_volume[:, 0].transpose(1, 2, 0), np.eye(4))
    nib.save(img, f'analytical_up{upsampling//2}_{test_subject}.nii.gz')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_subject', type=int,
                        help='index of test subject (0 or 1)')
    parser.add_argument('upsampling', type=int, default=8,
                        help='upsampling factor (2, 4 or 8)')
    args = parser.parse_args()
    main(test_subject=args.test_subject, upsampling=args.upsampling)
