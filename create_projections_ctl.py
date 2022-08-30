import os
from os.path import join as pjoin

import ctl
import h5py
import nibabel as nib
from tqdm import tqdm

from ct_utils import hu2mu


def create_projections(path_to_nifti: str,
                       projections_path: str):
    nib_vol = nib.load(path_to_nifti)
    nib_dims = tuple(float(f) for f in nib_vol.header['pixdim'][1:4])
    nib_vol = hu2mu(nib_vol.get_fdata())
    nib_vol[nib_vol < 0] = 0
    volume = ctl.VoxelVolumeF.from_numpy(nib_vol.transpose())
    volume.set_voxel_size(nib_dims)

    det_pixels = (256, 256)
    det_spacings = (2.0, 2.0)
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
    projections = projector.configure_and_project(setup, volume).numpy()[:, 0]

    os.makedirs(projections_path, exist_ok=True)
    file_name = path_to_nifti[path_to_nifti.rfind(os.sep)+1:].split('.', 1)[0]
    h5 = h5py.File(pjoin(projections_path, f'{file_name}.h5'), "w")
    h5.create_dataset('projections', data=projections)


def create_all_projections():
    data_dir = '/mnt/nvme2/lungs/lungs3d/'
    for data_file in tqdm([f for f in sorted(os.listdir(data_dir)) if f.endswith('.nii.gz')]):
        create_projections(
            pjoin(data_dir, data_file),
            'projections_ctl',
        )


if __name__ == "__main__":
    create_all_projections()
