import os
from os.path import join as pjoin

import ctl
import ctl.gui
import nibabel as nib
import numpy as np

from utils import DATA_DIRS


def create_system():
    system = ctl.SimpleCTSystem(
        detector=ctl.FlatPanelDetector((256, 256), (4.0, 4.0)),
        gantry=ctl.TubularGantry(1000, 750),
        source=ctl.XrayTube(),
    )
    assert system.is_valid()
    return system


def create_projections(system: ctl.SimpleCTSystem,
                       path_to_nifti: str,
                       projections_path: str):
    nib_volume = nib.load(path_to_nifti)
    nib_dims = tuple([float(f) for f in nib_volume.header['pixdim'][1:4]])
    nib_volume = nib_volume.get_fdata()
    volume = ctl.VoxelVolumeF.from_numpy(nib_volume.transpose())
    volume.set_voxel_size(nib_dims)

    num_views = 360

    setup = ctl.AcquisitionSetup(system, num_views)
    setup.apply_preparation_protocol(ctl.protocols.AxialScanTrajectory())

    projector = ctl.ocl.RayCasterProjector()
    projections = projector.configure_and_project(setup, volume).numpy()

    os.makedirs(projections_path, exist_ok=True)
    file_name = path_to_nifti[path_to_nifti.rfind(os.sep)+1:]
    img = nib.Nifti1Image(projections[:, 0].transpose(), np.eye(4))
    nib.save(img, f"{projections_path}/{file_name}")


def create_all_projections():
    data_dir = DATA_DIRS['datasets']
    for data_file in sorted(os.listdir(data_dir)):
        print(f'processing: {data_file}')
        create_projections(
            create_system(),
            pjoin(data_dir, data_file),
            'projections',
        )


if __name__ == "__main__":
    create_all_projections()
