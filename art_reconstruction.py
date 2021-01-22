import argparse
from create_projections import create_system
import json
from os.path import join as pjoin

import ctl
import nibabel as nib
import numpy as np

from utils import DATA_DIRS, load_nib


# [view, row, col]
def projections_to_ctl(projections):
    result = ctl.ProjectionData(projections.shape[1], projections.shape[2], 1)
    for idx in range(projections.shape[0]):
        svd = ctl.SingleViewData(ctl.Chunk2DF.from_numpy(projections[idx]))
        result.append(svd)
    return result


def main(method: str, test_subject: int):
    assert method in ['ana', 'nn2', 'nn4', 'nn8', 'nn2+ana']
    with open('train_valid.json', 'r') as json_file:
        json_dict = json.load(json_file)
        test_files = json_dict['test_files']
    assert test_subject < len(test_files)

    projections = load_nib(pjoin('projections', test_files[test_subject]))
    projections = np.transpose(projections, (2, 1, 0)).astype(np.float32)
    projections = projections[::8]
    nib_volume = nib.load(pjoin(DATA_DIRS['datasets'], test_files[test_subject]))
    nib_dims = tuple([float(f) for f in nib_volume.header['pixdim'][1:4]])
    print(nib_dims)
    system = create_system()
    assert system.is_valid()
    num_views = 360//8
    setup = ctl.AcquisitionSetup(system, num_views)
    setup.apply_preparation_protocol(ctl.protocols.AxialScanTrajectory())
    rec = ctl.ARTReconstructor()
    rec.set_positivity_constrain_enabled(True)
    rec.set_relaxation_estimation_enabled(True)
    rec.set_max_nb_iterations(5)
    reco = ctl.VoxelVolumeF.from_numpy(
        np.transpose(load_nib(f'fdk_{method}_{test_subject}.nii.gz'),
                     (2, 1, 0)))
    reco.set_voxel_size(nib_dims)
    ctl_projections = projections_to_ctl(projections)
    rec.configure_and_reconstruct_to(setup, ctl_projections, reco)
    img = nib.Nifti1Image(np.transpose(reco.numpy(), (2, 1, 0)), np.eye(4))
    nib.save(img, f'art_{method}_{test_subject}.nii.gz')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=int,
                        help='upsampling method (ana, nn2, nn4, nn8, nn2+ana')
    args = parser.parse_args()
    main(method=args.method, test_subject=args.test_subject)
