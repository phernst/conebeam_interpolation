import argparse
import json
from os.path import join as pjoin

import ctl
import nibabel as nib
import numpy as np

from create_projections import create_system
from utils import DATA_DIRS


def method2str(method: str):
    assert method in ['ana', 'nn2', 'nn4', 'nn8', 'nn2+ana']
    if method == 'ana':
        return 'analytical_up4'
    elif method == 'nn2':
        return 'upsampled_nd4_nd2_nd1_nn2'
    elif method == 'nn4':
        return 'upsampled_nd4_nd2_nd1_nn4'
    elif method == 'nn8':
        return 'upsampled_nd4_nd2_nd1_nn8'

    # else if method == 'nn2+ana':
    return 'upsampled_nd4_nd2_nd1'


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
        valid_files = json_dict['test_files']
    assert test_subject < len(valid_files)

    nib_volume = nib.load(pjoin(
        DATA_DIRS['datasets'], valid_files[test_subject]))
    nib_shape = nib_volume.header.get_data_shape()
    nib_dims = tuple([float(f) for f in nib_volume.header['pixdim'][1:4]])
    projections = nib.load(f'{method2str(method)}_{test_subject}.nii.gz')
    projections = projections.get_fdata()
    projections = np.transpose(projections, (2, 0, 1)).astype(np.float32)
    print(nib_dims)
    system = create_system()
    num_views = 360
    setup = ctl.AcquisitionSetup(system, num_views)
    setup.apply_preparation_protocol(ctl.protocols.AxialScanTrajectory())
    rec = ctl.ocl.FDKReconstructor()
    reco = ctl.VoxelVolumeF(nib_shape, nib_dims)
    reco.fill(0)
    ctl_projections = projections_to_ctl(projections)
    rec.configure_and_reconstruct_to(setup, ctl_projections, reco)
    img = nib.Nifti1Image(np.transpose(reco.numpy(), (2, 1, 0)), np.eye(4))
    nib.save(img, f'fdk_{method}_{test_subject}.nii.gz')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_subject', type=int,
                        help='index of test subject (0 or 1)')
    parser.add_argument('method', type=int,
                        help='upsampling method (ana, nn2, nn4, nn8, nn2+ana')
    args = parser.parse_args()
    main(method=args.method, test_subject=args.test_subject)
