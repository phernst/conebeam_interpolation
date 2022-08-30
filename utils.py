import json

import h5py
import nibabel as nib


def load_nib(path: str):
    return nib.load(path).get_fdata()


def load_normalizations():
    with open('normalization.json', 'r', encoding='utf-8') as file:
        return json.load(file)


def load_h5(path: str):
    return h5py.File(path, 'r', libver='latest', swmr=True)['projections']


NORMALIZATION = load_normalizations()
