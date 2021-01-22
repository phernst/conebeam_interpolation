import json
import os

import nibabel as nib


def load_nib(path: str):
    return nib.load(path).get_fdata()


def load_normalizations():
    with open('normalization.json', 'r') as file:
        return json.load(file)


def load_data_dirs():
    with open('data_dirs.json', 'r') as file:
        return {k: os.path.abspath(v) for (k, v) in json.load(file).items()}


def load_configuration():
    with open('configuration.json', 'r') as file:
        return json.load(file)


NORMALIZATION = load_normalizations()
DATA_DIRS = load_data_dirs()
CONFIGURATION = load_configuration()
