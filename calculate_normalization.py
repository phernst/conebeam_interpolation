import json
import os
from os.path import join as pjoin

import nibabel as nib
import numpy as np

from utils import DATA_DIRS


def calculate_and_save_normalization():
    projection_path = DATA_DIRS['projections']

    def normfunction(x):
        return np.percentile(x, 99)

    all_normvalues = []
    for fname in os.listdir(projection_path):
        print(fname)
        img = nib.load(pjoin(projection_path, fname)).get_fdata()
        all_normvalues.append(normfunction(img))

    with open('normalization.json', 'w') as json_file:
        json.dump({
            "projections_99": np.median(all_normvalues),
        }, json_file)


if __name__ == "__main__":
    calculate_and_save_normalization()
