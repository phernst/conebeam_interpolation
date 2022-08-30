import json
import os
from os.path import join as pjoin

import numpy as np

from utils import load_h5


def calculate_and_save_normalization():
    projection_path = 'projections'

    def normfunction(x):
        perc = np.percentile(x, 99)
        print(perc)
        return perc

    all_normvalues = []
    for fname in os.listdir(projection_path):
        print(fname)
        img = load_h5(pjoin(projection_path, fname))
        all_normvalues.append(normfunction(img))

    with open('normalization.json', 'w', encoding='utf-8') as json_file:
        json.dump({
            "projections_99": np.median(all_normvalues),
        }, json_file)


if __name__ == "__main__":
    calculate_and_save_normalization()
