import json
import os
import random

import numpy as np

from utils import DATA_DIRS


def create_and_save_train_valid():
    all_files = os.listdir(DATA_DIRS['datasets'])
    random.shuffle(all_files)
    split_idx = np.cumsum([round(sp*len(all_files)) for sp in [0.7, 0.2, 0.1]])
    train_files = all_files[:split_idx[0]]
    valid_files = all_files[split_idx[0]:split_idx[1]]
    test_files = all_files[split_idx[1]:split_idx[2]]
    with open('train_valid.json', 'w') as json_file:
        json.dump({
            'train_files': train_files,
            'valid_files': valid_files,
            'test_files': test_files,
        }, json_file)


if __name__ == '__main__':
    create_and_save_train_valid()
