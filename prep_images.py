# coding: utf-8
import os
import errno
import shutil
from keras.utils import to_categorical
import numpy as np


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_image_categories(image_path='msl-images'):
    classes = {}
    with open(os.path.join(image_path, 'msl_synset_words-indexed.txt')) as f:
        lines = f.read().splitlines()
        for line in lines:
            print(line)
            num, name = line.split('    ')
            # classes[int(num)] = name.replace(' ', '_')
            classes[num] = name.replace(' ', '_')

    return classes


def split_data(image_path='msl-images'):
    classes = get_image_categories(image_path)

    for type_ in ('train', 'val', 'test'):
        split_path = os.path.join(image_path, type_)
        mkdir_p(split_path)
        for label_name in classes.values():
            mkdir_p(os.path.join(split_path, label_name))

        split_name = os.path.join(image_path, '%s-calibrated-shuffled.txt' % type_)
        with open(split_name) as f:
            image_list, label_strs = zip(*[line.split() for line in f.read().splitlines()])

        # labels = to_categorical(np.array([int(label) for label in label_strs]))
        # np.save(os.path.join(image_path, type_, "labels.npy"), labels)

        for label, img in zip(label_strs, image_list):
            label_name = classes[label]
            shutil.copy(os.path.join(image_path, img), os.path.join(split_path, label_name))
