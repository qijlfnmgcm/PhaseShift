import datetime
import os
import random
import shutil

import numpy as np
import tensorflow as tf
import torch
from tensorflow import keras


def get_images():
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = torch.tensor(train_images / 255)
    test_images = torch.tensor(test_images / 255)

    train_images = train_images.reshape(-1, 28 * 28)
    test_images = test_images.reshape(-1, 28 * 28)

    return train_images, train_labels, test_images, test_labels


def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)


def create_result_dir(BASE_DIR):
    # create a directory to save the results
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    time = datetime.datetime.now()
    DIR_NAME = time.strftime("%Y.%m.%d.%H.%M.%S.%f")
    CREATED_DIR = os.path.join(RESULTS_DIR, DIR_NAME)
    os.makedirs(CREATED_DIR, exist_ok=True)

    # copy the config file
    shutil.copyfile(
        os.path.join(BASE_DIR, "config.txt"),
        os.path.join(CREATED_DIR, "config.txt"),
    )

    return CREATED_DIR
