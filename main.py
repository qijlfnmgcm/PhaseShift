import configparser
import os

import utils
from wrapper import Wrapper

config = configparser.ConfigParser()
config.read("./config.txt", "UTF-8")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.get("SETTINGS", "cuda_num")

    CREATED_DIR = utils.create_result_dir(".")

    model = Wrapper(config, CREATED_DIR)
    model.execute()


if __name__ == "__main__":
    main()
