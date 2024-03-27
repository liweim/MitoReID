import logging
import os
import os.path as osp

def get_logger(path):
    dirname = osp.split(path)[0]
    if not osp.exists(dirname):
        os.makedirs(dirname)

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=path, encoding='UTF-8')

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    return logger