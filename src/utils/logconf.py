import logging
from .envconf import get_var_envs
import os

vars_envs = get_var_envs()


def setup_logging(logger_name: str, file_name):
    root_dir = vars_envs['root']
    logs_folder = os.path.join(root_dir, "outputs/logs/")
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('"%(asctime)s - %(name)s - %(levelname)s - %(message)s"')

    file_handler = logging.FileHandler(logs_folder + file_name + '.log')
    file_handler.set_name("formatter")
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
