"""
Get environment variables from .env
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def get_var_envs():
    """
    Post-processing method that creates an instance of EnvConf
    """
    # ! auto activate .env
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    kwargs = {
        "root": Path(os.getenv('ROOT_DIR')),
        'file': os.getenv("FILE_NAME"),
    }
    return kwargs


def get_emb_params():
    params = {
        "w2v_size": 300,
        "w2v_window": 5,
        "w2v_min_count": 1,
        "w2v_epochs": 100,
        "maxlen": 24
    }
    return params
