""" Module for data integration --> loading and saving """

import os
from datetime import datetime
import pandas as pd
from src.utils import setup_logging, load_args
import json

TODAY = datetime.today().strftime("%Y%m%d")
logger = setup_logging("P6-DS-Integration", "processing")


class DataIntegrator:
    """ Class for data import and export. """
    root_dir, data_origins = load_args()

    def __init__(self) -> None:
        logger.info(" @@@ Initializing ...")
        self.data_dir = os.path.join(self.root_dir, 'data/')
        self.outputs_dir = os.path.join(self.root_dir, 'outputs/')

    def load_csv(self, file_name: str) -> pd.DataFrame:
        """ Load a csv file """
        data_path = self.data_dir + file_name
        data = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data - shape: {data.shape}")
        return data

    def save_csv(self, data: pd.DataFrame, file_name: str) -> None:
        """ Save data to outputs folder as csv file """
        path = f"data/{file_name}_{TODAY}.csv"
        data.to_csv(self.outputs_dir + path, index=False)
        logger.info(f"Successfully save data to outputs/{path}")

    def read_json(self, file_name: str):
        file_path = os.path.join(self.root_dir, f"outputs/data/{file_name}.json")
        with open(file_path, "r") as f:
            json_file = json.load(f)
        logger.info(f"Successfully opened the json file")
        return json_file

    def save_json(self, file_name: str, data) -> None:
        file_path = f"outputs/data/{file_name}.json"
        full_path = os.path.join(self.root_dir, file_path)
        with open(full_path, "w") as outfile:
            json.dump(data, outfile)
            logger.info(f"Successfully save data to outputs/{file_path}")
