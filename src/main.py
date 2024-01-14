import os

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.pipelines.data_preprocessing import DataProcessor


def main() -> None:
    """
    Root command.
    """

    preprocessor = DataProcessor()
    raw_data = preprocessor.load_data()
    cleaned_data = preprocessor.process_data(raw_data)
    cleaned_data = preprocessor.perform_eda(cleaned_data)
    final_data = preprocessor.full_process_data(cleaned_data)


if __name__ == "__main__":
    main()

