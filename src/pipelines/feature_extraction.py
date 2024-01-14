from src.utils import setup_logging, get_var_envs

logger = setup_logging("Preprocessing")
var_envs = get_var_envs()


class FeatureExtractor:
    """
    This class contains functions for data exploration, cleaning and preprocessing.
    """

    def __init__(self) -> None:
        """
        Initializes the DataProcessor class.
        """
        logger.info("========== Initializing ==========")
        print(Path(".").resolve())
        root_dir = var_envs['root']
        self.data_dir = os.path.join(root_dir, 'data/')
        logger.info("Data directory is loaded !!!")
        self.outputs_dir = os.path.join(root_dir, 'outputs/')
        logger.info("Outputs directory is loaded !!!")
        model = "en_core_web_sm"
        self.nlp = spacy.load(model)
        logger.info(f"Spacy model {model} is loaded !!!")

    def load_data(self) -> pd.DataFrame:
        """
        @param file_name: input data file name
        @return: dataframe
        """
        logger.info("========== Loading Data... ==========")
        data = pd.read_csv(self.data_dir + var_envs['file'])
        logger.info(f"Features: {list(data.columns)}")
        logger.info(f"Raw data shape: {data.shape}")
        return data

