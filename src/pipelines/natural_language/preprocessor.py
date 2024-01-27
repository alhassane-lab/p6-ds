import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

""" Module for data preprocessing """

from datetime import datetime
from src.utils import setup_logging
from src.pipelines.text.integrator import DataIntegrator

TODAY = datetime.today().strftime("%Y%m%d")
logger = setup_logging("P6-DS-Preprocessing", "processing")


class DataPreprocessor(DataIntegrator):
    """ Class for data transforming and cleaning"""

    def __init__(self) -> None:
        super().__init__()
        self.data = self.load_csv(self.data_origins)

    def parse_category_tree(self: str, level: int, i) -> str:
        return self.split(">>")[level].strip('["]')

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:

        tmp = data.description.str.len()
        data['description_length'] = tmp
        for i in range(2):
            data[f'category_{i}'] = data.product_category_tree.apply(
                lambda x: self.parse_category_tree(x, i))
            logger.info(f"Successfully extracted 4 new features ! - data shape: {data.shape}")
        return data

    @staticmethod
    def select_features(data: pd.DataFrame, features_to_keep=None):
        if features_to_keep is None:
            features_to_keep = ['uniq_id', 'description', 'category_1', 'category_0', 'description_length']
        features = features_to_keep
        return data[features]

    def univariate(self, data: pd.DataFrame, arg: str, ) -> None:
        """ Univariate distributio plots"""
        logger.info(f"Feature : {arg}")
        fig = plt.figure(figsize=(14, 7))
        plt.style.use('seaborn-v0_8-pastel')
        plt.suptitle(f"{arg} distribution", fontsize=14)
        ax1 = fig.add_subplot(121)
        sns.boxenplot(data[arg], ax=ax1)
        ax2 = fig.add_subplot(122)
        sns.histplot(data[arg], ax=ax2)
        plt.tight_layout()
        plt.savefig(self.outputs_dir + f"plots/univariate_{arg}_plot.png")
        logger.info(f"{arg} distribution plot saved : outputs/plots/bivariate_{arg}.png")

    def bivariate(self, data: pd.DataFrame, args: tuple[str, str]) -> None:
        """
        """
        logger.info(f"========== Bivariate Analysis ==========")
        logger.info(f"Features : {args[0]} & {args[1]}")
        plt.figure(figsize=(14, 7))
        sns.boxplot(x=data[args[0]], y=data[args[1]], showmeans=True, orient='h')
        plt.savefig(self.outputs_dir + f"plots/bivariate_{args[0]}_{args[1]}.png")
        plt.title(f"{args[0]} per {args[1]}")
        logger.info(
            f"{args[0]} x {args[1]} distribution plot saved : outputs/plots/bivariate_{args[0]}_{args[1]}_plot.png")

    def anova(self, data: pd.DataFrame, args: tuple[str, list]) -> None:
        """
        Performs an Analysis of Variance (ANOVA) test on the given data.
        @data : data table
        @args: a tuple of one numerical and one categorical
        """
        logger.info(f"anova test : {args[0]} & {args[1][0]}")
        model = smf.ols(args[0] + " ~ " + " + ".join(args[1]), data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_value = anova_table["PR(>F)"].loc[args[1][-1]]
        logger.info(f"pvalue : {p_value}")
        if p_value < 0.05:
            hypothesis = f"{args[0]} has a significant effect on one category at least"
            logger.info(f"Reject H0. {hypothesis}")
        else:
            hypothesis = f"{args[0]} has no effect on category"
            logger.info(f"Fail to reject H0. {hypothesis}")

    def outliers(self, data: pd.DataFrame, arg: str):
        """
        Detects Outliers for a numerical target
        @data : data table
        @arg: a numerical target
        """
        data_array = np.array(data[arg])
        threshold = 4
        logger.info(f"Method : Percentile <-->  Threshold: {threshold}")
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outliers_list = data_array[(data_array < lower_bound) | (data_array > upper_bound)]
        logger.info(f"Number of outliers_list: {len(outliers_list)}")

        ok = data[~data[arg].isin(outliers_list)]
        out = data[data[arg].isin(outliers_list)]
        file_name = f"data/{arg}_outliers.csv"
        out.to_csv(self.outputs_dir + file_name, index=False)
        logger.info(f"Outliers saved to : outputs/{file_name}")

        plt.figure(figsize=(14, 7))
        sns.scatterplot(x=ok[arg].index, y=ok[arg].values, )
        sns.scatterplot(x=out[arg].index, y=out[arg].values)
        plt.title(f'{arg} Outliers')
        file_name = f"plots/outliers_{arg}_plot_.png"
        plt.savefig(self.outputs_dir + file_name)
        logger.info(f"Outliers plot saved to : outputs/{file_name}")

        return ok, out

    def perform_eda(self, eda_results_path):
        numerical = 'description_length'
        categorical = 'category_0'
        data = self.data

        self.univariate(data, numerical)
        self.bivariate(data, (numerical, categorical))
        self.anova(self.data, (numerical, [categorical]))
        final_data, outs = self.outliers(self.data, numerical)
        logger.info(f"{len(outs)} outliers cleaned...")
        logger.info("Processed Data shape: {}".format(self.data.shape))
        return final_data
