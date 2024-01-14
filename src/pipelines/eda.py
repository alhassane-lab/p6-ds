import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from src.utils import setup_logging

logger = setup_logging('Exploratory-Data-Analysis')


def univariate(data: object, arg: str, outputs_dir: str) -> None:
    """
    Performs univariate analysis.
    @data : data table
    @arg: a numerical feature
    @results_dir: results saving directory path
    """
    logger.info(f"========== Univariate Analysis ==========")
    logger.info(f"Feature : {arg}")
    fig = plt.figure(figsize=(14, 7))
    plt.style.use('seaborn-v0_8-pastel')
    plt.suptitle(f"{arg} distribution", fontsize=14)
    ax1 = fig.add_subplot(121)
    sns.boxenplot(data[arg], ax=ax1)
    ax2 = fig.add_subplot(122)
    sns.histplot(data[arg], ax=ax2)
    plt.tight_layout()
    plt.savefig(outputs_dir + f"plots/univariate_{arg}_plot.png")
    logger.info(f"{arg} distribution plot saved : outputs/plots/bivariate_{arg}.png")


def bivariate(data: object, args: tuple[str, str], outputs_dir: str) -> None:
    """
    Performs bivariate analysis.
    @data : data table
    @args: a tuple of one numerical and one categorical
    @results_dir: results saving directory path
    """
    logger.info(f"========== Bivariate Analysis ==========")
    logger.info(f"Features : {args[0]} & {args[1]}")
    plt.figure(figsize=(14, 7))
    sns.boxplot(x=data[args[0]], y=data[args[1]], showmeans=True, orient='h')
    plt.savefig(outputs_dir + f"plots/bivariate_{args[0]}_{args[1]}.png")
    plt.title(f"{args[0]} per {args[1]}")
    logger.info(f"{args[0]} x {args[1]} distribution plot saved : outputs/plots/bivariate_{args[0]}_{args[1]}_plot.png")


def anova(data: object, args: tuple[str, list]) -> None:
    """
    Performs an Analysis of Variance (ANOVA) test on the given data.
    @data : data table
    @args: a tuple of one numerical and one categorical
    """
    logger.info(f"========== Correlation Test ==========")
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


def outliers(data: object, arg: str, outputs_dir: str):
    """
    Detects Outliers for a numerical feature
    @data : data table
    @arg: a numerical feature
    """
    logger.info(f"========== Outliers Analysis ==========")
    data_array = np.array(data[arg])
    threshold = 3
    logger.info(f"Method : Percentile <-->  Threshold: {threshold}")
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outliers = data_array[(data_array < lower_bound) | (data_array > upper_bound)]
    logger.info(f"Number of outliers: {len(outliers)}")

    ok = data[~data[arg].isin(outliers)]
    out = data[data[arg].isin(outliers)]
    file_name = f"data/{arg}_outliers.csv"
    out.to_csv(outputs_dir + file_name, index=False)
    logger.info(f"Outliers saved to : outputs/{file_name}")

    plt.figure(figsize=(14, 7))
    sns.scatterplot(x=ok[arg].index, y=ok[arg].values, )
    sns.scatterplot(x=out[arg].index, y=out[arg].values)
    plt.title(f'{arg} Outliers')
    file_name = f"plots/outliers_{arg}_plot_.png"
    plt.savefig(outputs_dir + file_name)
    logger.info(f"Outliers plot saved to : outputs/{file_name}")

    return ok, out
