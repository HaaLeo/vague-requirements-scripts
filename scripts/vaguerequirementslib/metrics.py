import logging
import pandas as pd
from .constants import TP, TN, FP, FN, VAGUE_LABEL, NOT_VAGUE_LABEL

LOGGER = logging.getLogger(__name__)


def calc_all_metrics(**kwargs) -> dict:
    return {
        'accuracy': calc_accuracy(**kwargs),
        'precision': calc_precision(**kwargs),
        'recall': calc_recall(**kwargs),
        'specificity': calc_specificity(**kwargs),
        'false_negative_rate': calc_false_negative_rate(**kwargs),
        'false_positive_rate': calc_false_positive_rate(**kwargs),
        'f1_score': calc_f1_score(**kwargs)
    }


def calc_accuracy(**kwargs) -> float:
    dividend = kwargs[TP] + kwargs[TN]
    denominator = kwargs[TP] + kwargs[TN] + kwargs[FP] + kwargs[FN]
    return _build_quotient(dividend, denominator)


def calc_precision(**kwargs) -> float:
    dividend = kwargs[TP]
    denominator = kwargs[TP] + kwargs[FP]
    return _build_quotient(dividend, denominator)


def calc_recall(**kwargs) -> float:
    dividend = kwargs[TP]
    denominator = kwargs[TP] + kwargs[FN]
    return _build_quotient(dividend, denominator)


def calc_specificity(**kwargs) -> float:
    dividend = kwargs[TN]
    denominator = kwargs[FP] + kwargs[TN]
    return _build_quotient(dividend, denominator)


def calc_false_negative_rate(**kwargs) -> float:
    dividend = kwargs[FN]
    denominator = kwargs[TP] + kwargs[FN]
    return _build_quotient(dividend, denominator)


def calc_false_positive_rate(**kwargs) -> float:
    dividend = kwargs[FP]
    denominator = kwargs[FP] + kwargs[TN]
    return _build_quotient(dividend, denominator)


def calc_f1_score(**kwargs):
    precision = calc_precision(**kwargs)
    recall = calc_recall(**kwargs)

    dividend = 2 * precision * recall
    denominator = precision + recall
    return _build_quotient(dividend, denominator)


def calc_mean_average_precision(df: pd.DataFrame, vague_prob_column='vague_prob', not_vague_prob_column='not_vague_prob', ground_truth_column='majority_label') -> float:
    """
    Calculate the mean average precision for a given data frame.
    Example data frame:
       vague_prob  not_vague_prob  majority_label
    0        0.98            0.02               1
    1        0.60            0.40               0

    Args:
        df (pd.DataFrame): [description]

    Returns:
        float: [description]
    """

    for label in [VAGUE_LABEL, NOT_VAGUE_LABEL]:
        target_column = vague_prob_column if label == VAGUE_LABEL else not_vague_prob_column
        sorted_df = df.sort_values(by=target_column)



def _build_quotient(dividend, denominator):
    if denominator != 0:
        return dividend / denominator
    else:
        result = 0
        LOGGER.warning(f'Denominator = 0. Skip metric calculation and set it to value="{result}".')
        return result
