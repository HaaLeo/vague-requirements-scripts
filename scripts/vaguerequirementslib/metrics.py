# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import logging
from typing import Tuple

import pandas as pd
import numpy as np

from .constants import TP, TN, FP, FN, VAGUE_LABEL, NOT_VAGUE_LABEL

LOGGER = logging.getLogger(__name__)

#pylint: disable=invalid-name

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


def calc_mean_average_precision(df: pd.DataFrame, vague_prob_column='vague_prob', not_vague_prob_column='not_vague_prob', ground_truth_column='majority_label') -> Tuple[float, float, float]:
    """
    Calculate the mean average precision for a given data frame.
    Example data frame:
       vague_prob  not_vague_prob  majority_label
    0        0.98            0.02               1
    1        0.60            0.40               0

    Args:
        df (pd.DataFrame): The sorted data frame
        vague_prob_column (str): The column name for vague probabilities
        not_vague_prob_column (str): The column name for not vague probabilites
        ground_truth_column (int): The column name for the ground truth

    Returns:
        float: mean average precision
    """

    precisions = [
        calc_average_precision_k(df, query=query, vague_prob_column=vague_prob_column, not_vague_prob_column=not_vague_prob_column, ground_truth_column=ground_truth_column)
        for query in ['vague', 'not_vague']
    ]

    # Binary case
    return (_build_quotient(np.sum(precisions), len(precisions)), *precisions)


def calc_average_precision_k(df: pd.DataFrame, query: str, k=None, vague_prob_column='vague_prob', not_vague_prob_column='not_vague_prob', ground_truth_column='majority_label') -> float: # pylint:disable=too-many-arguments
    """
    Calculate the average precision @ k.

    Args:
        df (pd.DataFrame): The sorted data frame
        vague_prob_column (str): The column name for vague probabilities
        not_vague_prob_column (str): The column name for not vague probabilites
        ground_truth_column (int): The column name for the ground truth
        query (str): The label to query for. 'vague' or 'not_vague'

    Returns:
        float: average precision @ k
    """

    if k is None or k > df.shape[0]:
        k = df.shape[0]
    LOGGER.info('Calculate average precision @ k for query="%s" and k="%s"', query, k)

    if query == 'vague':
        sorted_df = df.sort_values(by=vague_prob_column, ascending=False).reset_index(drop=True)
        query_column = vague_prob_column
        minor_column = not_vague_prob_column
        truth_label = VAGUE_LABEL
    elif query == 'not_vague':
        sorted_df = df.sort_values(by=not_vague_prob_column, ascending=False).reset_index(drop=True)
        query_column = not_vague_prob_column
        minor_column = vague_prob_column
        truth_label = NOT_VAGUE_LABEL
    else:
        raise ValueError(f'Query="{query}" is not supported.')

    score = 0
    num_hits = 0
    for index, row in sorted_df[:k].iterrows():

        # Only consider hits for the queried column
        if row[query_column] > row[minor_column]:
            prediction = truth_label

            # Hit
            if prediction == row[ground_truth_column]:
                num_hits += 1
                score += num_hits / (index+1)

        elif row[query_column] == row[minor_column]:
            raise ValueError('Cannot handle tie.')

    return _build_quotient(score, num_hits)


def _build_quotient(dividend, denominator):
    if denominator != 0:
        return dividend / denominator

    result = 0
    LOGGER.warning('Denominator = 0. Skip metric calculation and set it to value="%s".', result)
    return result
