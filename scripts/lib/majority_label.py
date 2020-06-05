import logging

import pandas as pd

from lib.constants import CM_NOT_VAGUE_COUNT_COLUMN, CM_VAGUE_COUNT_COLUMN, VAGUE_LABEL, NOT_VAGUE_LABEL, MAJORITY_LABEL_COLUMN

LOGGER = logging.getLogger(__name__)


def calc_majority_label(confusion_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the majority label based on the vote for "vague" and "not vague".

    Args:
        confusion_matrix (pd.DataFrame): The data frame containing information how often one voted for "vague" and "not vague"

    Returns:
        pd.DataFrame: The data frame for indicating whether a requirement is vague (1) or not (0)
    """
    df = confusion_matrix.copy()
    df[MAJORITY_LABEL_COLUMN] = df.apply(_get_majority_label, axis=1)
    label_counts = df[MAJORITY_LABEL_COLUMN].value_counts()
    LOGGER.info(f'"vague" majority label count = {label_counts[VAGUE_LABEL]}. "not vague" majority label count = {label_counts[NOT_VAGUE_LABEL]}.')
    return df


def _get_majority_label(row: pd.Series) -> int:
    # If it is a tie consider it not vague
    if row[CM_NOT_VAGUE_COUNT_COLUMN] <= row[CM_VAGUE_COUNT_COLUMN]:
        return VAGUE_LABEL
    else:
        return NOT_VAGUE_LABEL
