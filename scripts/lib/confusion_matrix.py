import logging
from typing import Iterable

import pandas as pd

from lib.constants import *

LOGGER = logging.getLogger(__name__)


def build_confusion_matrix(
        data_frame: pd.DataFrame,
        requirement_column: str = MTURK_REQUIREMENT_COLUMN,
        answer_column: str = MTURK_ANSWER_COLUMN,
        vague_answer_labels: list = MTURK_VAGUE_ANSWER_LABELS,
        not_vague_answer_labels: list = MTURK_NOT_VAGUE_ANSWER_LABELS) -> pd.DataFrame:
    """
    Build a confusion matrix of the given data_frame.
    The defaults are set to handle an Amazon MTurk Batch Result data frame.

    Args:
        data_frame (pd.DataFrame): The data frame
        requirement_column (str, optional): The columns name that contains the requirements. Defaults to MTURK_REQUIREMENT_COLUMN.
        answer_column (str, optional): The columns name that contains the answers/labels. Defaults to MTURK_ANSWER_COLUMN.
        vague_answer_labels (list, optional): List of answers/labels that indicate a vague requirement. Defaults to MTURK_VAGUE_ANSWER_LABELS.
        not_vague_answer_labels (list, optional): List of answers/labels that indicate a non vague requirement. Defaults to MTURK_NOT_VAGUE_ANSWER_LABELS.

    Returns:
        pd.DataFrame: The confusion matrix
    """

    LOGGER.info('Build confusion matrix.')
    requirements_to_label_map = {}

    # Group data by requirement.
    grouped_data = data_frame.groupby(requirement_column)

    for requirement, group in grouped_data:
        # print(requirement)
        # print(group[MTURK_ANSWER_COLUMN])

        # Answered Is vague
        vague_count = 0
        # Answered Is not vague
        not_vague_count = 0

        # Loop all answers for this requirement
        for value in group[answer_column]:

            # Count score
            if value in vague_answer_labels:
                vague_count += 1
            elif value in not_vague_answer_labels:
                not_vague_count += 1
            else:
                LOGGER.warning(f'Found unknown answer="{value}" for requirement="{requirement}".')

        # Build map
        if requirement not in requirements_to_label_map:
            requirements_to_label_map[requirement] = {'vague_count': 0, 'not_vague_count': 0}

        requirements_to_label_map[requirement]['vague_count'] += vague_count
        requirements_to_label_map[requirement]['not_vague_count'] += not_vague_count

    # Build data frame
    req_list = [[requirement, counts['vague_count'], counts['not_vague_count']] for requirement, counts in requirements_to_label_map.items()]
    result = pd.DataFrame(req_list, columns=[CM_REQUIREMENT_COLUMN, CM_VAGUE_COUNT_COLUMN, CM_NOT_VAGUE_COUNT_COLUMN])

    sums = result.sum(axis=0, numeric_only=True)
    LOGGER.info(f'Overall "vague" count = {sums[CM_VAGUE_COUNT_COLUMN]}. Overall "not vague" count = {sums[CM_NOT_VAGUE_COUNT_COLUMN]}')

    return result
