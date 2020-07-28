# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import logging
import sys
import csv
import pandas as pd

from vaguerequirementslib.constants import MTURK_ANSWER_COLUMN, MTURK_NOT_VAGUE_ANSWER_LABELS, MTURK_VAGUE_ANSWER_LABELS, MTURK_REQUIREMENT_COLUMN
from vaguerequirementslib.read_csv import read_csv_files

logging.basicConfig(
    format='%(asctime)s [%(name)-12.12s] [%(levelname)-5.5s]  %(message)s',
    stream=sys.stdout,
    level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def main():
    batch_names = [
        # '../../../Desktop/Masters_Thesis/datasets/corpus/labeled/corpus-batch-0-mturk.csv',
        # '../../../Desktop/Masters_Thesis/datasets/corpus/labeled/corpus-batch-1-mturk.csv',
        # '../../../Desktop/Masters_Thesis/datasets/corpus/labeled/corpus-batch-2-mturk.csv',
        # '../../../Desktop/Masters_Thesis/datasets/corpus/labeled/corpus-batch-3-mturk.csv',
        # '../../../Desktop/Masters_Thesis/datasets/corpus/labeled/corpus-batch-4-mturk.csv',
        '../../../Desktop/Masters_Thesis/datasets/corpus/labeled/corpus-batch-5-27-mturk.csv'
    ]

    out_file = '../../../Desktop/Masters_Thesis/datasets/corpus/labeled/corpus-batch-5-27-mturk-ties.csv'
    df = read_csv_files(batch_names)

    df_with_ties = _build_tie_df(df)

    # Uncomment the following line to print the confusion matrix
    df_with_ties.to_csv(out_file, sep=',', index=False, quoting=csv.QUOTE_NONNUMERIC)


def _build_tie_df(
        data_frame: pd.DataFrame,
        requirement_column: str = MTURK_REQUIREMENT_COLUMN,
        answer_column: str = MTURK_ANSWER_COLUMN,
        vague_answer_labels: list = MTURK_VAGUE_ANSWER_LABELS,
        not_vague_answer_labels: list = MTURK_NOT_VAGUE_ANSWER_LABELS) -> pd.DataFrame:
    """
    Build a data frame that contains all ties.
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

    LOGGER.info('Build requirement tie matrix.')
    requirements_to_label_map = {}

    # Group data by requirement.
    grouped_data = data_frame.groupby(requirement_column)

    for requirement, group in grouped_data:

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
                LOGGER.warning('Found unknown answer="%s" for requirement="%s".', value, requirement)

        # Build map
        if requirement not in requirements_to_label_map:
            requirements_to_label_map[requirement] = {'vague_count': 0, 'not_vague_count': 0}

        requirements_to_label_map[requirement]['vague_count'] += vague_count
        requirements_to_label_map[requirement]['not_vague_count'] += not_vague_count

    # Build data frame
    req_list = [
        [requirement, None]
        for requirement, counts in requirements_to_label_map.items()
        if counts['vague_count'] == counts['not_vague_count']
    ]
    LOGGER.info('Found %s requirements with ties in votes.', len(req_list))

    result = pd.DataFrame(req_list, columns=[MTURK_REQUIREMENT_COLUMN, MTURK_ANSWER_COLUMN])

    LOGGER.info('Built data frame containing only ties with %s of %s requirements. ', result.shape[0], len(requirements_to_label_map.items()))

    return result


if __name__ == '__main__':
    main()
