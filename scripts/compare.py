# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import logging
import sys

import pandas as pd

from vaguerequirementslib.read_csv import read_csv_files
from vaguerequirementslib.confusion_matrix import build_confusion_matrix
from vaguerequirementslib.majority_label import calc_majority_label
from vaguerequirementslib.constants import CM_REQUIREMENT_COLUMN, MAJORITY_LABEL_COLUMN, CM_NOT_VAGUE_COUNT_COLUMN, CM_VAGUE_COUNT_COLUMN
from vaguerequirementslib.kappa import calculate_free_marginal_kappa, calculate_fleiss_kappa

logging.basicConfig(
    format='%(asctime)s [%(name)-20.20s] [%(levelname)-5.5s]  %(message)s',
    stream=sys.stdout,
    level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

# pylint:disable=invalid-name

def main():
    first_batch_files = [
        '../../../Desktop/Masters_Thesis/datasets/vague_requirements/2020_05_19_Batch.csv',  # Vague Requirements 2 Assignments per HIT including 'cannot decide' option. Needs to handle ties
        '../../../Desktop/Masters_Thesis/datasets/vague_requirements/2020_05_05_Batch.csv',  # Vague Requirements 3 Assignments per HIT
        '../../../Desktop/Masters_Thesis/datasets/vague_requirements/Batch_4012720_batch_results.csv',  # Vague Requirements 2 Assignments per HIT. Needs to handle ties
        '../../../Desktop/Masters_Thesis/datasets/Batch_3996415_batch_results.csv',  # Vague words 3 Assignments per HIT
        '../../../Desktop/Masters_Thesis/datasets/102Requirements_expert_LH.csv',  # My Labels
        # '../../../Desktop/Masters_Thesis/datasets/2020_06_17_Batch_Leo_star.csv',  # Vague Requirements 2 Assignments per HIT altered by LH.
        # '../../../Desktop/Masters_Thesis/datasets/102Requirements_expert_SE.csv',  # Basti's Labels
    ]
    second_batch_files = [
        #  '../../../Desktop/Masters_Thesis/datasets/vague_requirements/2020_05_19_Batch.csv',  # Vague Requirements 2 Assignments per HIT including 'cannot decide' option. Needs to handle ties
        # '../../../Desktop/Masters_Thesis/datasets/vague_requirements/2020_05_05_Batch.csv',  # Vague Requirements 3 Assignments per HIT
        # '../../../Desktop/Masters_Thesis/datasets/vague_requirements/Batch_4012720_batch_results.csv',  # Vague Requirements 2 Assignments per HIT. Needs to handle ties
        # '../../../Desktop/Masters_Thesis/datasets/Batch_3996415_batch_results.csv',  # Vague words 3 Assignments per HIT
        # '../../../Desktop/Masters_Thesis/datasets/102Requirements_expert_LH.csv',  # My Labels
        # '../../../Desktop/Masters_Thesis/datasets/2020_06_17_Batch_Leo_star.csv',  # Vague Requirements 2 Assignments per HIT altered by LH.
        '../../../Desktop/Masters_Thesis/datasets/102Requirements_expert_SE.csv',  # Basti's Labels
    ]

    separator = ','
    drop_ties = True

    LOGGER.info('Compare the labels of two batches containing the same requirements.')
    LOGGER.info('Preprocess first data frame.')
    first_frame = _preprocess(first_batch_files, separator, drop_ties)

    LOGGER.info('Preprocess second data frame.')
    second_frame = _preprocess(second_batch_files, separator, drop_ties)

    frames = [first_frame, second_frame]
    # Initially concatenate all dataframes into one
    df = pd.concat(frames, ignore_index=True)

    # If ties were dropped earlier. Drop requirements that were not rated within all frames
    filtered_df = pd.concat(g for _, g in df.groupby(CM_REQUIREMENT_COLUMN) if len(g) == len(frames))

    confusion_matrix = build_confusion_matrix(filtered_df, CM_REQUIREMENT_COLUMN, MAJORITY_LABEL_COLUMN, [1], [0])

    LOGGER.info('Start comparison of the two data frames.')
    _calc_percentage(confusion_matrix)
    _calc_kappa(confusion_matrix)


def _calc_kappa(confusion_matrix: pd.DataFrame) -> None:
    """
    Calculate the kappas using the majority label of each frame as rater.

    Args:
        *frames (Iterable[pd.DataFrame]): The data frames to consider as raters.
    """
    # import csv
    # confusion_matrix.to_csv('./confusion_matrix.csv', sep=',', index=False, quoting=csv.QUOTE_NONNUMERIC)

    fleiss_kappa = calculate_fleiss_kappa(confusion_matrix)
    LOGGER.info('Calculated Fleiss\' kappa = %s.', fleiss_kappa)

    free_kappa = calculate_free_marginal_kappa(confusion_matrix)
    LOGGER.info('Calculated free kappa k_free = %s.', free_kappa)


def _calc_percentage(confusion_matrix: pd.DataFrame) -> None:
    """
    Calculate the percentage of unequally labeled requirements of the confusion matrix.
    The confusion matrix must be generated from two frames

    Args:
        first_frame (pd.DataFrame): The first data frame.
        second_frame (pd.DataFrame): The second data frame.
    """
    overall_count = 0
    equal_label_count = 0
    unequal_label_count = 0
    # Iterate rows of the evaluated data frame
    for (_, row) in confusion_matrix.iterrows():

        not_vague_count = row[CM_NOT_VAGUE_COUNT_COLUMN]
        vague_count = row[CM_VAGUE_COUNT_COLUMN]

        if not_vague_count == 0 or vague_count == 0:
            equal_label_count += 1
        else:
            unequal_label_count += 1
        overall_count += 1

    LOGGER.info('Overall requirements="%s". Equal label="%s". Unequal label="%s".', overall_count, equal_label_count, unequal_label_count)
    LOGGER.info('Percentage of unequally labeled requirements="%s".' % (unequal_label_count/overall_count)*100)


def _preprocess(files_list: list, separator: str, drop_ties: bool) -> pd.DataFrame:
    """
    Calculate the majority label for the given source file list

    Args:
        files_list (list): The CSV files to calculate the majority label for
        separator (str): The CSV separator
        drop_ties (bool): If there is a tie in votes (e.g.: One votes for vague one for not vague) then drop this entry from the confusion matrix.

    Returns:
        pd.DataFrame: The dataframe containing the majority label.
    """
    df = read_csv_files(files_list, separator)
    confusion_matrix = build_confusion_matrix(df, drop_ties=drop_ties)
    return calc_majority_label(confusion_matrix)


if __name__ == '__main__':
    main()
