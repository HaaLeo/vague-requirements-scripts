from os import path
import inspect
import logging
import sys
from typing import Iterable

import pandas as pd

from lib.read_csv import read_csv_files
from lib.confusion_matrix import build_confusion_matrix
from lib.majority_label import calc_majority_label
from lib.constants import CM_REQUIREMENT_COLUMN, MAJORITY_LABEL_COLUMN
from lib.kappa import calculate_free_marginal_kappa, calculate_fleiss_kappa

logging.basicConfig(
    format='%(asctime)s [%(name)-20.20s] [%(levelname)-5.5s]  %(message)s',
    stream=sys.stdout,
    level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def main():
    first_batch_files = [
        '../../../Desktop/Masters_Thesis/datasets/vague_requirements/2020_05_19_Batch.csv',  # Vague Requirements 2 Assignments per HIT including 'cannot decide' option. Needs to handle ties
        '../../../Desktop/Masters_Thesis/datasets/vague_requirements/2020_05_05_Batch.csv',  # Vague Requirements 3 Assignments per HIT
        '../../../Desktop/Masters_Thesis/datasets/vague_requirements/Batch_4012720_batch_results.csv',  # Vague Requirements 2 Assignments per HIT. Needs to handle ties
        '../../../Desktop/Masters_Thesis/datasets/Batch_3996415_batch_results.csv',  # Vague words 3 Assignments per HIT
        '../../../Desktop/Masters_Thesis/datasets/102Requirements_expert_LH.csv',  # My Labels
        # '../../../Desktop/Masters_Thesis/datasets/102Requirements_expert_SE.csv',  # Basti's Labels
    ]
    second_batch_files = [
        #  '../../../Desktop/Masters_Thesis/datasets/vague_requirements/2020_05_19_Batch.csv',  # Vague Requirements 2 Assignments per HIT including 'cannot decide' option. Needs to handle ties
        # '../../../Desktop/Masters_Thesis/datasets/vague_requirements/2020_05_05_Batch.csv',  # Vague Requirements 3 Assignments per HIT
        # '../../../Desktop/Masters_Thesis/datasets/vague_requirements/Batch_4012720_batch_results.csv',  # Vague Requirements 2 Assignments per HIT. Needs to handle ties
        # '../../../Desktop/Masters_Thesis/datasets/Batch_3996415_batch_results.csv',  # Vague words 3 Assignments per HIT
        # '../../../Desktop/Masters_Thesis/datasets/102Requirements_expert_LH.csv',  # My Labels
        '../../../Desktop/Masters_Thesis/datasets/102Requirements_expert_SE.csv',  # Basti's Labels
    ]

    separator = ','

    LOGGER.info('Compare the labels of two batches containing the same requirements.')
    LOGGER.info('Preprocess first data frame.')
    first_frame = _preprocess(first_batch_files, separator)

    LOGGER.info('Preprocess second data frame.')
    second_frame = _preprocess(second_batch_files, separator)

    LOGGER.info('Start comparison of the two data frames.')
    _calc_percentage(first_frame, second_frame)
    _calc_kappa(first_frame, second_frame)


def _calc_kappa(*frames: Iterable[pd.DataFrame]) -> None:
    """
    Calculate the kappas using the majority label of each frame as rater.

    Args:
        *frames (Iterable[pd.DataFrame]): The data frames to consider as raters.
    """
    df = pd.concat(frames, ignore_index=True)
    confusion_matrix = build_confusion_matrix(df, CM_REQUIREMENT_COLUMN, MAJORITY_LABEL_COLUMN, [1], [0])
    import csv
    confusion_matrix.to_csv('./confusion_matrix.csv', sep=',', index=False, quoting=csv.QUOTE_NONNUMERIC)

    fleiss_kappa = calculate_fleiss_kappa(confusion_matrix)
    free_kappa = calculate_free_marginal_kappa(confusion_matrix)

    LOGGER.info(f'Calculated Fleiss\' kappa = {fleiss_kappa}.')
    LOGGER.info(f'Calculated free kappa k_free = {free_kappa}.')


def _calc_percentage(first_frame: pd.DataFrame, second_frame: pd.DataFrame) -> None:
    """
    Calculate the percentage of unequally labeled requirements of the two data frames.

    Args:
        first_frame (pd.DataFrame): The first data frame.
        second_frame (pd.DataFrame): The second data frame.
    """
    overall_count = 0
    equal_label_count = 0
    unequal_label_count = 0
    # Iterate rows of the evaluated data frame
    for (_, first_frame_row), (_, second_frame_row) in zip(first_frame.iterrows(), second_frame.iterrows()):

        first_req = first_frame_row[CM_REQUIREMENT_COLUMN]
        second_req = second_frame_row[CM_REQUIREMENT_COLUMN]
        if first_req == second_req:
            first_label = first_frame_row[MAJORITY_LABEL_COLUMN]
            second_label = second_frame_row[MAJORITY_LABEL_COLUMN]

            if first_label == second_label:
                equal_label_count += 1
            else:
                unequal_label_count += 1
            overall_count += 1
        else:
            LOGGER.warn('The two batch files are not aligned correctly.')

    LOGGER.info('Overall requirements="%s". Equal label="%s". Unequal label="%s".', overall_count, equal_label_count, unequal_label_count)
    LOGGER.info(f'Percentage of unequally labeled requirements="{(unequal_label_count/overall_count)*100}%".')


def _preprocess(files_list: list, separator: str) -> pd.DataFrame:
    df = read_csv_files(files_list, separator)
    confusion_matrix = build_confusion_matrix(df)
    return calc_majority_label(confusion_matrix)


if __name__ == '__main__':
    main()
