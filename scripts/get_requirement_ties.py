import logging
from typing import Iterator
import sys
import csv

import pandas as pd

from vaguerequirementslib.read_csv import read_csv_files
from vaguerequirementslib.confusion_matrix import build_confusion_matrix
from vaguerequirementslib.kappa import calculate_fleiss_kappa, calculate_free_marginal_kappa

logging.basicConfig(
    format='%(asctime)s [%(name)-12.12s] [%(levelname)-5.5s]  %(message)s',
    stream=sys.stdout,
    level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def main():
    batch_names = [
        '../../../Desktop/Masters_Thesis/datasets/datasets/corpus/labeled/corpus-batch-0-mturk.csv',
        '../../../Desktop/Masters_Thesis/datasets/datasets/corpus/labeled/corpus-batch-1-mturk.csv',
        '../../../Desktop/Masters_Thesis/datasets/datasets/corpus/labeled/corpus-batch-2-mturk.csv',
        '../../../Desktop/Masters_Thesis/datasets/datasets/corpus/labeled/corpus-batch-3-mturk.csv',
        '../../../Desktop/Masters_Thesis/datasets/datasets/corpus/labeled/corpus-batch-4-mturk.csv',
        '../../../Desktop/Masters_Thesis/datasets/datasets/corpus/labeled/corpus-batch-5-27-mturk.csv'
    ]

    out_file = '../../../Desktop/Masters_Thesis/datasets/datasets/corpus/labeled/corpus-batch-0-27-mturk-ties.csv'
    df = read_csv_files(batch_names)

    confusion_matrix = build_confusion_matrix(df)
    # Wikipedia test data
    # confusion_matrix = pd.DataFrame([
    #     [0, 0, 0, 0, 14],
    #     [0, 2, 6, 4, 2],
    #     [0, 0, 3, 5, 6],
    #     [0, 3, 9, 2, 0],
    #     [2, 2, 8, 1, 1],
    #     [7, 7, 0, 0, 0],
    #     [3, 2, 6, 3, 0],
    #     [2, 5, 3, 2, 2],
    #     [6, 5, 2, 1, 0],
    #     [0, 2, 2, 3, 7]
    # ])
    # Uncomment the following line to print the confusion matrix
    # confusion_matrix.to_csv('./confusion_matrix.csv', sep=',', index=False, quoting=csv.QUOTE_NONNUMERIC)

    fleiss_kappa = calculate_fleiss_kappa(confusion_matrix)
    free_kappa = calculate_free_marginal_kappa(confusion_matrix)

    LOGGER.info(f'Calculated Fleiss\' kappa = {fleiss_kappa}.')
    LOGGER.info(f'Calculated free kappa k_free = {free_kappa}.')


if __name__ == '__main__':
    main()
