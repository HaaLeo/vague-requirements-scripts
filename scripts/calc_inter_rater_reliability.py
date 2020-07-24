# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import logging
import sys

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
        # '../../../Desktop/Masters_Thesis/datasets/vague_requirements/2020_05_19_Batch.csv',  # Vague Requirements 2 Assignments per HIT including 'cannot decide' option. Needs to handle ties
        # '../../../Desktop/Masters_Thesis/datasets/vague_requirements/2020_05_05_Batch.csv',  # Vague Requirements 3 Assignments per HIT
        # '../../../Desktop/Masters_Thesis/datasets/vague_requirements/Batch_4012720_batch_results.csv',  # Vague Requirements 2 Assignments per HIT. Needs to handle ties
        # '../../../Desktop/Masters_Thesis/datasets/Batch_3996415_batch_results.csv',  # Vague words 3 Assignments per HIT
        '../../../Desktop/Masters_Thesis/datasets/102Requirements_expert_LH.csv',  # My Labels
        '../../../Desktop/Masters_Thesis/datasets/102Requirements_expert_SE.csv',  # Basti's Labels
    ]

    df = read_csv_files(batch_names) # pylint:disable=invalid-name

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

    LOGGER.info('Calculated Fleiss\' kappa = %s.', fleiss_kappa)
    LOGGER.info('Calculated free kappa k_free = %s.', free_kappa)


if __name__ == '__main__':
    main()
