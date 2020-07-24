# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import logging
import sys
import csv


from vaguerequirementslib.read_csv import read_csv_files
from vaguerequirementslib.confusion_matrix import build_confusion_matrix
from vaguerequirementslib.majority_label import calc_majority_label

logging.basicConfig(
    format='%(asctime)s [%(name)-12.12s] [%(levelname)-5.5s]  %(message)s',
    stream=sys.stdout,
    level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def main():
    batch_names = [
        '../../../Desktop/Masters_Thesis/datasets/Batch_3996415_batch_results.csv',  # Vague words 3 Assignments per HIT
        # '../../../Desktop/Masters_Thesis/datasets/102Requirements_expert_LH.csv',  # My Labels
        # '../../../Desktop/Masters_Thesis/datasets/102Requirements_expert_SE.csv',  # Basti's Labels
    ]

    all_df = read_csv_files(batch_names)

    confusion_matrix = build_confusion_matrix(all_df)

    # Uncomment the following line to print the confusion matrix
    # confusion_matrix.to_csv('./confusion_matrix.csv', sep=',', index=False, quoting=csv.QUOTE_NONNUMERIC)

    majority_label_df = calc_majority_label(confusion_matrix)
    LOGGER.info('Calculated majority labels')
    majority_label_df.to_csv('./majority_label.csv', sep=',', index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    main()
