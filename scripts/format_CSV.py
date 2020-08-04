# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import logging
import sys
from glob import iglob
import json
from bisect import bisect
import csv

import numpy as np
import pandas as pd

from vaguerequirementslib import read_csv_file, read_csv_files, build_confusion_matrix

logging.basicConfig(
    format='%(asctime)s [%(name)-20.20s] [%(levelname)-5.5s]  %(message)s',
    stream=sys.stdout,
    level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

#pylint: disable=too-many-locals
def main():
    file_glob = '/Users/leohanisch/Desktop/Masters_Thesis/datasets/corpus/labeled/expert/*.csv'
    best_n = None
    metric = 'recall'

    LOGGER.info('Search all evaluation results matching the glob pattern="%s"and return the %s results.', file_glob, best_n)

    evaluation_files_iterator = iglob(file_glob, recursive=True)

    for file_path in evaluation_files_iterator:

        df = read_csv_file(file_path, separator=';')
        df.to_csv(file_path, sep=',', index=False, quoting=csv.QUOTE_NONNUMERIC)
        LOGGER.info(file_path)

    # df = read_csv_files(evaluation_files_iterator)

    build_confusion_matrix(df)
if __name__ == '__main__':
    main()
