# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import logging
import sys
from glob import iglob
import csv
from collections import Counter

import numpy as np
import pandas as pd
from nltk import word_tokenize
from vaguerequirementslib import read_csv_files_iterator

logging.basicConfig(
    format='%(asctime)s [%(name)-20.20s] [%(levelname)-5.5s]  %(message)s',
    stream=sys.stdout,
    level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

#pylint: disable=too-many-locals


def main():
    file_glob = '/Users/leohanisch/Desktop/Masters_Thesis/datasets/corpus/train_data.csv'

    LOGGER.info('Count all words of files matching the glob pattern="%s".', file_glob)

    files_iterator = iglob(file_glob, recursive=True)

    counter = Counter()
    vague_counter = Counter()
    not_vague_counter = Counter()
    for df in read_csv_files_iterator(files_iterator):
        for _, row in df.iterrows():
            tokens = word_tokenize(row.requirement)
            counter.update(tokens)
            if row.majority_label == 1:
                vague_counter.update(tokens)
            elif row.majority_label == 0:
                not_vague_counter.update(tokens)
            else:
                raise ValueError(f'Cannot handle unknown majority_label="{row.majority_label}".')

    result = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    result = result.rename(columns={'index': 'word', 0: 'overall_count'})
    result['vague_count'] = 0
    result['not_vague_count'] = 0
    result['difference'] = 0

    for index, row in result.iterrows():
        vague_count = vague_counter.get(row.word, 0)
        not_vague_count = not_vague_counter.get(row.word, 0)
        result.loc[index, 'vague_count'] = vague_count
        result.loc[index, 'not_vague_count'] = not_vague_count
        result.loc[index, 'difference'] = np.abs(not_vague_count - vague_count)
    result.to_csv('word_count.csv', sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)
    LOGGER.info('\n%s', result.head())


if __name__ == '__main__':
    main()
