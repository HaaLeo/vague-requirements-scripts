import logging
import sys
import csv

import pandas as pd
import numpy as np

from vaguerequirementslib.read_csv import read_csv_file

logging.basicConfig(
    format='%(asctime)s [%(name)-20.20s] [%(levelname)-5.5s]  %(message)s',
    stream=sys.stdout,
    level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def main():
    corpus_file = '../../../Desktop/Masters_Thesis/datasets/corpus/raw/requirement-corpus-all.csv'  # Corpus

    separator = ';'

    batch_size = 100
    LOGGER.info(f'Slice corpus in batches of size="{batch_size}".')

    LOGGER.info('Read corpus.')
    df = read_csv_file(corpus_file, separator=';').groupby('label').get_group('requirement')
    # The raw df contains duplicates of requirements
    df = df.drop_duplicates()
    for k, batch in df.groupby(np.arange(len(df))//batch_size):
        batch = batch.rename(columns={'sentence': 'requirement'})
        batch.to_csv(f'./corpus1/corpus-batch-{k}.csv', sep=separator, index=False, quoting=csv.QUOTE_NONNUMERIC, columns=['requirement'])


if __name__ == '__main__':
    main()
