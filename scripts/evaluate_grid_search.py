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

import pandas as pd

logging.basicConfig(
    format='%(asctime)s [%(name)-20.20s] [%(levelname)-5.5s]  %(message)s',
    stream=sys.stdout,
    level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

#pylint: disable=too-many-locals
def main():
    file_glob = '/Users/leohanisch/Desktop/Masters_Thesis/runs/grid-search/1/**/*evaluation.json'
    best_n = None
    metric = 'recall'

    LOGGER.info('Search all evaluation results matching the glob pattern="%s"and return the %s results.', file_glob, best_n)

    evaluation_files_iterator = iglob(file_glob, recursive=True)

    best_metrics = []  # List of the best entries regarding a specific metric
    best_entries = {}  # The best entries regarding a specific metric. Include all the hyperparameter.

    for file_path in evaluation_files_iterator:
        with open(file_path, 'r') as eval_file:
            current_result = json.load(eval_file)

        # The metric value with respect to the test set after the last trained fold.
        metrics_dict = current_result['fold_results'][-1]['metrics']['test']['vague']
        metric_value = metrics_dict[metric]
        insertion_index = bisect(best_metrics, metric_value)

        # The used hyperparameter
        hyperparameter_dict = current_result['hyperparameter']
        hyperparameter_dict.pop('ngram_range', None)
        hyperparameter_dict.pop('max_features', None)

        merged = {
            **metrics_dict,
            **hyperparameter_dict,
            'resampling_strategy': current_result['data_set']['resampling_strategy']
        }

        if not 'kfold_splits' in merged:
            merged['kfold_splits'] = len(current_result['fold_results'])

        if not best_entries:
            best_entries = {key: [] for key in merged}

        best_metrics.insert(insertion_index, metric_value)
        for key, value in merged.items():
            best_entries[key].insert(insertion_index, value)

        # Trim result if required
        if best_n and len(best_metrics) > best_n:
            best_metrics.pop(0)
            for key in best_entries.keys():
                best_entries[key].pop(0)

    df = pd.DataFrame.from_dict(best_entries) #pylint:disable=invalid-name
    df.to_csv('grid-search-evaluation.csv', sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)
    LOGGER.info('\n%s', df.head())


# class ComparableDict(dict):
#     def __init__(self, *args, **kwargs):
#         self.__key_to_compare = kwargs.pop('key_to_compare')
#         super(ComparableDict, self).__init__(*args, **kwargs)

#     def __lt__(self, other):
#         return self.__getitem__(self.__key_to_compare) < other[self.__key_to_compare]

if __name__ == '__main__':
    main()
