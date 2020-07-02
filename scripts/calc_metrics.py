from os import path
import inspect
import logging
import sys

import pandas as pd

from vaguerequirementslib.read_csv import read_csv_file

logging.basicConfig(
    format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s',
    stream=sys.stdout,
    level=logging.INFO)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
CONTAINS_VAGUE_WORDS_LABEL = 1
NO_VAGUE_WORDS_LABEL = 0
CANNOT_DECIDE_LABEL = -1


def main():
    truth_file_path = '../../../Desktop/Masters_Thesis/datasets/102Requirements_truth.csv'
    labeld_data_file_path = f'../../../Desktop/Masters_Thesis/datasets/vague_requirements/2020_05_05_Batch_evaluated.csv'
    separator = ';'

    LOGGER.info(f'Red data')
    data_frame = read_csv_file(labeld_data_file_path, separator)
    truth_frame = read_csv_file(truth_file_path, separator)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    # Iterate rows of the evaluated data frame
    for index, row in data_frame.iterrows():

        # Get the ground truth for the current requirement
        truth_label = int(truth_frame.loc[truth_frame['requirement'] == row['requirement']].get(key='vague'))
        assigned_label = row['majority label']

        if assigned_label == CANNOT_DECIDE_LABEL:
            assigned_label = CONTAINS_VAGUE_WORDS_LABEL  # Treat undecided entries as vague

        # Calculate TP TN FP FN
        if assigned_label == CONTAINS_VAGUE_WORDS_LABEL:
            if assigned_label == truth_label:
                true_positive += 1
            else:
                false_positive += 1
        elif assigned_label == NO_VAGUE_WORDS_LABEL:
            if assigned_label == truth_label:
                true_negative += 1
            else:
                false_negative += 1
        else:
            raise ValueError(f'Detected unsupported label="{assigned_label}".')

    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = (true_negative + true_positive) / (true_positive + true_negative + false_positive + false_negative)
    metrics['precision'] = true_positive / (true_positive + false_positive)
    metrics['recall'] = true_positive / (true_positive + false_negative)
    metrics['specificity'] = true_negative / (false_positive + true_negative)
    metrics['false_negative_rate'] = false_negative / (true_positive + false_negative)
    metrics['false_positive_rate'] = false_positive / (false_positive + true_negative)

    metrics['true_positive'] = true_positive
    metrics['true_negative'] = true_negative
    metrics['false_positive'] = false_positive
    metrics['false_negative'] = false_negative
    metrics['f1_score'] = (2*metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])

    print_result(metrics, separator)


def print_result(metrics: dict, separator):
    LOGGER.info(f'True positives: {metrics["true_positive"]}.')
    LOGGER.info(f'True negatives: {metrics["true_negative"]}.')
    LOGGER.info(f'False positives: {metrics["false_positive"]}.')
    LOGGER.info(f'False negatives: {metrics["false_negative"]}.')

    LOGGER.info('Calculated metrics.')
    LOGGER.info(f'Accuracy: {metrics["accuracy"]}')
    LOGGER.info(f'Precision: {metrics["precision"]}')
    LOGGER.info(f'Recall: {metrics["recall"]}')
    LOGGER.info(f'Specificity: {metrics["specificity"]}')
    LOGGER.info(f'False negative rate: {metrics["false_negative_rate"]}')
    LOGGER.info(f'False positive rate: {metrics["false_positive_rate"]}')
    LOGGER.info(f'F1 score: {metrics["f1_score"]}')

    LOGGER.info(f'Print CSV style:\n\
"True positives:"{separator}{metrics["true_positive"]}\n\
"True negatives:"{separator}{metrics["true_negative"]}\n\
"False positives:"{separator}{metrics["false_positive"]}\n\
"False negatives:"{separator}{metrics["false_negative"]}\n\n\
"Accuracy:"{separator}{metrics["accuracy"]}\n\
"Precision:"{separator}{metrics["precision"]}\n\
"Recall:"{separator}{metrics["recall"]}\n\
"Specificity:"{separator}{metrics["specificity"]}\n\
"False negative rate:"{separator}{metrics["false_negative_rate"]}\n\
"False positive rate:"{separator}{metrics["false_positive_rate"]}\n\
"F1 score:"{separator}{metrics["f1_score"]}')

    LOGGER.info(f'Print Mardown table:\n\
|Metric|Value|\n\
|-|-|\n\
|True positives|{metrics["true_positive"]}|\n\
|True negatives|{metrics["true_negative"]}|\n\
|False positives|{metrics["false_positive"]}|\n\
|False negatives|{metrics["false_negative"]}|\n\
|-|-|\n\
|Accuracy|{metrics["accuracy"]}|\n\
|Precision|{metrics["precision"]}|\n\
|Recall|{metrics["recall"]}|\n\
|Specificity|{metrics["specificity"]}|\n\
|False negative rate|{metrics["false_negative_rate"]}|\n\
|False positive rate|{metrics["false_positive_rate"]}|\n\
|F1 score|{metrics["f1_score"]}|')


if __name__ == '__main__':
    main()
