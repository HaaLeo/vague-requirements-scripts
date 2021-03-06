# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

from os import path
import logging
import sys
from typing import Iterator
import pandas as pd

from .constants import MTURK_LEGACY_ANSWER_COLUMN, MTURK_ANSWER_COLUMN

LOGGER = logging.getLogger(__name__)

#pylint: disable=invalid-name


def read_csv_file(file_name: str, separator: str = ',') -> pd.DataFrame:
    """
    Read a single CSV file in a data frame.

    Args:
        file_name (str): The CSV file path.
        separator (str, optional): The CSV column separator. Defaults to ','.

    Returns:
        pd.DataFrame: The read data frame.
    """
    if path.isabs(file_name):
        abs_file_path = file_name
    else:
        abs_file_path = path.join(path.abspath(path.dirname(sys.modules['__main__'].__file__)), file_name)
    df = pd.read_csv(abs_file_path, sep=separator)
    LOGGER.debug('Read file="%s" with %s rows.', file_name, df.shape[0])

    # Preprocessing
    df = df.rename(columns={MTURK_LEGACY_ANSWER_COLUMN: MTURK_ANSWER_COLUMN})  # Rename legacy column names
    return df


def read_csv_files_iterator(file_names: list, separator: str = ',') -> Iterator[pd.DataFrame]:
    """
    Read all csv files and return an iterator over the corresponding data frames.

    Args:
        file_names (list): A list of all file names.
        separator (str, optional): The CSV file column separator. Defaults to ','.

    Yields:
        Iterator[pd.DataFrame]: The iterator of the data frame.
    """
    for name in file_names:
        yield read_csv_file(name, separator=separator)


def read_csv_files(file_names: list, separator: str = ',') -> pd.DataFrame:
    """
    Read csv files into one single data frame.

    Args:
        file_names (list): A list of all file names.
        separator (str, optional): The CSV file column separator. Defaults to ','.

    Returns:
        pd.DataFrame: The data frame containing all CSV files.
    """
    result = pd.concat(read_csv_files_iterator(file_names, separator=separator), ignore_index=True)
    return result
