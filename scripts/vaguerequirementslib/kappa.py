# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import logging

import pandas as pd
import numpy as np
LOGGER = logging.getLogger(__name__)

# pylint:disable=invalid-name

def calculate_fleiss_kappa(confusion_matrix: pd.DataFrame) -> float:
    """
    Calculate Fleiss' Kappa

    Args:
        confusion_matrix (pd.DataFrame): The confusion matrix

    Returns:
        float: The Fleiss' Kappa
    """
    raters_count = None  # Labeler

    raters_counts = confusion_matrix.sum(axis=1, numeric_only=True)
    assert len(set(raters_counts)) == 1, 'The raters count is inconsistent'
    raters_count = int(raters_counts[0])
    LOGGER.debug('Each requirement was labeled by %s workers.', raters_count)

    # Calc P mean
    # Omit requirement column
    numeric_df = confusion_matrix.select_dtypes(include=[np.number])
    # LOGGER.debug(f'Numeric df {numeric_df}')

    P_is = []
    for _, row in numeric_df.iterrows():
        P_i = 1 / (raters_count*(raters_count-1)) * (row.map(lambda category_count: category_count**2).sum() - raters_count)
        P_is.append(P_i)

    P = 1/confusion_matrix.shape[0] * np.sum(P_is)
    P_E = confusion_matrix.sum(axis=0, numeric_only=True).map(lambda overall_category_count: (overall_category_count/(raters_count*confusion_matrix.shape[0]))**2).sum()
    kappa = (P-P_E) / (1-P_E)

    return kappa


def calculate_free_marginal_kappa(confusion_matrix: pd.DataFrame) -> float:
    """
    Calculate the free marginal kappa
    Ref: https://eric.ed.gov/?id=ED490661

    Args:
        confusion_matrix (pd.DataFrame): The confusion matrix

    Returns:
        float: The kappa
    """

    raters_count = None  # Labeler

    raters_counts = confusion_matrix.sum(axis=1, numeric_only=True)
    assert len(set(raters_counts)) == 1, 'The raters count is inconsistent'
    raters_count = int(raters_counts[0])
    LOGGER.debug('Each requirement was labeled by %s workers.', raters_count)

    # Calc P mean
    # Omit requirement column
    numeric_df = confusion_matrix.select_dtypes(include=[np.number])
    # LOGGER.debug(f'Numeric df {numeric_df}')

    P_is = []
    for _, row in numeric_df.iterrows():
        P_i = 1 / (raters_count*(raters_count-1)) * (row.map(lambda category_count: category_count**2).sum() - raters_count)
        P_is.append(P_i)

    P = 1/confusion_matrix.shape[0] * np.sum(P_is)

    # This is the only difference to Fleiss' Kappa
    P_E = 1 / numeric_df.shape[1]
    kappa = (P-P_E) / (1-P_E)
    return kappa
