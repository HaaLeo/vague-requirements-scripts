# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import logging
from typing import List

import numpy as np

LOGGER = logging.getLogger(__name__)


def predict_with_threshold(probabilities, vague_threshold=0.5) -> List[int]:
    """
    Classify the requirements as vague or not with a custom threshold.
    The first entry of a probability is considered the _not vague_ and the second entry the _vague_ probability
    example probabilities:
       [[0.6, 0.4],
        [0.3, 0.7],
        [0.9, 0.1]]

    Args:
        probabilities: The classification probabilities
        vague_threshold (int): The threshold to count as vague requirement (default 0.5)

    Returns:
        List[int]: The classification 0 if not vague 1 if vague
    """

    result = [1 if probability[1] >= vague_threshold else 0 for probability in probabilities]
    return result
