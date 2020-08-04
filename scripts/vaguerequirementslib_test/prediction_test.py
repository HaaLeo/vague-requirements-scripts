# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import numpy as np
import pytest

from vaguerequirementslib import predict_with_threshold


@pytest.fixture
def probabilities():
    return np.array([[0.1, 0.9], [0.4, 0.5], [0.7, 0.3]])

@pytest.mark.parametrize('threshold,expected_result', [
    (0.5, [1, 1, 0]),
    (0.6, [1, 0, 0]),
    (0.91, [0, 0, 0])
])
def test_predict_correctly(probabilities, threshold, expected_result):
    result = predict_with_threshold(probabilities, threshold)

    assert result == expected_result
