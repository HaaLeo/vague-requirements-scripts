import pandas as pd
from vaguerequirementslib.metrics import calc_mean_average_precision

def test_calc_mean_average_precision_correctly():
    df = pd.DataFrame.from_dict({
        'vague_prob': [0.6, 0.2, 0.3, 0.9, 0.9],
        'not_vague_prob': [0.4, 0.8, 0.7, 0.1, 0.1],
        'majority_label': [1, 0, 1, 0, 1]
    })

    result = calc_mean_average_precision(df)
