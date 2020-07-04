from .constants import TP, TN, FP, FN


def calc_all_metrics(**kwargs) -> dict:
    return {
        'accuracy': calc_accuracy(**kwargs),
        'precision': calc_precision(**kwargs),
        'recall': calc_recall(**kwargs),
        'specificity': calc_specificity(**kwargs),
        'false_negative_rate': calc_false_negative_rate(**kwargs),
        'false_positive_rate': calc_false_positive_rate(**kwargs),
        'f1_score': calc_f1_score(**kwargs)
    }


def calc_accuracy(**kwargs) -> float:
    dividend = kwargs[TP] + kwargs[TN]
    denominator = kwargs[TP] + kwargs[TN] + kwargs[FP] + kwargs[FN]
    return dividend / denominator


def calc_precision(**kwargs) -> float:
    dividend = kwargs[TP]
    denominator = kwargs[TP] + kwargs[FP]
    return dividend / denominator


def calc_recall(**kwargs) -> float:
    dividend = kwargs[TP]
    denominator = kwargs[TP] + kwargs[FN]
    return dividend / denominator


def calc_specificity(**kwargs) -> float:
    dividend = kwargs[TN]
    denominator = kwargs[FP] + kwargs[TN]
    return dividend / denominator


def calc_false_negative_rate(**kwargs) -> float:
    dividend = kwargs[FN]
    denominator = kwargs[TP] + kwargs[FN]
    return dividend / denominator


def calc_false_positive_rate(**kwargs) -> float:
    dividend = kwargs[FP]
    denominator = kwargs[FP] + kwargs[TN]
    return dividend / denominator


def calc_f1_score(**kwargs):
    precision = calc_precision(**kwargs)
    recall = calc_recall(**kwargs)

    dividend = 2 * precision * recall
    denominator = precision + recall
    return dividend / denominator
