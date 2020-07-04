# Confusion matrix related constants
CM_VAGUE_COUNT_COLUMN = 'vague_count'
CM_NOT_VAGUE_COUNT_COLUMN = 'not_vague_count'
CM_REQUIREMENT_COLUMN = 'requirement'

# MTurk related constants
MTURK_REQUIREMENT_COLUMN = 'Input.requirement'  # The requirement column name of MTurk's batch result
MTURK_LEGACY_ANSWER_COLUMN = 'Answer.vague-words.label'
MTURK_ANSWER_COLUMN = 'Answer.vague-requirement.label'
MTURK_VAGUE_ANSWER_LABELS = ['1 - Yes, it is vague', '1 - Yes, contains vague words', '3 - Cannot decide']
MTURK_NOT_VAGUE_ANSWER_LABELS = ['2 - No, it is not vague', '2 - No, contains no vague words']

MAJORITY_LABEL_COLUMN = 'majority_label'

VAGUE_LABEL = 1
NOT_VAGUE_LABEL = 0

TP = 'true_positive'
TN = 'true_negative'
FP = 'false_positive'
FN = 'false_negative'
