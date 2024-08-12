RAW_DATA_PATH = 'data/raw/insurance_dataset.csv'
PROCESSED_DATA_PATH = 'data/processed/insurance_dataset.csv'

LR_MODEL_PATH = 'models/lr.pkl'

NUMERICAL_FEATURES = [
    'age',
    'bmi',
    'children'
    ]

CATEGORICAL_FEATURES = [
    #'gender',
    'smoker',
    'region',
    'medical_history',
    'family_medical_history',
    'exercise_frequency',
    'occupation',
    'coverage_level'
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = 'charges'
