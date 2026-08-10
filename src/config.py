from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_PATH = PROJECT_ROOT / "data"

SUBMISSION_PATH = DATA_PATH / "inference" / "submission.csv"

DATA_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "bureau": "bureau.csv",
    "prev_loans": "previous_loans.csv",
    "transactions": "transactions.csv",
}


LEAK_COLS = [
    "post_loan_collection_score",
    "days_until_first_overdue",
    "internal_decision_code",
]

DROP_COLS = ["region_coefficient"]

DUPLICATE_COLS = ["incoming_amount"]

USELESS_COLS = [
    "application_id",
    "client_id",
    "hash_id",
    "application_date",
    "previous_loan_id",
    "transaction_date",
]

# models settings
RANDOM_STATE = 42  # 5k-fold cross-validation
N_SPLITS = 5

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "eval_metric": "AUC",
    "random_seed": RANDOM_STATE,
    "verbose": 0,
}

LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

LOGREG_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

BLEND_WEIGHTS = {"catboost": 0.6, "lgbm": 0.4}
