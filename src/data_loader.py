# src/data_loader.py
import pandas as pd

from src.config import DATA_FILES, DATA_PATH


def load_data():
    data = {}
    for name, filename in DATA_FILES.items():
        data[name] = pd.read_csv(DATA_PATH / filename)
    return data
