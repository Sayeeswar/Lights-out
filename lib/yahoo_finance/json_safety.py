"""
Convert yfinance / pandas / NumPy objects into values that can safely be
passed through json.dumps().
"""

import json
import math
from datetime import date, datetime

import numpy as np
import pandas as pd


def clean_for_json(obj):
    """
    Convert yfinance / pandas / NumPy objects into values
    that can safely be passed through json.dumps().
    """

    if obj is None:
        return None

    if isinstance(obj, (str, int, bool)):
        return obj

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        df.index = [clean_for_json(index) for index in df.index]
        df.columns = [clean_for_json(column) for column in df.columns]

        result = {}
        for index, row in df.iterrows():
            index_key = str(index)
            result[index_key] = {
                str(column): clean_for_json(value)
                for column, value in row.items()
            }
        return result

    if isinstance(obj, pd.Series):
        result = {}
        for index, value in obj.items():
            index_key = str(clean_for_json(index))
            result[index_key] = clean_for_json(value)
        return result

    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            key = str(clean_for_json(key))
            result[key] = clean_for_json(value)
        return result

    if isinstance(obj, (list, tuple, set)):
        return [clean_for_json(value) for value in obj]

    return str(obj)


def make_json_safe(data):
    """
    Final validation layer. Ensures the returned object can
    actually be serialized with json.dumps().
    """
    cleaned = clean_for_json(data)
    json.dumps(cleaned, ensure_ascii=False)  # raises early if anything slipped through
    return cleaned
