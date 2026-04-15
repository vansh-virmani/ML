import numpy as np


def preprocess_input(input_data: dict, scaler, feature_names: list) -> np.ndarray:
    """
    Convert user input dict → scaled numpy array.

    Args:
        input_data:    dict of {feature_name: value}
        scaler:        fitted sklearn scaler
        feature_names: ordered list of feature names

    Returns:
        scaled 2D numpy array of shape (1, n_features)
    """
    try:
        data = [float(input_data[f]) for f in feature_names]
    except KeyError as e:
        raise ValueError(f"Missing feature in input: {e}")
    except ValueError as e:
        raise ValueError(f"Non-numeric input detected: {e}")

    arr = np.array(data).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    return arr_scaled
