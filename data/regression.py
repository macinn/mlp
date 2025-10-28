import numpy as np
from data.types import RegressionType, Size


def _get_regression_filename(
    class_type: RegressionType, class_size: Size, test: bool = False
) -> str:
    suffix = "test" if test else "train"
    return f"data/Regression/data.{class_type.value}.{suffix}.{class_size.value}.csv"


def get_regression_data(
    class_type: RegressionType, class_size: Size, test: bool = False
):
    filename = _get_regression_filename(class_type, class_size, test)
    data = np.loadtxt(filename, delimiter=",", skiprows=1)

    X = data[:, 0:1]
    y = data[:, 1:2]

    return X, y
