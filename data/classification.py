import numpy as np
from data.types import ClassificationType, Size


def _get_classification_filename(
    class_type: ClassificationType, class_size: Size
) -> str:
    return f"data/Classification/data.{class_type.value}.train.{class_size.value}.csv"


def get_classification_data(class_type: ClassificationType, class_size: Size):
    filename = _get_classification_filename(class_type, class_size)
    data = np.loadtxt(filename, delimiter=",", skiprows=1)

    X = data[:, 0:2]
    y_labels = data[:, 2].astype(int)
    classes = np.unique(y_labels)

    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    y_indices = np.array([class_to_index[c] for c in y_labels])

    y = np.eye(len(classes))[y_indices]

    return X, y
