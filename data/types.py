from enum import StrEnum, IntEnum


class ClassificationType(StrEnum):
    CIRCLES = "circles"
    NOISY_XOR = "noisyXOR"
    XOR = "XOR"
    SIMPLE = "simple"
    THREE_GAUSS = "three_gauss"


class RegressionType(StrEnum):
    MULTIMODAL = "multimodal"
    CUBE = "cube"
    LINEAR = "linear"
    ACTIVATION = "activation"
    SQUARE = "square"


class Size(IntEnum):
    _100 = 100
    _500 = 500
    _1000 = 1000
    _10000 = 10000
