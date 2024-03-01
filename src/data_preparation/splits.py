from enum import Enum


class Split(Enum):
    TRAIN: str = 'train'
    TEST: str = 'test'
    VALIDATION: str = 'validation'