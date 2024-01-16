from enum import Enum

from .infer_cls import TextClassifier
from .infer_det import TextDetector
from .infer_e2e import TextEnd2End
from .infer_rec import TextRecognizer

__all__ = [
    "TextDetector",
    "TextClassifier",
    "TextRecognizer",
    "TextEnd2End",
    "TaskType",
    "SUPPORTED_TASK_BASIC_MODULE",
]


class TaskType(Enum):
    DET = 0  # Detection Model
    CLS = 1  # Classification Model
    REC = 2  # Recognition Model
    DET_REC = 3  # Detection And Detection Model
    DET_CLS_REC = 4  # Detection, Classification and Recognition Model
    E2E = 5  # End2End Detection And Detection Model


SUPPORTED_TASK_BASIC_MODULE = {
    TaskType.DET: [TaskType.DET],
    TaskType.CLS: [TaskType.CLS],
    TaskType.REC: [TaskType.REC],
    TaskType.DET_REC: [TaskType.DET, TaskType.REC],
    TaskType.DET_CLS_REC: [TaskType.DET, TaskType.CLS, TaskType.REC],
    TaskType.E2E: [TaskType.E2E],
}
