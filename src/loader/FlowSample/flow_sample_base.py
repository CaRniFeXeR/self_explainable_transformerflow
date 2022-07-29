from abc import ABC, abstractmethod
from typing import List
import pandas as pd

from ...datastructures.Gate import Gate


class FlowSampleBase(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_sample_name(self) -> str:
        pass

    @abstractmethod
    def get_sample_file_name(self) -> str:
        pass

    @abstractmethod
    def get_events(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_gates(self) -> List[Gate]:
        pass

    @abstractmethod
    def get_convex_gates(self) -> List[Gate]:
        pass

    @abstractmethod
    def get_gate_labels(self) -> pd.DataFrame:
        pass  