from typing import List
import pandas as pd

from src.loader.IO.pickle_file_handler import PickleFileHandler
from ...datastructures.Gate import Gate
from .flow_sample_file import FlowSampleFile
from .flow_sample_base import FlowSampleBase
from pathlib import Path


class FlowSampleCache(FlowSampleBase):

    @staticmethod
    def create_from_file_sample(flowsample_file: FlowSampleFile):
        """
        converts a FlowSampleFile to a FlowSampleCache
        """
        flowsamplecache = FlowSampleCache()
        flowsamplecache.events = flowsample_file.get_events()
        flowsamplecache.gates = flowsample_file.get_gates()
        flowsamplecache.labels = flowsample_file.get_gate_labels()
        flowsamplecache.name = flowsample_file.get_sample_name()
        flowsamplecache.filename = flowsample_file.get_sample_file_name()
        flowsamplecache.convex_gates = flowsample_file.get_convex_gates()

        return flowsamplecache

    @staticmethod
    def load_from_file(path: Path):
        """
        loads a FlowSampleCache from a previsouly saved File
        """
        pklFilehandler = PickleFileHandler(path)
        flow_sample_cache = pklFilehandler.load_from_pickle_file()

        if not isinstance(flow_sample_cache, FlowSampleCache):
            raise TypeError("loaded pkl is not from type 'FlowSampleCache'")

        return flow_sample_cache

    def __init__(self) -> None:
        super().__init__()

    def save_to_file(self, path: Path):
        """
        saves a FlowSampleCache to a file
        """
        pklFilehandler = PickleFileHandler(path)
        pklFilehandler.save_obj_as_pickle(self)

    # Region [FlowSampleBase implementation]

    def get_events(self) -> pd.DataFrame:
        return self.events

    def get_gate_labels(self) -> pd.DataFrame:
        return self.labels

    def get_gates(self) -> List[Gate]:
        return self.gates

    def get_sample_name(self) -> str:
        return self.name

    def get_sample_file_name(self) -> str:
        return self.filename

    def get_convex_gates(self) -> List[Gate]:
        return self.convex_gates

    # endregion
