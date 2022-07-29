import random
from typing import List
import torch
from torch.utils.data import IterableDataset

from ..loader.FlowSample.flowsample_tensor_data import FlowSampleTensorData
from ..datastructures.ProcessedFlowSample import ProcessedFlowSample
from ..datastructures.configs.dataproviderconfig import DataProviderConfig
from .DeviceManager import DeviceManager
from ..datastructures.SampleRetrieveOptions import SampleRetrieveOptions
from ..loader.DataSet.BaseDataSet import BaseDataSet
from copy import deepcopy


class DataProvider(IterableDataset):
    """
    Handles loading, caching and processing the next samples while training. Can be used to parallelise data loading.
    """

    def __init__(self, dataset: BaseDataSet, device_manager: DeviceManager, dataprovider_config: DataProviderConfig,  retrieve_config: SampleRetrieveOptions) -> None:
        super(DataProvider).__init__()
        self.dataset = dataset
        self.dataset_iter = iter(self.dataset)
        self.retrieve_config = retrieve_config
        self.config = dataprovider_config
        self.device_manager = device_manager
        self.sample_count = 0
        self.sample_cache: List[ProcessedFlowSample] = []

    def _load_samples(self):
        """
        loads n samples from the disk and process them
        """

        self.sample_cache = []

        for i in range(self.config.n_samples_to_load):
            try:
                sample = next(self.dataset_iter)
                self.sample_count += 1
                proceeded_sample = ProcessedFlowSample(sample, self.retrieve_config)
                self.sample_cache.append(proceeded_sample)
            except StopIteration as stopiterSignal:
                print(f"data provider reached end of dataset after  {self.sample_count} samples")
                break
            except Exception as ex:
                print(f"error for sample {sample.get_sample_file_name()}")
                print(ex)

    def move_sample_to_gpu(self, tensor_data: FlowSampleTensorData) -> FlowSampleTensorData:
        """
        moves the tensor data on the gpu
        """

        tensor_data.events = self.device_manager.move_to_gpu(tensor_data.events)
        tensor_data.padding_mask = self.device_manager.move_to_gpu(tensor_data.padding_mask)
        tensor_data.polygons = self.device_manager.move_to_gpu(tensor_data.polygons)

        return tensor_data

    def provide_next_sample(self) -> FlowSampleTensorData:
        """
        moves next sample to the GPU and returns it
        """
        if len(self.sample_cache) == 0:
            self._load_samples()
            if len(self.sample_cache) == 0:
                raise StopIteration()

        return self.sample_cache.pop().to_tensors()

    # region iter implementation
    def __iter__(self):
        """
        returns a copied version of itself with shuffled dataset in order to not dublicate behaviour in parallelised training
        """
        workerinfo = torch.utils.data.get_worker_info()
        if workerinfo is not None:
            random.seed(workerinfo.seed)
            print("set random seed to: " + str(workerinfo.seed))
        copied_self = deepcopy(self)
        copied_self.dataset.shuffle()
        copied_self.dataset_iter = iter(copied_self.dataset)
        copied_self.sample_cache = []
        copied_self.sample_count = 0
        return copied_self

    def __next__(self) -> List[FlowSampleTensorData]:

        return self.provide_next_sample()

    @staticmethod
    def collate_fn(data_batch: List[FlowSampleTensorData]) -> FlowSampleTensorData:
        """
        concatenates tensor batch data
        """
        result = data_batch[0]

        if len(data_batch) >= 1:
            for i in range(1, len(data_batch)):
                result.concat_data(data_batch[i])

        return result

    # endregion
