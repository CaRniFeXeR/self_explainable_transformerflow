import traceback
from typing import List

from ..loader.FlowSample.flow_sample_file import FlowSampleFile
from ..utils.outlier_handler import OutlierHandler

from ..loader.DataSet.BaseDataSet import BaseDataSet
from ..loader.FlowSample.flow_sample_cache import FlowSampleCache
from ..loader.DataSet.FileTypeFolderDataSet import FileTypeFolderDataSet
from ..datastructures.configs.cachedatacreationconfig import CacheDataCreationConfig

import pandas as pd
from pathlib import Path


class CacheCreator:
    """
    Enables to create cached FlowSample files in order to speed up data loading & processing
    """

    def __init__(self, cachecreation_config: CacheDataCreationConfig) -> None:
        self.config = cachecreation_config
        self.__prepare_creator()

    def __prepare_creator(self):

        self.datasets: List[BaseDataSet] = []

        for dataloader_config in self.config.source_datasets:
            self.datasets.append(FileTypeFolderDataSet.init_from_config(dataloader_config))

        if self.config.ignore_blacklist == False:
            blacklist_df = pd.read_csv(self.config.blacklist_path)
            self.blacklist = list(blacklist_df[blacklist_df.columns[-1]])
        else:
            self.blacklist = []

        self.config.output_location.mkdir(parents=True, exist_ok=True)
        if not self.config.output_location.is_dir() or not self.config.output_location.exists():
            raise ValueError(f"CacheCreator output location does not exist or isn't a dir. Given path: {self.config.output_location}")

    def create_data_cache(self):
        """
        create cached FlowSample files from original FlowSamples
        """
        FlowSampleFile.Artifical_gates_defintions = self.config.gate_defintions
        FlowSampleFile.outlier_handler = OutlierHandler(self.config.outlier_handler_config)
        total_count = 0
        blacklist_count = 0
        for dataset in self.datasets:
            print(f"\ncreating cached files for dataset '{dataset.name}'")
            per_dataset_count = 0

            dataset_out_folder = self.config.output_location / Path(dataset.name)

            dataset_out_folder.mkdir(exist_ok=True)

            for sample in dataset:
                try:
                    if not self.config.ignore_blacklist and sample.get_sample_file_name() in self.blacklist:
                        blacklist_count += 1
                        print(f"\n\nskipped {sample.get_sample_file_name()} due to blacklist \n")
                    else:
                        cached_sample = FlowSampleCache.create_from_file_sample(sample)
                        cached_sample.save_to_file(dataset_out_folder / Path(cached_sample.get_sample_file_name()))
                        per_dataset_count += 1
                        total_count += 1
                except Exception as ex:
                    # raise Exception(f"Exception while creating cache of sample {sample.get_sample_file_name()}") from ex
                    print(f"Exception while creating cache of sample {sample.get_sample_file_name()}")
                    print(ex)
                    print("\n Stacktrace:")
                    traceback.print_exc()
                    print("\n")

            print(f"\ncreated '{per_dataset_count}' cached files for dataset '{dataset.name}'")

        print(f"\ncreated '{total_count}' cached files in total. skipped due to blacklist : '{blacklist_count}'")
