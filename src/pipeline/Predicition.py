
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from src.datastructures.ProcessedFlowSample import ProcessedFlowSample
from src.datastructures.configs.dataproviderconfig import DataProviderConfig
from src.loader import FlowSample
from src.pipeline.DataProvider import DataProvider
from src.pipeline.GateComparisionLogger import GateComparisionLogger
from src.utils.execution_timer import timer
from ..datastructures.configs.predictionconfig import PredictionConfig
from ..loader.DataSet.FileTypeFolderDataSet import FileTypeFolderDataSet
from ..loader.IO.model_file_handler import ModelFileHandler
from ..pipeline.DeviceManager import DeviceManager
from ..pipeline.ModelFactory import ModelFactory
from ..pipeline.ValidationService import ValidationService
from ..utils.reproducibility_helper import setSeeds
from torchsummary import summary


class Prediction:

    def __init__(self, config: PredictionConfig) -> None:
        self.config = config
        self.prepare_prediction()

    def prepare_prediction(self):

        # load dataset
        self.dataset = FileTypeFolderDataSet.init_from_config(self.config.dataset)
        # load model from storage / init model
        modelFactory = ModelFactory(self.config.model_factory)
        self.model = modelFactory.create_instance()
        self.model_file_handler = ModelFileHandler(self.config.model_storage)

        if self.config.model_storage.load_stats_from_file:
            self.model_file_handler.load_state_from_file(self.model, True)

        print(self.model)
        self.device_manager = DeviceManager(self.config.gpu_name)
        self.config.default_retrieve_options.preprocess_data()
        self.validationSrv = ValidationService(self.config.default_retrieve_options, self.config.gpu_name)
        self.val_dataprovider = DataProvider(self.dataset, self.device_manager, DataProviderConfig(n_samples_to_load=1), self.config.default_retrieve_options)

        self.config.output_path.mkdir(exist_ok=True)

        setSeeds(self.config.random_seed)
        self.device_manager.move_to_gpu(self.model)

        summary(self.model, (2000, 10))

    def prediction_on_data(self) -> pd.DataFrame:

        self.model.train(False)
        print(f"\n\n starting model predict on dataset {self.dataset.name} \n\n")
        res_per_gate = self.validationSrv.validate_dataset(self.model, self.dataset, self.config.default_retrieve_options.events_seq_length)
        # todo save prediction result to csv file
        res_per_gate.to_csv(self.config.output_path / Path(f"{self.config.dataset.dataset_name}_validation_result.csv"))
        res_per_gate.mean().to_csv(self.config.output_path / Path(f"{self.config.dataset.dataset_name}_validation_result_mean.csv"))
        res_per_gate.median().to_csv(self.config.output_path / Path(f"{self.config.dataset.dataset_name}_validation_result_median.csv"))
        print(f"successfully predicted data of dataset {self.dataset.name} \n")

        return res_per_gate

    @timer
    def predict_sample(self, sample: FlowSample) -> Tuple[ProcessedFlowSample, torch.Tensor, torch.Tensor]:

        processed_sample = ProcessedFlowSample(sample, self.config.default_retrieve_options)
        val_sample_tensors = self.val_dataprovider.move_sample_to_gpu(processed_sample.to_tensors())
        val_polygons_pred = self.model(val_sample_tensors.events.unsqueeze(0), val_sample_tensors.padding_mask.unsqueeze(0))

        val_polygons_pred_dt = val_polygons_pred.detach().cpu().squeeze(0)
        polygons_gt = self.device_manager.move_to_cpu(val_sample_tensors.polygons)
        # necessary in order to free up memory in cuda
        val_polygons_pred = None

        return processed_sample, val_polygons_pred_dt, polygons_gt

    def log_gate_comparision_to_file(self, eval_res_df: pd.DataFrame = None):

        plotLogger = GateComparisionLogger(wandb_logger=None, retrieve_options=self.config.default_retrieve_options, file_extension=self.config.file_extension)

        for sample in self.dataset:
            try:

                processed_sample, val_polygons_pred_dt, polygons_gt = self.predict_sample(sample)

                plotLogger.save_all_gate_comparision_figures(self.config.output_path, sample.get_sample_file_name(), val_polygons_pred_dt, polygons_gt, eval_res_df=eval_res_df, processed_sample=processed_sample)
            except Exception as ex:
                print(f"error while plotting {sample.get_sample_file_name()}")
                print(ex)

    def log_gate_comparision_for_augmentation_variations(self, sample_name: str, n_variations: int):

        plotLogger = GateComparisionLogger(wandb_logger=None, retrieve_options=self.config.default_retrieve_options)

        for i in range(n_variations):
            val_dataprovider = DataProvider(self.dataset, self.device_manager, DataProviderConfig(n_samples_to_load=1), self.config.default_retrieve_options)

            for sample in self.dataset:
                if sample.get_sample_file_name() == sample_name:
                    try:
                        self.config.default_retrieve_options.augmentation_config.disable_augmentation = False
                        processed_sample = ProcessedFlowSample(sample, self.config.default_retrieve_options)
                        val_sample_tensors = val_dataprovider.move_sample_to_gpu(processed_sample.to_tensors())

                        val_polygons_pred = self.model(val_sample_tensors.events.unsqueeze(0), val_sample_tensors.padding_mask.unsqueeze(0))

                        val_polygons_pred_dt = val_polygons_pred.detach().cpu().squeeze(0)
                        # necessary in order to free up memory in cuda
                        val_polygons_pred = None
                        polygons_gt = self.device_manager.move_to_cpu(val_sample_tensors.polygons)

                        plotLogger.save_all_gate_comparision_figures(self.config.output_path, sample.get_sample_file_name() + f"_{i}", val_polygons_pred_dt, polygons_gt, processed_sample=processed_sample)
                    except Exception as ex:
                        print(f"error while plotting {sample.get_sample_file_name()}")
                        print(ex)
