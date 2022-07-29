from typing import List
import numpy as np
import pandas as pd

from src.utils.points_in_polygon_helper import points_in_polygon
from ..datastructures.configs.dataproviderconfig import DataProviderConfig
from ..loader.DataSet.BaseDataSet import BaseDataSet

from ..model.FlowGATR import FlowGATR
from ..pipeline.DataProvider import DataProvider
from ..pipeline.DeviceManager import DeviceManager
from ..utils.convex_hull_handler import convex_hull

from ..datastructures.SampleRetrieveOptions import SampleRetrieveOptions
from ..datastructures.GatePredictionValidationResult import GatePredictionValidationResult

from ..pipeline.DataTransformers import unscale_polygon_points

from ..datastructures.ProcessedFlowSample import ProcessedFlowSample
from sklearn.metrics import f1_score
import copy
import numpy as np
import traceback


class ValidationService:
    """
    provides methods to validate a model on given data
    """

    def __init__(self, retrieve_options: SampleRetrieveOptions, gpu_name: str) -> None:
        self.retrieve_options = copy.deepcopy(retrieve_options)
        self.device_manager = DeviceManager(gpu_name)

    def validate_dataset(self, model: FlowGATR, dataset: BaseDataSet, val_events_seq_len : int = 400000) -> pd.DataFrame:
        """
        validates each sample in the given dataset on the given model and returns the result for each sample & gate as a Pandas Dataframe
        """

        val_dataprovider = DataProvider(dataset, self.device_manager, DataProviderConfig(n_samples_to_load=1), self.retrieve_options)
        validation_results = []
        self.retrieve_options.augmentation_config.disable_augmentation = True
        self.retrieve_options.always_keep_blasts = False
        self.retrieve_options.events_seq_length = val_events_seq_len

        for sample in dataset:
            try:
                processed_sample = ProcessedFlowSample(sample, self.retrieve_options)
                val_sample_tensors = val_dataprovider.move_sample_to_gpu(processed_sample.to_tensors())

                val_polygons_pred = model(val_sample_tensors.events.unsqueeze(0), val_sample_tensors.padding_mask.unsqueeze(0))

                val_polygons_pred_np = val_polygons_pred.detach().cpu().squeeze(0).numpy()

                # necessary in order to free up memory in cuda
                val_polygons_pred = None
                res = self.validate_sample(processed_sample, val_polygons_pred_np)
                validation_results.append(res)
            except Exception as ex:
                print(f"error while validating on {sample.get_sample_file_name()}")
                traceback.print_exc()
                print(ex)

        res_per_gate = self.convert_validation_result_to_dataframe(validation_results)
        return res_per_gate

    def validate_sample(self, sample: ProcessedFlowSample, predicted_gates: np.ndarray) -> List[GatePredictionValidationResult]:
        """
        computes validations scores for a given sample
        input: sample events, gt polygons, predicted polygons

        output: per polygon f1 score (events in predicted polygon vs. events in gt polygon)
        """

        gt_gates = sample.get_used_gates()
        events = sample.get_unscaled_events()

        if len(predicted_gates) != len(gt_gates):
            raise ValueError(f"inconsitent number of gates: len(gt_gates) {len(gt_gates)} len(predicted_gates) {len(predicted_gates)}")

        events.reset_index(drop=True, inplace=True)
        n_events = len(events)

        selected_gt = np.ones(len(events)) == 1
        selected_pd = np.ones(len(events)) == 1
        gate_results = []

        for gt_gate, predicted_gate in zip(gt_gates, predicted_gates):

            predicted_gate_hull = convex_hull(predicted_gate)
            predicted_poly_unscaled = unscale_polygon_points(predicted_gate_hull, self.retrieve_options.polygon_min, self.retrieve_options.polygon_max)

            # get ground truth idx of marker classes
            x_idx_gt = self.retrieve_options.get_marker_index(gt_gate.x_marker)
            y_idx_gt = self.retrieve_options.get_marker_index(gt_gate.y_marker)

            # calc points in polygon (all events both polygons predicted and gt)
            events_in_gt_gate = points_in_polygon(events, gt_gate.polygon, x_idx=x_idx_gt, y_idx=y_idx_gt)
            events_in_pd_gate = points_in_polygon(events, predicted_poly_unscaled, x_idx=x_idx_gt, y_idx=y_idx_gt)

            # calculate f1 score based on gt events
            selected_pd_gt_based = selected_gt & events_in_pd_gate
            selected_gt = selected_gt & events_in_gt_gate
            f1_gt_events = f1_score(selected_gt, selected_pd_gt_based)

            # calculate f1 score in sequence
            selected_pd = selected_pd & events_in_pd_gate
            f1_in_sequence = f1_score(selected_gt, selected_pd)

            # calculate f1 score gt label (no polygon calculation for gt)
            gt_labels = sample.gate_labels.loc[sample.padding_mask == 0, gt_gate.originalname]
            f1_gt_label = f1_score(gt_labels, selected_pd)

            gt_labels_n_events = gt_labels.reset_index(drop=True).sum()
            gate_val_res = GatePredictionValidationResult(sample.flowsample.get_sample_file_name(), gt_gate, f1_gt_events, f1_in_sequence, f1_gt_label, n_events_gt=gt_labels.sum(),
                                                          n_events_pd=selected_pd.sum(), n_events_label_gt=gt_labels_n_events)
            gate_results.append(gate_val_res)

            gate_val_res.mrd_predicted = sum(selected_pd) / n_events
            gate_val_res.mrd_true = sum(selected_gt) / n_events

        return gate_results

    def validate_sample_polygon_against_gt_label(self, sample: ProcessedFlowSample) -> List[GatePredictionValidationResult]:
        """
        validates the processed poylgons against the operator provided gt labels
        """

        gt_gates = sample.get_used_gates()
        polygons = sample.gates_polygons
        events = sample.get_unscaled_events()

        events.reset_index(drop=True, inplace=True)
        n_events = len(events)

        selected_gt = np.ones(len(events)) == 1
        gate_results = []

        for gt_gate, gt_polygon in zip(gt_gates, polygons):

            x_idx_gt = self.retrieve_options.get_marker_index(gt_gate.x_marker)
            y_idx_gt = self.retrieve_options.get_marker_index(gt_gate.y_marker)

            # calc points in polygon (all events both polygons predicted and gt)
            gt_polygon = unscale_polygon_points(gt_polygon, self.retrieve_options.polygon_min, self.retrieve_options.polygon_max)
            events_in_gt_gate = points_in_polygon(events, gt_polygon, x_idx=x_idx_gt, y_idx=y_idx_gt)
            selected_gt = selected_gt & events_in_gt_gate

            # calculate f1 score gt label (no polygon calculation for gt)
            gt_labels = sample.gate_labels.loc[sample.padding_mask == 0, gt_gate.originalname]
            f1_gt_label = f1_score(gt_labels, selected_gt)

            gt_labels_n_events = gt_labels.reset_index(drop=True).sum()
            gate_val_res = GatePredictionValidationResult(sample.flowsample.get_sample_file_name(), gt_gate, -1, -1, f1_gt_label, n_events_gt=gt_labels.sum(),
                                                          n_events_pd=-1, n_events_label_gt=gt_labels_n_events)
            gate_results.append(gate_val_res)

            gate_val_res.mrd_true = sum(selected_gt) / n_events
            gate_val_res.mrd_predicted = -1

        return gate_results

    def convert_validation_result_to_dataframe(self, validation_results: List[List[GatePredictionValidationResult]], use_gt_labels : bool = True) -> pd.DataFrame:
        """
        extracts a Pandas Dataframe from a list of validation_results (on per sample), that includes the most important information
        """

        col_names = ["name"] + self.retrieve_options.used_gates + ["in sequence"] + ["in sequence gt labels"] + ["mrd_true"] + ["mrd_predicted"]
        per_gate_results = pd.DataFrame(columns=col_names)
        for sample_validation_result in validation_results:
            sample_results = []
            sample_results.append(sample_validation_result[0].sample_name)
            for i, validation_res in enumerate(sample_validation_result):
                if use_gt_labels:
                    sample_results.append(validation_res.f1_score_ground_truth_labels)
                else:
                    sample_results.append(validation_res.f1_score_ground_truth_events)

            sample_results.append(sample_validation_result[-1].f1_score_predicted_events_sequential)
            sample_results.append(sample_validation_result[-1].f1_score_ground_truth_labels)
            sample_results.append(sample_validation_result[-1].mrd_true)
            sample_results.append(sample_validation_result[-1].mrd_predicted)

            per_gate_results = per_gate_results.append(pd.Series(sample_results, index=col_names), ignore_index=True)

        return per_gate_results

    def validate_poylgons_against_gt(self, dataset: BaseDataSet) -> pd.DataFrame:

        print(f"\nvalidating {dataset.name}'s polygons against gt labels\n")

        validation_results = []
        augmentation_disabled = self.retrieve_options.augmentation_config.disable_augmentation == True
        if not augmentation_disabled:
            self.retrieve_options.augmentation_config.disable_augmentation = True

        for sample in dataset:
            try:
                processed_sample = ProcessedFlowSample(sample, self.retrieve_options)

                res = self.validate_sample_polygon_against_gt_label(processed_sample)
                validation_results.append(res)
            except Exception as ex:
                print(f"error while validating on {sample.get_sample_file_name()}")
                traceback.print_exc()
                print(ex)

        res_per_gate = self.convert_validation_result_to_dataframe(validation_results, use_gt_labels=True)
        self.retrieve_options.augmentation_config.disable_augmentation = augmentation_disabled
        return res_per_gate
