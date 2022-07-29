import traceback
from typing import List

from copy import deepcopy

from ..pipeline.DataAugmenter import DataAugmenter
from ..utils.polygon_interpolation import interpolate_polygon, reduce_polygon_length

from ..loader.GateCollection import GateCollection
from ..datastructures.Gate import Gate, GateType
from ..loader.FlowSample.flowsample_tensor_data import FlowSampleTensorData
from ..pipeline.DataTransformers import denormalise_events, normalise_events, scale_polygon_points
from .SampleRetrieveOptions import SampleRetrieveOptions
from ..loader.FlowSample.flow_sample_base import FlowSampleBase
import torch
import numpy as np
import pandas as pd


class ProcessedFlowSample:
    """
    Can be created from a FlowSample (file or cache) and represents the actual FlowSample that's used in training.
    SampleRetrieveOptions define the transformations that have to be applied: eg.: padding, selecting markers, selecting gates etc.
    """

    def __init__(self, flowsample: FlowSampleBase, retrieve_options: SampleRetrieveOptions):
        self.flowsample = flowsample
        self.retrieve_options = retrieve_options
        if self.retrieve_options.augmentation_config is not None:
            self.augmenter = DataAugmenter(self.retrieve_options.augmentation_config)
        self._process_sample()

    def _process_sample(self):
        self._select_markers()
        self._load_gate_labels()
        self._filter_events()
        self._load_gates()
        self._shuffle_data()
        self._augment_data()
        self._scale_events()
        self._apply_padding()
        self._process_gates()

    def _select_markers(self):
        self.events = self.flowsample.get_events()
        markers = self.events.columns.array

        # if not all(requested_marker in markers for requested_marker in self.retrieve_options.used_markers):
        for requested_marker in self.retrieve_options.used_markers:
            if requested_marker not in markers:
                raise ValueError(f" requested marker '{requested_marker}' is not in the dataframe of sample '{self.flowsample.get_sample_file_name()}'. Containing markers are '{list(markers)}'")

        self.events = self.events.loc[:, self.retrieve_options.used_markers]

    def _load_gate_labels(self):

        self.gate_labels = self.flowsample.get_gate_labels()
        self.gate_labels = GateCollection.renameGates(self.gate_labels)
        self.gate_labels = self.gate_labels[self.retrieve_options.used_original_gates]

    def _filter_events(self):
        if self.retrieve_options.filter_gate is not None and self.retrieve_options.filter_gate != "":
            filter_gate = self.retrieve_options.filter_gate
            if filter_gate not in self.gate_labels.columns:
                raise ValueError(f"filtergate '{filter_gate}' not presented in the Labels")

            self.events = self.events[self.gate_labels[filter_gate] == True]
            self.gate_labels = self.gate_labels[self.gate_labels[filter_gate] == True]

    def _load_gates(self):

        self.gates = self.flowsample.get_convex_gates() if self.retrieve_options.use_convex_gates == True else self.flowsample.get_gates()
        self.unmodified_gates = self.gates
        self.gates = deepcopy(self.gates)

        gates_to_use = []
        for gate in self.gates:
            if gate.name in self.retrieve_options.used_gates:
                gates_to_use.append(gate)

        self.gates = gates_to_use

    def _shuffle_data(self):
        """
        shuffle events and labels together
        index of gt and events must be the same
        """
        if self.retrieve_options.shuffle:
            n_events = len(self.events)
            n_gate_labels = len(self.gate_labels)

            if n_events != n_gate_labels:
                raise Exception(f"miss match between n_events ('{n_events}') and n_gate_labels ('{n_gate_labels}')")

            shuffled_idx = list(range(0, n_events))
            np.random.shuffle(shuffled_idx)
            self.events = self.events.iloc[shuffled_idx, :]
            self.gate_labels = self.gate_labels.iloc[shuffled_idx, :]

    def _augment_data(self):

        if self.retrieve_options.augmentation_config is not None and self.retrieve_options.augmentation_config.disable_augmentation == False:
            try:
                events, gates = self.augmenter.augment_data(self.events, self.gate_labels, deepcopy(self.gates), self.retrieve_options.used_marker_classes)
                self.events = events
                self.gates = gates
            except Exception as ex:
                traceback.print_exc()
                print(f"error while data augmentation: '{self.flowsample.get_sample_file_name()}'")
                print(ex)

    def _scale_events(self):

        self.events = normalise_events(self.events, self.retrieve_options.events_mean, self.retrieve_options.events_sd)

    def _apply_padding(self):
        n_events = len(self.events)
        needed_seq_len = self.retrieve_options.events_seq_length

        # todo consider using a mask

        if n_events >= needed_seq_len:  # more events than needed --> cut
            self.padding_mask = np.zeros(needed_seq_len)

            if self.retrieve_options.always_keep_blasts == False:
                self.events = self.events.head(needed_seq_len)
                self.gate_labels = self.gate_labels.head(needed_seq_len)
            else:
                # first select blasts, than cut the rest
                blast_mask = self.gate_labels["Blasts"] == 1
                blast_events = self.events[blast_mask]
                blast_labels = self.gate_labels[blast_mask]
                n_blasts = len(blast_events)
                if n_blasts > needed_seq_len * 0.2:
                    #blasts should never be more than 20% of all events
                    needed_blasts = int(needed_seq_len * 0.2)
                    blast_events = blast_events.head(needed_blasts)
                    blast_labels = blast_labels.head(needed_blasts)
                    n_blasts = len(blast_events)

                # sample from non-blast to fill up n_events
                needed_non_blasts = needed_seq_len - n_blasts
                non_blast_events = self.events[~blast_mask].head(needed_non_blasts)
                non_blast_labels = self.gate_labels[~blast_mask].head(needed_non_blasts)
                shuffled_idx = list(range(0, needed_seq_len))
                np.random.shuffle(shuffled_idx)
                self.events = pd.concat([non_blast_events, blast_events], ignore_index=True).iloc[shuffled_idx, :]
                self.gate_labels = pd.concat([non_blast_labels, blast_labels], ignore_index=True).iloc[shuffled_idx, :]

        else:  # less events than needed --> pad
            n_to_pad = needed_seq_len - n_events
            pad_event_values = np.zeros((n_to_pad, self.events.shape[-1]), dtype=np.double)
            pad_events_df = pd.DataFrame(pad_event_values, columns=self.events.columns)
            self.events = pd.concat([self.events, pad_events_df])

            pad_label_values = np.zeros((n_to_pad, self.gate_labels.shape[-1]))
            pad_labels_df = pd.DataFrame(pad_label_values, columns=self.gate_labels.columns)
            self.gate_labels = pd.concat([self.gate_labels, pad_labels_df])

            self.padding_mask = np.array([0] * n_events + [1] * n_to_pad)

    def _scale_gate_polygons(self, polygon: np.ndarray) -> np.ndarray:
        return scale_polygon_points(polygon, self.retrieve_options.polygon_min, self.retrieve_options.polygon_max)

    def _process_gates(self):

        self.used_gates = []
        self.gates_polygons = []

        for gatename in self.retrieve_options.used_gates:
            current_gate = None
            for gate in self.gates:
                if gate.name == gatename:
                    current_gate = gate
                    break

            if current_gate is None:
                raise Exception(f"'{gatename}' is not presented in the sample: {self.flowsample.get_sample_name()}")


            polygon_adjusted = interpolate_polygon(current_gate.polygon, self.retrieve_options.gate_polygon_interpolation_length)
            polygon_adjusted = reduce_polygon_length(polygon_adjusted, self.retrieve_options.gate_polygon_seq_length)
            polygon_adjusted = self._scale_gate_polygons(polygon_adjusted)

            self.gates_polygons.append(polygon_adjusted)
            self.used_gates.append(gate)

        self.gates_polygons = np.array(self.gates_polygons)

    # region [public methods]

    def to_tensors(self) -> FlowSampleTensorData:
        """
        converts processed FlowSample to Pytorch Tensors.
        Returns: FlowSampleTensorData(Padded Event Tensor, Padded Event Label Tensor, Padding Mask Tensor, Padded Polygon Tensor, Padded Gate Class Tensor)
        """
        sample_names = [self.flowsample.get_sample_file_name()]
        events_tensor = torch.from_numpy(self.events.to_numpy()).float()
        labels_tensor = torch.from_numpy(self.gate_labels.to_numpy()).float()
        padding_mask_tensor = torch.from_numpy(self.padding_mask) > 0
        polygon_tensor = torch.from_numpy(self.gates_polygons).float()

        return FlowSampleTensorData(sample_names, events_tensor, labels_tensor, padding_mask_tensor,  polygon_tensor)

    def get_unscaled_events(self):

        return denormalise_events(self.events, self.retrieve_options.events_mean, self.retrieve_options.events_sd).loc[self.padding_mask == 0, :]

    def get_screen_scaled_events_with_gate_infos(self, interested_gate: Gate) -> pd.DataFrame:

        unscaled_events = self.get_unscaled_events().loc[:, [interested_gate.x_marker, interested_gate.y_marker]]

        unscaled_events.columns = ["x_marker", "y_marker"]
        unscaled_events = unscaled_events / 4.5

        if "\\" in interested_gate.parentname:
            interested_gate.parentname = interested_gate.parentname.split("\\")[-1]

        if interested_gate.parentname is not None and interested_gate.parentname != "" and "\\" not in interested_gate.parentname and interested_gate.parentname in self.gate_labels.columns.array:
            unscaled_events["in_parent_gate"] = self.gate_labels.loc[self.padding_mask == 0, interested_gate.parentname]

        unscaled_events["in_gate"] = self.gate_labels.loc[self.padding_mask == 0, interested_gate.originalname]

        return unscaled_events

    def get_gate_by_name(self, gatename: str, gate_type: GateType = GateType.convex_hull_augmented_gt_gate) -> Gate:

        if gate_type == GateType.convex_hull_augmented_gt_gate:
            gates = self.gates
        elif gate_type == GateType.convex_hull_non_augmented_gt_gate:
            gates = self.unmodified_gates
        elif gate_type == GateType.operator_gt_gate:
            gates = self.flowsample.get_gates()
        elif gate_type == GateType.predicted_gate:
            raise ValueError("GateType predicted_gate not supported for this method")

        for gate in gates:
            if gate.name == gatename:
                return gate

        raise ValueError(f"Gate '{gatename}' not found!")

    def get_used_gates(self) -> List[Gate]:

        return self.used_gates

    # endregion
