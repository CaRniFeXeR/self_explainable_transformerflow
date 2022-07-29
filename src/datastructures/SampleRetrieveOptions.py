from dataclasses import dataclass
from typing import List

import numpy as np

from src.datastructures.ArtificalGateDefinition import ArtificalGateDefinition
from .configs.datashiftaugmentationconfig import DatashiftAugmentationConfig


@dataclass
class SampleRetrieveOptions:
    used_markers: List[str]
    used_gates: List[str]
    events_mean: List[float]
    events_sd: List[float]
    gate_definitions : List[ArtificalGateDefinition]
    filter_gate : str = None
    used_marker_classes: List[str] = None
    used_original_gates : List[str] = None
    use_convex_gates: bool = True
    events_seq_length: int = 200000
    always_keep_blasts : bool = False
    gate_polygon_interpolation_length : int = 60
    gate_polygon_seq_length: int = 10
    polygon_min: float = -0.5
    polygon_max: float = 2.0
    shuffle: bool = True
    augmentation_config: DatashiftAugmentationConfig = None
    _preprocessed : bool = False

    def get_marker_index(self, marker_name :str) -> int:
        return self.used_markers.index(marker_name)
            

    def get_gate_definition_by_name(self, name : str) -> ArtificalGateDefinition:

        for gate_def in self.gate_definitions:
            if gate_def.name == name:
                return gate_def
        
        raise ValueError(f"gate definition with name '{name}' not found")

    def preprocess_data(self):

        if self._preprocessed == False:  # ensure preprocessing is only executed once

            if self.used_marker_classes == None:
                used_markers = []
                for gate in self.gate_definitions:
                    used_markers.append(gate.x_marker)
                    used_markers.append(gate.y_marker)
                
                self.used_marker_classes = list(set(used_markers))

            if self.used_original_gates == None:
                used_original_gates = []
                for gate in self.gate_definitions:
                    used_original_gates.append(gate.original_name)
                
                self.used_original_gates = list(set(used_original_gates))


            if self.augmentation_config is not None:
                # increase the data range to the possible augmentation values
                self.polygon_min -= self.polygon_min * self.augmentation_config.shift_percent
                self.polygon_max += self.polygon_max * self.augmentation_config.shift_percent
                # increase data sd to the possible augmentation values
                self.events_sd = np.array(self.events_sd) + np.array(self.events_sd) * self.augmentation_config.shift_percent

            self._preprocessed = True
