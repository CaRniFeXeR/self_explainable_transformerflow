from typing import List, Tuple
import numpy as np
import pandas as pd
import random

from ..datastructures.Gate import Gate
from ..datastructures.configs.datashiftaugmentationconfig import DatashiftAugmentationConfig


def increased_shift_augmentation(values, percentage: float, scale_factor: float = 1.0):
    return values + (percentage * scale_factor)


def decreased_shift_augmentation(values, percentage: float, scale_factor: float = 1.0):
    return values - (percentage * scale_factor)


class DataAugmenter:

    def __init__(self, augmentation_config: DatashiftAugmentationConfig) -> None:
        self.config = augmentation_config

    def augment_data(self, events: pd.DataFrame, labels: pd.DataFrame, gates: List[Gate], possible_markers: List[str]) -> Tuple[pd.DataFrame, List[Gate]]:

        if random.random() > 0.0:
            events, gates = self.min_max_transformation(events, labels, gates)
        if random.random() > 0.0:
            events, gates = self.shift_augmentation(events, gates, possible_markers)
        if random.random() > 0.05:
            events, gates = self.single_gate_shift_augmentaion(events, labels, gates)
        if self.config.enable_shear_augmentation:
            events, gates = self.single_gate_shear_augmentation(events, labels, gates)

        return events, gates

    def shift_augmentation(self, events: pd.DataFrame, gates: List[Gate], possible_markers: List[str]) -> Tuple[pd.DataFrame, List[Gate]]:

        for possible_marker in possible_markers:
            if self.config.shift_propability > random.random():

                augmentation_direction = increased_shift_augmentation if random.random() > 0.5 else decreased_shift_augmentation

                # don't use always the same shift percent but rather something in the range of (0, self.config.shift_percent)
                shift_percent = random.random() * self.config.shift_percent

                events[possible_marker] = augmentation_direction(events[possible_marker], shift_percent, scale_factor=4.5)
                for gate in gates:
                    if gate.y_marker == possible_marker:
                        gate.polygon[:, 1] = augmentation_direction(gate.polygon[:, 1], shift_percent)
                    if gate.x_marker == possible_marker:
                        gate.polygon[:, 0] = augmentation_direction(gate.polygon[:, 0], shift_percent)

        return events, gates

    def single_gate_shift_augmentaion(self, events: pd.DataFrame, labels: pd.DataFrame, gates: List[Gate]) -> Tuple[pd.DataFrame, List[Gate]]:
        # randomly select gate
        # randomly select marker
        # shift events inside gate
        # shift coresponding polygons and all sibling and ascending polygons
        already_considered_marker = set()
        for gate_to_augment in gates:
            if self.config.scale_propability > random.random():

                marker_idxes = list([0, 1])
                random.shuffle(marker_idxes)
                for i, marker_idx in enumerate(marker_idxes):
                    already_considered_gates = set()
                    if i == 0 or self.config.scale_propability_2nd_marker > random.random():
                        marker = gate_to_augment.x_marker if marker_idx == 0 else gate_to_augment.y_marker

                        if marker not in already_considered_marker:
                            already_considered_marker.add(marker)

                            change_factor = random.random() * 0.8 * self.config.polygon_scale_range[gate_to_augment.name]
                            augmentation_direction = increased_shift_augmentation if random.random() > 0.45 else decreased_shift_augmentation

                            if marker in ["CD45", "CD20", "CD38"]:
                                change_factor = change_factor / 5

                            # scale events
                            events_label_mask = labels[gate_to_augment.originalname] == True
                            if events_label_mask.sum() > 10:
                                events.loc[events_label_mask, marker] = augmentation_direction(events[events_label_mask][marker], change_factor, scale_factor=4.5)

                                gates_to_scale = [gate_to_augment]
                                while len(gates_to_scale) > 0:
                                    next_gates_to_scale = []
                                    for gate_polys_to_scale in gates_to_scale:
                                        if gate_polys_to_scale.name not in already_considered_gates:
                                            already_considered_gates.add(gate_polys_to_scale.name)
                                            if gate_polys_to_scale.x_marker == marker or gate_polys_to_scale.y_marker == marker:
                                                current_marker_idx = 0 if gate_polys_to_scale.x_marker == marker else 1
                                                gate_polys_to_scale.polygon[:, current_marker_idx] = augmentation_direction(gate_polys_to_scale.polygon[:, current_marker_idx], change_factor)

                                            next_gates_to_scale = next_gates_to_scale + self._get_siblings_gates(gates, gate_polys_to_scale.parentname, gate_polys_to_scale.name)

                                    next_gates_to_scale = [next_gate for next_gate in next_gates_to_scale if next_gate.name not in already_considered_gates]
                                    gates_to_scale = next_gates_to_scale

        return events, gates

    def single_gate_shear_augmentation(self, events: pd.DataFrame, labels: pd.DataFrame, gates: List[Gate]) -> Tuple[pd.DataFrame, List[Gate]]:

       

        for gate_to_augment in gates:
            if gate_to_augment.name in ["Syto", "Blasts_CD45CD10", "Blasts_CD20CD10", "Blasts_CD38CD10"]:
                # x' = x + Sh_x * y
                shear_factor = 0.1 * random.random()
                shear_factor = shear_factor if 0.5 > random.random() else -1 * shear_factor
                if gate_to_augment.name != "Syto":
                    shear_factor = shear_factor * 0.8
                shear_matrix = np.eye(2)
                shear_matrix[1, 0] = shear_factor
                shear_res = events.loc[:, [gate_to_augment.x_marker, gate_to_augment.y_marker]].dot(shear_matrix)
                events[gate_to_augment.x_marker] = shear_res[0]
                events[gate_to_augment.y_marker] = shear_res[1]
                gate_to_augment.polygon = np.matmul(gate_to_augment.polygon, shear_matrix)

        return events, gates

    def min_max_transformation(self, events: pd.DataFrame, labels: pd.DataFrame, gates: List[Gate]) -> Tuple[pd.DataFrame, List[Gate]]:
        # randomly select gate
        # randomly select marker
        # min-max scale 80% - 120%
        # scale
        # events in inside the polygon
        # the polygon itself
        # all following polygons if the use the selected marker
        for gate_to_augment in gates:
            if self.config.scale_propability > random.random():

                marker_idxes = list([0, 1])
                random.shuffle(marker_idxes)
                for i, marker_idx in enumerate(marker_idxes):
                    already_considered_gates = set()

                    if i == 0 or self.config.scale_propability_2nd_marker > random.random() / 2:
                        marker = gate_to_augment.x_marker if marker_idx == 0 else gate_to_augment.y_marker

                        change_factor = random.random() * 1.4 * self.config.polygon_scale_range[gate_to_augment.name]
                        scale_factor = 1 + change_factor if random.random() > 0.45 else 1 - change_factor

                        if marker == "CD10":
                            scale_factor = 1 + change_factor if random.random() > 0.7 else 1 - change_factor

                        polygon_scale_factor = scale_factor + 0.08 if scale_factor >= 1.0 else scale_factor + 0.02

                        # scale events
                        events_label_mask = labels[gate_to_augment.originalname] == True
                        if events_label_mask.sum() > 10:
                            events_min = min(events.loc[events_label_mask, marker])
                            events_max = max(events.loc[events_label_mask, marker])
                            min_max_dif = events_max - events_min
                            center_point = events_min + 0.5 * min_max_dif
                            events.loc[events_label_mask, marker] = (events[events_label_mask][marker] - center_point) * scale_factor + center_point

                            gates_to_scale = [gate_to_augment]
                            poly_min = min(gate_to_augment.polygon[:, marker_idx])
                            poly_max = max(gate_to_augment.polygon[:, marker_idx])
                            poly_min_max_dif = poly_max - poly_min
                            poly_center_point = poly_min + 0.5 * poly_min_max_dif
                            while len(gates_to_scale) > 0:
                                next_gates_to_scale = []
                                for gate_poly_to_scale in gates_to_scale:
                                    if gate_poly_to_scale.name not in already_considered_gates:
                                        already_considered_gates.add(gate_poly_to_scale.name)
                                        if gate_poly_to_scale.x_marker == marker or gate_poly_to_scale.y_marker == marker:
                                            current_marker_idx = 0 if gate_poly_to_scale.x_marker == marker else 1
                                            gate_poly_to_scale.polygon[:, current_marker_idx] = np.array(gate_poly_to_scale.polygon[:, current_marker_idx] - poly_center_point, dtype=np.float64) * polygon_scale_factor + poly_center_point

                                        next_gates_to_scale = next_gates_to_scale + self._get_gate_by_parent(gates, gate_poly_to_scale.originalname) + self._get_siblings_gates(gates, gate_poly_to_scale.parentname, gate_poly_to_scale.name)

                                next_gates_to_scale = [next_gate for next_gate in next_gates_to_scale if next_gate.name not in already_considered_gates]
                                gates_to_scale = next_gates_to_scale

        return events, gates

    def _get_gate_by_parent(self, gates: List[Gate], parent_gate_name: str):
        result = []
        for gate in gates:
            if gate.parentname == parent_gate_name:
                result.append(gate)

        return result

    def _get_siblings_gates(self, gates: List[Gate], parent_gate_name: str, current_gate_name: str):
        result = []
        for gate in gates:
            if gate.parentname == parent_gate_name and gate.name != current_gate_name:
                result.append(gate)

        return result
