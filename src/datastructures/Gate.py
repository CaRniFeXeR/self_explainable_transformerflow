from dataclasses import dataclass
import numpy as np
import torch
import enum

from .ArtificalGateDefinition import ArtificalGateDefinition


class GateType(enum.Enum):
    operator_gt_gate = 1,
    predicted_gate = 2,
    convex_hull_augmented_gt_gate = 3,
    convex_hull_non_augmented_gt_gate = 4


@dataclass
class Gate:
    name: str
    originalname: str
    parentname: str
    x_marker: str
    y_marker: str
    polygon: np.ndarray

    @property
    def n_vertex(self):
        return self.polygon.shape[0]

    def set_polygon_from_numpy(self, polygon: np.ndarray):
        polygon_filtered = np.array([gate_value for gate_value in polygon if gate_value[0] > -1])
        self.polygon = polygon_filtered

    def set_polygon_from_tensor(self, polygon: torch.Tensor):

        polygon_np = polygon.detach().numpy()
        self.set_polygon_from_numpy(polygon_np)

    @staticmethod
    def from_gate_definition(gate_def: ArtificalGateDefinition):
        return Gate(gate_def.name, gate_def.original_name, gate_def.parent_name, gate_def.x_marker, gate_def.y_marker, np.empty(0))