import numpy as np
from src.datastructures.configs.outlierhandlerconfig import OutlierHandlerConfig

from src.utils.outlier_handler import OutlierHandler

from ..datastructures.ProcessedFlowSample import ProcessedFlowSample
from ..utils.convex_hull_handler import convex_hull
from ..datastructures.Gate import Gate, GateType
import matplotlib.pyplot as plt
import matplotlib


class PolygonComparisionPlot:

    def __init__(self, ax:  matplotlib.axes = plt.gca()) -> None:
        self.ax = ax

    def draw(self, gate_a: Gate, gate_b: Gate, sub_title: str = None, eval_info: str = None, processed_sample: ProcessedFlowSample = None):

        self.ax.set_ylim([-0.2, 1.7])
        self.ax.set_xlim([-0.2, 1.7])
        self.ax.set_xlabel(gate_a.x_marker)
        self.ax.set_ylabel(gate_a.y_marker)
        self.ax.set_title(gate_a.name)

        convex_polygon_a = gate_a.polygon  # convex_hull(gate_a.polygon)
        convex_polygon_b = convex_hull(gate_b.polygon)

        self.ax.fill(convex_polygon_a[:, 0], convex_polygon_a[:, 1], facecolor='none', edgecolor='black', linewidth=2, linestyle="dashed", label="used GT")
        self.ax.fill(convex_polygon_b[:, 0], convex_polygon_b[:, 1], facecolor='none', edgecolor='red', linewidth=3, label="Prediction")

        if processed_sample is not None:
            # events must be scaled to be on same range with polygons
            events_x_y = processed_sample.get_screen_scaled_events_with_gate_infos(gate_a)
            events_x_y_in_gate = events_x_y[events_x_y["in_gate"] == 1]
            self.ax.scatter(events_x_y_in_gate["x_marker"], events_x_y_in_gate["y_marker"], s=0.05, alpha=0.8, color="green", rasterized=True)
            # eval_info += f"\n n_events_predicted : {(events_x_y['in_gate'] == 1).sum()}"
            gate_labels = processed_sample.gate_labels.loc[processed_sample.padding_mask == 0]
            eval_info += f"\n n_events_gt : {(gate_labels[gate_a.originalname] == 1).sum()}"

            if gate_a.parentname is not None and gate_a.parentname != "":
                events_parent = events_x_y[(gate_labels[gate_a.parentname] == 1) & (events_x_y["in_gate"] == 0)]
                self.ax.scatter(events_parent["x_marker"], events_parent["y_marker"], s=0.025, color="gray", rasterized=True)

            if gate_a.name == gate_a.originalname:
                gate_c = processed_sample.get_gate_by_name(gate_a.name, GateType.operator_gt_gate)
                if gate_c.x_marker == gate_a.x_marker and gate_c.y_marker == gate_a.y_marker:
                    polygon_c = gate_c.polygon
                    self.ax.fill(polygon_c[:, 0], polygon_c[:, 1], facecolor='none', edgecolor='orange', linewidth=3, linestyle="dashed", label="unaugmented operator")
            # try:
            #     outHandler = OutlierHandler(OutlierHandlerConfig(alpha = 0.00001, n_events_threshold=300))
            #     outlier_mask = outHandler.get_non_outliers(events_x_y_in_gate.loc[:,["x_marker","y_marker"]])
            #     events_convex_hull = convex_hull(np.array(events_x_y_in_gate[outlier_mask == True]))
            #     self.ax.fill(events_convex_hull[:, 0], events_convex_hull[:, 1], facecolor='none', edgecolor='green', linewidth=1, linestyle="dashed", label="augmented gt gate")
            # except Exception as ex:
            #     pass
            gate_e = processed_sample.get_gate_by_name(gate_a.name, GateType.convex_hull_augmented_gt_gate)
            self.ax.fill(gate_e.polygon[:, 0], gate_e.polygon[:, 1], facecolor='none', edgecolor='green', linewidth=1, linestyle="dashed", label="augmented gt gate")

            gate_d = processed_sample.get_gate_by_name(gate_a.name, GateType.convex_hull_non_augmented_gt_gate)
            self.ax.fill(gate_d.polygon[:, 0], gate_d.polygon[:, 1], facecolor='none', edgecolor='blue', linewidth=2, linestyle="dashed", label="unaugmented gt gate")

        if eval_info is not None:
            self.ax.text(0.3, 0.8, eval_info, fontsize=12, transform=self.ax.transAxes)

        if sub_title is not None:
            plt.suptitle(sub_title)

        self.ax.legend()
