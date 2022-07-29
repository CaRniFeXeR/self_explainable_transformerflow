from copy import copy
import pandas as pd
from pathlib import Path
import matplotlib
import torch

from ..datastructures.ProcessedFlowSample import ProcessedFlowSample
from ..pipeline.DataTransformers import unscale_polygon_points
from ..datastructures.Gate import Gate
from ..datastructures.LossResult import LossResult
from ..datastructures.SampleRetrieveOptions import SampleRetrieveOptions
from .WandbLogger import WandbLogger
import matplotlib.pyplot as plt
from ..visualizations.polygon_comparison_plot import PolygonComparisionPlot


class GateComparisionLogger:
    """
    Enables to log Gate Comparision Figures of a sample to Wandb or Harddrive
    """

    def __init__(self, wandb_logger: WandbLogger, retrieve_options: SampleRetrieveOptions, file_extension : str = ".png") -> None:
        self.logger = wandb_logger
        self.retrieve_options = retrieve_options
        self.file_extension = file_extension

    def log_all_gate_comparision_figures(self, sample_name: str, predicted_ploygons: torch.FloatTensor, gt_polygons: torch.FloatTensor, loss: LossResult = None, step: int = None):
        """
        Logs Gate Comparision Figure to Wandb based on the given data for all 5 Gates (Syto, Singlets, Intact, CD19 and Blasts)
        """
        if self.logger is None:
            raise ValueError("no wandb Logger given!")

        try:
            for idx, gate_name in enumerate(self.retrieve_options.used_gates):
                self._log_gate_comparision_figure(sample_name, predicted_ploygons, gt_polygons, idx, gate_name, loss, step)

        except Exception as ex:
            print("exception while logging gate comaprsision figures")
            print(ex)

    def save_all_gate_comparision_figures(self, file_path: Path, sample_name: str, predicted_ploygons: torch.FloatTensor, gt_polygons: torch.FloatTensor, processed_sample: ProcessedFlowSample = None, loss: LossResult = None, eval_res_df: pd.DataFrame = None):

        file_path.mkdir(exist_ok=True)
        n_cols = len(self.retrieve_options.gate_definitions)
        fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(n_cols * 10, 10))
        for idx, gate_def in enumerate(self.retrieve_options.gate_definitions):
            self._create_gate_plot(sample_name,  predicted_ploygons,  gt_polygons, idx, gate_def.name, processed_sample, loss, ax=axes[idx], eval_res_df=eval_res_df)

        img_file_path = file_path / Path(sample_name + self.file_extension)
        plt.savefig(img_file_path)
        plt.clf()
        plt.close()
        plt.cla()
        plt.clf()
        print(f"sucessfully saved plot to '{img_file_path}'")

    def _log_gate_comparision_figure(self, sample_name: str, predicted_ploygons: torch.FloatTensor,  gt_polygons: torch.FloatTensor, index: int, gate_name: str, loss: LossResult, step: int):

        fig, ax = plt.subplots(figsize=(9, 9))
        self._create_gate_plot(sample_name, predicted_ploygons, gt_polygons, index, gate_name, loss=loss,  ax=ax)
        self.logger.log_figure_as_img(f"comparision_plot {gate_name}", fig, False, step)
        plt.clf()
        plt.close()
        plt.cla()
        plt.clf()

    def _create_gate_plot(self,  sample_name: str, predicted_ploygons: torch.FloatTensor,  gt_polygons: torch.FloatTensor, index: int, gate_name: str, processed_sample: ProcessedFlowSample = None, loss: LossResult = None, ax:  matplotlib.axes = plt.gca(), eval_res_df: pd.DataFrame = None):

        gateA = Gate.from_gate_definition(self.retrieve_options.get_gate_definition_by_name(gate_name))
        gateB = copy(gateA)
        gateA.set_polygon_from_tensor(unscale_polygon_points(gt_polygons[index], self.retrieve_options.polygon_min, self.retrieve_options.polygon_max))
        gateB.set_polygon_from_tensor(unscale_polygon_points(predicted_ploygons[index], self.retrieve_options.polygon_min, self.retrieve_options.polygon_max))

        additional_info = f"{sample_name}"
        eval_info = ""
        if loss is not None:
            additional_info += f"\npolygonloss: {loss.point_loss[index].detach().numpy():.5f}"

        if eval_res_df is not None and gateA.name in eval_res_df.columns.array and eval_res_df["name"].str.contains(sample_name).any():
            row = eval_res_df[eval_res_df["name"] == sample_name].head(1).iloc[0]
            eval_info += f"F1 Score: {row[gateA.name]:.5f}"

            if "Blasts" in gateA.name:
                eval_info += f"\n in seq: {row['in sequence']:.5f}"
                eval_info += f"\n in seq gt: {row['in sequence gt labels']:.5f}"

        polygonPlot = PolygonComparisionPlot(ax)
        polygonPlot.draw(gateA, gateB, additional_info, eval_info=eval_info, processed_sample=processed_sample)
