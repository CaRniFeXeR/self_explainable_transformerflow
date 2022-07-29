from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from src.datastructures.configs.predictionconfig import PredictionConfig
from ..SampleRetrieveOptions import SampleRetrieveOptions


from .dataloaderconfig import DataLoaderConfig


@dataclass
class SingleSampleAnalysisConfig:
    dataset: DataLoaderConfig
    outputpath: Path
    retrieve_options: SampleRetrieveOptions
    gate_color_dict: Dict[str, str]
    n_sampling: int = 5000
    file_extention: str = ".png"
    filename_filter: str = ""
    calculate_f1_score : bool = False
    predictionconfig: PredictionConfig = None
