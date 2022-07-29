from dataclasses import dataclass
from pathlib import Path
from ..SampleRetrieveOptions import SampleRetrieveOptions


from .dataloaderconfig import DataLoaderConfig


@dataclass
class PloygonDistributionAnalysisConfig:
    dataset: DataLoaderConfig
    outputpath: Path
    retrieve_options: SampleRetrieveOptions
    file_extention: str = ".png"
