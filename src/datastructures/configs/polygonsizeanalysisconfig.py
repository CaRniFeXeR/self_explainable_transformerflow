from dataclasses import dataclass
from pathlib import Path
from typing import List
from ..SampleRetrieveOptions import SampleRetrieveOptions


from .dataloaderconfig import DataLoaderConfig


@dataclass
class PolygonSizeAnalysisConfig:
    datasets_config: List[DataLoaderConfig]
    outputpath: Path
    sizes_to_consider : List[int] 
    retrieve_options: SampleRetrieveOptions