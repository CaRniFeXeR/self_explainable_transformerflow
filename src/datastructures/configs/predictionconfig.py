
from dataclasses import dataclass
from pathlib import Path

from .modelfactoryconfig import ModelFactoryConfig
from .modelstorageconfig import ModelStorageConfig
from .dataloaderconfig import DataLoaderConfig
from ..SampleRetrieveOptions import SampleRetrieveOptions


@dataclass
class PredictionConfig:
    name: str
    default_retrieve_options: SampleRetrieveOptions
    dataset: DataLoaderConfig
    model_storage: ModelStorageConfig
    model_factory: ModelFactoryConfig
    output_path : Path
    polygon_only_prediction : bool = False
    random_seed : int = 42
    gpu_name : str = "cuda"
    file_extension : str = ".png"