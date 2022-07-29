from dataclasses import dataclass
from typing import List
from pathlib import Path
from src.datastructures.configs.dataloaderconfig import DataLoaderConfig
from src.datastructures.configs.modelfactoryconfig import ModelFactoryConfig

from src.datastructures.configs.modelstorageconfig import ModelStorageConfig

from ..SampleRetrieveOptions import SampleRetrieveOptions
from ..ArtificalGateDefinition import ArtificalGateDefinition


@dataclass
class DashAppConfig:
    sample_retrieve_options: SampleRetrieveOptions
    gate_defintions: List[ArtificalGateDefinition]
    model_storage: ModelStorageConfig
    model_factory: ModelFactoryConfig
    dataset: DataLoaderConfig
    validation_result_dataframe_path: Path
    gpu_name: str = "cuda"
