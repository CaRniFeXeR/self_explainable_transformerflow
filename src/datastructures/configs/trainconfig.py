
from dataclasses import dataclass

from .wandbconfig import WandbConfig
from .modelfactoryconfig import ModelFactoryConfig
from .modelstorageconfig import ModelStorageConfig
from .trainparams import TrainParams
from .dataloaderconfig import DataLoaderConfig
from ..SampleRetrieveOptions import SampleRetrieveOptions


@dataclass
class TrainConfig:
    name: str
    default_retrieve_options: SampleRetrieveOptions
    train_data: DataLoaderConfig
    validation_data: DataLoaderConfig
    model_storage: ModelStorageConfig
    train_params: TrainParams
    model_factory: ModelFactoryConfig
    wandb_config : WandbConfig
    gpu_name : str = "cuda"
    n_workers : int = 0