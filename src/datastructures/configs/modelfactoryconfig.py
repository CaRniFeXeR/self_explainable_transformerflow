from dataclasses import dataclass


@dataclass
class ModelFactoryConfig:
    model_type: str
    params_type: str
    params: dict
