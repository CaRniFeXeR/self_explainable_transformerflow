from dataclasses import dataclass


@dataclass
class DataProviderConfig:
    n_samples_to_load: int = 10
    batchsize: int = 1
