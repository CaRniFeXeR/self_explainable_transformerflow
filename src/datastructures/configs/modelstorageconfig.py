from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelStorageConfig:
    file_path: Path
    load_stats_from_file: bool
    gpu_name : str
