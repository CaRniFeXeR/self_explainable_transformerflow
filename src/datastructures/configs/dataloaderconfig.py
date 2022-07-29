from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataLoaderConfig:
    root_path: Path
    dataset_name: str
    filename_must_contain: str = ""
    file_extension: str = ".xml"
    sample_factory_type : str = "src.loader.FlowSample.file_sample_factory.FileSampleFactory"