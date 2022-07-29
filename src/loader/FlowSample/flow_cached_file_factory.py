from pathlib import Path
from .flow_sample_cache import FlowSampleCache

class CachedSampleFactory:

    def __init__(self) -> None:
        pass

    def init_blood_sample(self, inputfile_path: Path):
        return FlowSampleCache.load_from_file(inputfile_path)
