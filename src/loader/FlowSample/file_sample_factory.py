from pathlib import Path
from .flow_sample_file import FlowSampleFile

class FileSampleFactory:

    def __init__(self) -> None:
        pass

    def init_blood_sample(self, inputfile_path: Path):
        return FlowSampleFile(inputfile_path)
