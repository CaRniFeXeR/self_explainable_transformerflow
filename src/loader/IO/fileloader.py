from abc import ABC
from pathlib import Path


class FileLoader(ABC):

    def __init__(self, inputfile: Path):

        if inputfile == "":
            raise ValueError("inputfile is empty")

        self.checkFileExists(inputfile)

        self.inputfile = inputfile

    def checkFileExists(self, inputfile: Path):

        if not inputfile.exists():
            message = f"{inputfile} does not exist"
            print(message)
            raise Exception(message)
