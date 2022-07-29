from .fileloader import FileLoader
import json


class JsonFileLoader(FileLoader):
    """
    Handles JSON File loading
    """

    def loadJsonFile(self) -> dict:
        with open(self.inputfile) as json_file:
            config_dict = json.load(json_file)

        return config_dict
