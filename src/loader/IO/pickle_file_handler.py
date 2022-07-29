import pickle
from pathlib import Path


class PickleFileHandler:
    """
    Handles Pickle File IO operations
    """

    def __init__(self, file_path: Path) -> None:

        if not isinstance(file_path, Path):
            raise TypeError(f"file_path must be of type 'Path'. Type given '{type(file_path)}'")

        self.file_path = file_path

    def save_obj_as_pickle(self, obj):
        """
        Saves a given object as pickle file to the specified location
        """
        try:
            with open(str(self.file_path) + ".pkl", "wb+") as outfile:
                pickle.dump(obj, outfile, protocol= pickle.HIGHEST_PROTOCOL)
            print(f"\nsuccessfully saved an obj to '{self.file_path}'")
        except Exception as ex:
            raise Exception(f"exception while saving obj as pickle to '{self.file_path}'") from ex

    def load_from_pickle_file(self):
        """
        Loads an object from the specified pickle file
        """

        try:
            with open(self.file_path, "rb") as file:
                obj = pickle.load(file)
            return obj
        except Exception as ex:
            raise Exception(f"exception while loading pickle from '{self.file_path}'") from ex
