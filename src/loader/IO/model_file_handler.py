import torch
from pathlib import Path
from datetime import datetime
from...datastructures.configs.modelstorageconfig import ModelStorageConfig


class ModelFileHandler:
    """
    Handles pytorch model state saving and loading
    """

    def __init__(self, modelstorage_config: ModelStorageConfig) -> None:
        self.config = modelstorage_config
        folder = self.config.file_path
        if folder.exists() and folder.is_file():
            folder = folder.parent
        elif not folder.exists():
            folder.mkdir()
        self.path = folder / Path(f"{datetime.now().strftime('%d_%m_%Y %H_%M')}.pt")

    def save_model_state_to_file(self, model: torch.nn.Module):
        torch.save(model.state_dict(), self.path)
        print(f"successfully saved model state to {self.path}")

    def load_state_from_file(self, model: torch.nn.Module, use_config_path: bool) -> torch.nn.Module:

        path = self.config.file_path if use_config_path else self.path

        model.load_state_dict(torch.load(path, map_location= self.config.gpu_name))
        return model
