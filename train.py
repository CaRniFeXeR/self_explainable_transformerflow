from src.pipeline.Trainer import Trainer
from pathlib import Path
from src.pipeline.ConfigParser import ConfigParser
import torch
torch.multiprocessing.set_start_method('spawn', force=True)


if __name__ == "__main__":

    configParser = ConfigParser()

    config = configParser.parse_config_from_file(Path(".//config//train//train_vie14_val_bln.json"))

    print(config)

    trainer = Trainer(config)
    trainer.train()
