from typing import List
import torch
import wandb
from ..datastructures.configs.wandbconfig import WandbConfig


class WandbLogger:

    def __init__(self, wandb_config: WandbConfig, init_config: dict) -> None:

        self.config = wandb_config

        if self.config.enabled:
            wandb.init(entity=self.config.entity,
                       project=self.config.prj_name,
                       notes=self.config.notes,
                       tags=self.config.tags,
                       config=init_config)

    def watch_model(self, model: torch.nn.Module):

        if self.config.enabled:
            wandb.watch(model, log="all", log_freq=100, log_graph=False)

    def log(self, log_dict: dict, commit=True, step=None):
        if self.config.enabled:
            if step is not None:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)

    def set_config_value(self, key: str, value):

        if self.config.enabled:
            wandb.config[key] = value

    def log_figure_as_img(self, key: str, figure, commit: bool = True, step=None):

        if self.config.enabled:
            if step is not None:
                wandb.log({key: wandb.Image(figure)}, commit=commit, step=step)
            else:
                wandb.log({key: wandb.Image(figure)}, commit=commit)
            print(f"succesfully logged figure {key}")

    def log_table(self, key: str, columns : List[str], data, commit: bool = True, step=None):
        if self.config.enabled:
            table = wandb.Table(columns=columns, data = data)
            if step is not None:
                wandb.log({key: table}, commit=commit, step=step)
            else:
                wandb.log({key: table}, commit=commit, step=step)

    def finish(self):
        if self.config.enabled:
            wandb.finish()
