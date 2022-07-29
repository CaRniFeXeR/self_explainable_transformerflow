from dataclasses import dataclass

import torch


@dataclass
class LossResult:
    point_loss: torch.FloatTensor
