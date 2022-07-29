import torch
import random
import numpy as np

def setSeeds(seed : int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

