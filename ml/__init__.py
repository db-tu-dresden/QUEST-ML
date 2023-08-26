import random

import numpy as np
import torch

from ml.config import Config
from ml.data import ProcessDataset
from ml.logger import Logger
from ml.models.base import Model, DistributedDataParallel
from ml.parser import Parser
from ml.trainer import Trainer


def seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
