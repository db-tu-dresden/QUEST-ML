from ml.config import Config
from ml.data import ProcessDataset
from ml.logger import Logger
from ml.models.base import Model, DistributedDataParallel
from ml.parser import Parser
from ml.trainer import Trainer


def seed(seed: int = 0, deterministic: bool = True):
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(mode=deterministic)
    pass
