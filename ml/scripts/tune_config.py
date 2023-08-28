import numpy as np
from ray import tune


TUNE_CONFIG = {
    'hidden_size': tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    'hidden_layers': tune.sample_from(lambda _: np.random.randint(2, 4)),
    'learning_rate': tune.loguniform(1e-4, 1e-1),
    'batch_size': tune.choice([2, 4, 8, 16])
}
