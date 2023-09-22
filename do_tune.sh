#!/usr/bin/env bash

trap '
  trap - INT # restore default INT handler
  kill -s INT "$$"
' INT

GPUS=0.2
DATA_PATH=./save/-/
ARCH=flat_mlp

TUNE_CONFIG_PATH=./ml/scripts/tune_config.py

list=(1 3 5 10 20 30)

for item in "${list[@]}"
do
  echo """
import numpy as np
from ray import tune


TUNE_CONFIG = {
    'accumulation_window': $item,
    'hidden_size': tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    'hidden_layers': tune.sample_from(lambda _: np.random.randint(2, 4)),
    'learning_rate': tune.loguniform(1e-4, 1e-1),
}
""" > $TUNE_CONFIG_PATH

  echo """
Running with config:

TUNE_CONFIG = {
  'accumulation_window': $item,
  'hidden_size': tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
  'hidden_layers': tune.sample_from(lambda _: np.random.randint(2, 4)),
  'learning_rate': tune.loguniform(1e-4, 1e-1),
}
"""

  python3 tune.py --path $DATA_PATH --arch $ARCH --load_datasets --gpus $GPUS -s 1 \
  | grep --after-context 2 'Best trial config' \
  > "$TUNE_RESULTS_PATH"
done
