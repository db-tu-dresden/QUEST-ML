import os

import torch

from ml import Config as MLConfig, Parser
from ml.models import build_model
from ml.recommender import Recommender
from system import Config as SysConfig


def run():
    cwd = os.path.dirname(os.path.realpath(__file__))
    ml_config = MLConfig(os.path.join(cwd, 'ml/config.yaml'))

    parser = Parser(ml_config)
    parser.add_argument('--tgt', nargs='+', type=int, required=True, help='The target distribution')
    args = parser.parse_args()

    ml_config.update_from_args(args)
    ml_config['load_model'] = True

    model = build_model(ml_config)
    model.load(ml_config)

    sys_config = SysConfig(os.path.join(ml_config['base_path'], 'config.yaml'))

    recommender = Recommender(ml_config, model,
                              target_dist=torch.tensor(args.tgt),
                              initial_state=torch.zeros(len(sys_config['processes']) + 1, len(sys_config['jobs'])))
    recommender.predict()


if __name__ == '__main__':
    run()
