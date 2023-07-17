import argparse
import os

from notation import Notation
from system import Config, Environment, System
import timing


parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-p', '--path', help='Path where a config.yaml describing the system and '
                                         'a graph_description.note describing the process graph lie.')


def get_notation(path: str):
    with open(os.path.join(path, 'graph_description.note')) as f:
        text = f.read()

    notation = Notation.parse(text)
    notation.draw(os.path.join(path, 'graph.png'), show=False)
    return notation


def run_simulation(config: Config, notation: Notation, path: str):
    env = Environment()

    system = System(config, notation, env=env)
    system.build()
    system.run()
    system.logger.plot(path=os.path.join(path, 'dist.png'), show=False)
    system.logger.save(os.path.join(path, 'df.pkl'))


def run(args):
    base_path = args.path
    notation = get_notation(base_path)

    config = Config(os.path.join(base_path, 'config.yaml'))

    states = [
        {
            'dir': 'train',
            'steps': 10000
        },
        {
            'dir': 'valid',
            'steps': 2000
        },
        {
            'dir': 'test',
            'steps': 10000
        }
    ]

    for state in states:
        path = os.path.join(base_path, 'data', state['dir'])
        if not os.path.exists(path):
            os.makedirs(path)

        config['until'] = state['steps']

        run_simulation(config, notation, path)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
