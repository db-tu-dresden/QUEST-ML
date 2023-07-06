import argparse
import os.path

from notation import Notation
from system import Config, Environment, System
import timing


parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-p', '--path', help='Path where a config.yaml describing the system and '
                                         'a graph_description.note describing the process graph lie.',
                    default='./save/<>')


def run(args):
    base_path = args.path

    with open(os.path.join(base_path, 'graph_description.note')) as f:
        text = f.read()

    notation = Notation.parse(text)
    notation.draw(os.path.join(base_path, 'graph.png'))

    config = Config(os.path.join(base_path, 'config.yaml'))

    env = Environment()

    system = System(config, notation, env=env)
    system.build()
    system.run()
    system.logger.plot(path=os.path.join(base_path, 'dist.png'))


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
