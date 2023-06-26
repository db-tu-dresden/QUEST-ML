from notation import Notation
from system.config import Config
from system.environment import Environment
from system.system import System


def main():
    path = 'graph_description.note'
    with open(path) as f:
        text = f.read()

    notation = Notation.parse(text)

    config = Config('config.yaml')

    env = Environment()

    system = System(config, notation, env=env)
    system.build()
    system.run()


if __name__ == '__main__':
    main()
