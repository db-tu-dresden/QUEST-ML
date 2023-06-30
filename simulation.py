from notation import Notation
from system import Config, Environment, System
import timing


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
    system.logger.plot()
    system.logger.save_df('./save/df.pkl')


if __name__ == '__main__':
    main()
