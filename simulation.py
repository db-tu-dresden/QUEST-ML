from notation import Notation
from system import Config, Environment, System
import timing


def main():
    base_path = './save/<>/'

    path = 'graph_description.note'
    with open(path) as f:
        text = f.read()

    notation = Notation.parse(text)
    notation.draw(base_path + 'graph.png', show=False)

    config = Config('config.yaml')

    env = Environment()

    system = System(config, notation, env=env)
    system.build()
    system.run()
    system.logger.plot(path=base_path + 'dist.png', show=False)
    system.logger.save_df(base_path + 'df.pkl')


if __name__ == '__main__':
    main()
