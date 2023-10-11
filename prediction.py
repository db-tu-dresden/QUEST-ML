import os
import timeit
from pathlib import Path

import torch
import yaml

import ml
import notation
import system


class Predictor:
    def __init__(self):
        cwd = os.path.dirname(os.path.realpath(__file__))
        self.config = ml.Config(os.path.join(cwd, 'ml/config.yaml'))

        parser = ml.Parser(self.config)
        self.add_parser_arguments(parser)
        args = parser.parse_args()
        self.config.update_from_args(args)
        self.config['load_model'] = True
        self.config['job_arrival_save_dir'] = (self.config['job_arrival_save_dir'] or
                                               os.path.join(self.config['save_dir'], 'job_arrivals'))

        self.model = ml.models.build_model(self.config)
        self.model.load(self.config)

        self.sys_config = system.Config(os.path.join(self.config['base_path'], 'config.yaml'))

        notation_path = os.path.join(self.config['base_path'], 'graph_description.note')
        with open(notation_path) as f:
            notation_string = f.read()
        self.notation = notation.Notation.parse(notation_string)

        self.inf_config = ml.InferenceConfig(os.path.join(self.config['base_path'], 'inference_config.yaml'))
        self.initial_state = torch.tensor(self.inf_config['initialState'], dtype=torch.float, requires_grad=False)
        self.tgt_dist = torch.tensor(self.inf_config['targetDist'], dtype=torch.float, requires_grad=False)

    @staticmethod
    def add_parser_arguments(parser: ml.Parser):
        parser.add_argument('--method', type=str, required=True,
                            choices=['method1', 'method2', 'method3'],
                            help='Methode to be run')

        parser.add_argument('--k_model', metavar='N', type=int, default=1,
                            help='K for k-fold validation of the model')

        parser.add_argument('--max_simulations', metavar='N', type=int, default=1000,
                            help='Maximum times for simulation to run')

        parser.add_argument('--mutation_low', metavar='N', type=int, default=-2,
                            help='Low value of the uniform probability distribution used for mutation')
        parser.add_argument('--mutation_high', metavar='N', type=int, default=2,
                            help='Low value of the uniform probability distribution used for mutation')

        parser.add_argument('--print_quickest', metavar='N', type=int, default=3,
                            help='Print quickest N elements')

        parser.add_argument('--job_arrival_save_dir', type=str,
                            help='Directory where job arrival files should be saved. '
                                 'Defaults to save_dir + /job_arrivals/ , i.e. ./graphs/-/save/job_arrivals/')

    def mutate(self, state: torch.Tensor) -> torch.Tensor:
        shape = state.shape
        mutation = torch.distributions.Uniform(
            self.config['mutation_low'],
            self.config['mutation_high'],
        ).sample(shape).round()
        mutation[mutation < 0] = 0
        return state + mutation

    @staticmethod
    def contains_tgt(state, target_dist) -> bool:
        return bool((state[0, -1] >= target_dist).all())

    def method1(self):
        sim_data = system.simulate_to_target(
            self.sys_config, self.notation, self.initial_state, self.tgt_dist,
            k=self.config['max_simulations'],
            max_steps=self.config['max_simulation_steps'])
        return sim_data

    def method2(self):
        return []

    def method3(self):
        return []

    def evaluate(self, run_data: list[dict]):
        run_data.sort(key=lambda x: x['steps'])

        for i in range(self.config['print_quickest']):
            if len(run_data) <= i:
                break
            print(f'\n\n'
                  f'{i + 1}. quickest:')
            if not self.contains_tgt(run_data[i]['final_state'], self.tgt_dist.numpy()):
                print('did NOT include target distribution!')
            print(f'initial state: \n{run_data[i]["initial_state"]}\n'
                  f'final state: \n{run_data[i]["final_state"]}\n'
                  f'steps: {run_data[i]["steps"]:.{4}f}\n'
                  f'runtime: {run_data[i]["runtime"]:.{7}f} sec\n')

            job_arrival_path = os.path.join(self.config['job_arrival_save_dir'], f'res_{i + 1}.yaml')
            Path(self.config['job_arrival_save_dir']).mkdir(parents=True, exist_ok=True)
            with open(job_arrival_path, 'w') as f:
                yaml.dump(run_data[i]['job_arrivals'], f)

            print(f'saved job arrivals to {job_arrival_path}')

        return

    def run(self):
        res = None
        start = timeit.default_timer()

        if self.config['method'] == 'method1':
            res = self.method1()
        if self.config['method'] == 'method2':
            res = self.method2()
        if self.config['method'] == 'method3':
            res = self.method3()

        stop = timeit.default_timer()
        runtime = stop - start

        print(f'Whole runtime was {runtime:.{3}f} sec')

        self.evaluate(res)


if __name__ == '__main__':
    predictor = Predictor()
    predictor.run()
