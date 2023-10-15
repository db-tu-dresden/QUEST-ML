import os
import timeit
from pathlib import Path

import numpy as np
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
        self.job_names = [job['name'] for job in self.sys_config['jobs']]

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

        parser.add_argument('--max_model_steps', metavar='N', type=int, default=100,
                            help='Maximum steps for the model to take.'
                                 'Note that one model step does NOT necessarily equate to one simulation step.'
                                 'The model step size is dependent on the config values of \'scaling_factor\''
                                 'and \'loggingRate\', the model step size is \'scaling_factor\' * \'loggingRate\'!')

        parser.add_argument('--max_simulations', metavar='N', type=int, default=5,
                            help='Maximum times for simulation to run')
        parser.add_argument('--max_simulation_steps', metavar='N', type=int, default=20,
                            help='Maximum steps for the simulation to take')

        parser.add_argument('--mutations', metavar='N', type=int, default=300,
                            help='Number of times the original job arrivals from the model are mutated. '
                                 'Used in method3 after initial job_arrivals are created '
                                 'by (backwards) model inference')
        parser.add_argument('--sub_mutations', metavar='N', type=int, default=1,
                            help='Number of times the original mutated state is subsequently mutated')
        parser.add_argument('--mutation_low', metavar='N', type=int, default=-3,
                            help='Low value of the uniform probability distribution used for mutation')
        parser.add_argument('--mutation_high', metavar='N', type=int, default=3,
                            help='Low value of the uniform probability distribution used for mutation')

        parser.add_argument('--print_quickest', metavar='N', type=int, default=3,
                            help='Print quickest N elements')

        parser.add_argument('--job_arrival_save_dir', type=str,
                            help='Directory where job arrival files should be saved. '
                                 'Defaults to save_dir + /job_arrivals/ , i.e. ./graphs/-/save/job_arrivals/')

    def mutate_state(self, state: torch.Tensor) -> torch.Tensor:
        shape = state.shape
        mutation = torch.distributions.Uniform(
            self.config['mutation_low'],
            self.config['mutation_high'],
        ).sample(shape).round()
        mutation += state
        mutation[mutation < 0] = 0
        return mutation

    def mutate_job_arrivals(self, job_arrivals: list[dict]) -> list[dict]:
        arrival_types = [job['type'] for job in job_arrivals]
        job_types = [job_type['name'] for job_type in self.sys_config['jobs']]
        job_dist = torch.tensor(
            [sum(1 for _type in arrival_types if _type == name) for name in job_types],
            dtype=torch.float)
        diff = self.mutate_state(job_dist) - job_dist

        for i, (count, job_type) in enumerate(zip(diff, job_types)):
            count = int(count)
            if count < 0:
                # remove
                job_type_count = sum(1 for job_arr in job_arrivals if job_arr['type'] == job_type)
                job_index = np.random.choice(range(job_type_count))

                # iterate over job arrivals; find job_index's occurrence of job with type job_name; delete it
                c = 0
                for i, job_arr in enumerate(job_arrivals):
                    if job_arr['type'] == job_type:
                        if c == job_index:
                            del job_arrivals[i]
                            break
                        c += 1

            if count > 0:
                # add
                new_arrivals = [
                    {
                        'time': time,
                        'type': job_type,
                    } for time in np.zeros(count)]

                job_arrivals = job_arrivals + new_arrivals
                job_arrivals.sort(key=lambda x: x['time'])

        return job_arrivals

    @staticmethod
    def shift_job_arrivals(job_arrivals: list[dict]) -> list[dict]:
        start = None
        for job_arr in job_arrivals:
            if start is None:
                start = job_arr['time']
            job_arr['time'] -= start
        return job_arrivals

    def get_job_arrivals(self, prev_state: torch.Tensor, curr_state: torch.Tensor, step: float) -> list[dict[str, int]]:
        prev_state = prev_state.round()
        curr_state = curr_state.round()
        diff = (curr_state - prev_state).sum(axis=1)
        return [{'time': step, 'type': name} for name, count in zip(self.job_names, diff) for _ in range(int(count))]

    @staticmethod
    def contains_tgt(state, target_dist) -> bool:
        return bool((state[-1] >= target_dist).all())

    def method1(self):
        print(f'Running simulation {self.config["max_simulations"]} times '
              f'with max {self.config["max_simulation_steps"]} steps '
              f'from initial state until target distribution is present')

        return system.simulate_to_target(
            self.sys_config, self.notation, self.initial_state, self.tgt_dist,
            k=self.config['max_simulations'],
            max_steps=self.config['max_simulation_steps'])

    def method2(self):
        print(f'Running forward model inference with max {self.config["max_model_steps"]} steps.\n'
              f'NOTE: this necessitates that the model was trained with a POSITIVE offset')

        state = self.initial_state.unsqueeze(0)         # add batch dim
        prev_state = state
        job_arrivals = []
        start = timeit.default_timer()

        with torch.no_grad():
            for step in range(self.config['max_model_steps']):
                if self.contains_tgt(state.squeeze(), self.tgt_dist):
                    break
                state = self.model(state)
                job_arrivals.extend(self.get_job_arrivals(prev_state.squeeze(), state.squeeze(), step))
                prev_state = state

        stop = timeit.default_timer()
        runtime = stop - start

        res = {
            'steps': step,
            'runtime': runtime,
            'initial_state': self.initial_state.numpy(),
            'final_state': state.numpy(),
            'job_arrivals': job_arrivals,
        }

        return [res]

    def method3(self):
        res = []

        print(f'Running (backward) model inference with max {self.config["max_model_steps"]} steps, '
              f'from a state with the target distribution to reach an initial state. '
              f'This initial state is used for further validation with the simulation '
              f'to reach a state with the target distribution.\n'
              f'NOTE: this necessitates that the model was trained with a NEGATIVE offset')

        start_state = torch.zeros(self.initial_state.shape)
        start_state[-1] = self.tgt_dist
        start_state = start_state.unsqueeze(0)      # add batch dim

        state = start_state
        prev_state = state

        model_step_size = self.config['scaling_factor'] * self.sys_config['loggingRate']

        job_arrivals = []
        start = timeit.default_timer()

        with torch.no_grad():
            # this is stepping backwards because model direction is negative
            for step in reversed(range(self.config['max_model_steps'])):
                if (state.round() == self.initial_state).all():
                    break
                prev_state = state
                state = self.model(state)
                job_arrivals.extend(
                    self.get_job_arrivals(state.squeeze(), prev_state.squeeze(), step=0.0))

        stop = timeit.default_timer()
        runtime = stop - start

        print(f'Reached state:\n'
              f'{state.numpy()}\n'
              f'rounded:\n'
              f'{state.round().numpy()}\n'
              f'after {self.config["max_model_steps"] - step} steps.')
        print(f'Created job arrivals are:\n'
              f'{yaml.dump(job_arrivals, indent=4)}')

        original_job_arrivals = job_arrivals

        self.sys_config['continueWithRndJobs'] = True
        self.sys_config['jobArrivalPath'] = None

        for _ in range(max(1, self.config['mutations'] + 1)):
            for _ in range(max(1, self.config['sub_mutations'] + 1)):
                res.extend(system.simulate_to_target(
                    self.sys_config, self.notation, self.initial_state, self.tgt_dist,
                    k=self.config['max_simulations'],
                    max_steps=self.config['max_simulation_steps'],
                    job_arrivals=job_arrivals))

                job_arrivals = self.mutate_job_arrivals(job_arrivals)
                job_arrivals = self.shift_job_arrivals(job_arrivals)
            job_arrivals = self.mutate_job_arrivals(original_job_arrivals)
            job_arrivals = self.shift_job_arrivals(job_arrivals)

        return res

    def evaluate(self, run_data: list[dict]):
        run_data = [elem for elem in run_data if elem['steps'] is not None]
        run_data.sort(key=lambda x: x['steps'])

        if len(run_data) == 0:
            print('No valid run found!')

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
