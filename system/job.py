import numpy as np

from system.config import Config
from system.environment import Environment


class JobType:
    def __init__(self, name: str, arrival_prob: float, failure_rate: float, env: Environment):
        self.name = name
        self.arrival_prob = arrival_prob
        self.failure_rate = failure_rate
        self.env = env

    def __eq__(self, other):
        return hasattr(other, 'name') and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(name={self.name!r}, arrival_prob={self.arrival_prob!r}, failure_rate={self.failure_rate!r})'


class JobTypeCollection:

    def __init__(self, types: {JobType}, env: Environment):
        super().__init__()
        self.types = types
        self.env = env

    def get_rand_job(self, job_id: int, rng: np.random.Generator):
        job_type = rng.choice(list(self.types), p=[job_type.arrival_prob for job_type in self.types])
        return Job(job_id, job_type, env=self.env)

    @classmethod
    def from_config(cls, config: Config, env: Environment):
        types = set(JobType(job['name'], job['arrivalProbability'], job['failureRate'], env=env)
                    for job in config['jobs'])
        return cls(types, env=env)

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(types={set(type for type in self.types)!r}, env={self.env!r})'


class Job:
    def __init__(self, id: int, type: JobType, env: Environment):
        super().__init__()
        self.id = id
        self.type = type
        self.env = env

    def service(self, rng: np.random.Generator, mean: float, std: float):
        t = rng.normal(loc=mean, scale=std)
        return self.env.timeout(t)

    def did_fail(self, rng: np.random.Generator):
        return self.type.failure_rate > rng.uniform()

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(id={self.id!r}, type={self.type!r})'
