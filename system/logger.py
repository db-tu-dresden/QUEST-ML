from system.job import Job


class Logger:
    def __init__(self, system):
        self.system = system
        self.log = {}

        self.default_dist = {job_type.name: 0 for job_type in self.system.job_types.types}

    def get_job_dist(self, jobs: [Job]):
        dist = dict(self.default_dist)
        for job in jobs:
            dist[job.type.name] += 1
        return dist

    def log_processes(self):
        job_dists = {}
        for _, process in self.system.processes.items():
            if process.queue:
                job_dists[process.id] = self.get_job_dist(process.queue.items)
        self.log[round(self.system.env.now, 1)] = job_dists

    def process(self):
        self.log_processes()

        yield self.system.env.timeout(0.1)

    def run(self):
        while True:
            yield from self.process()
