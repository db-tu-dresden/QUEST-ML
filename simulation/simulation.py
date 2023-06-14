import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simpy


class Job:
    def __init__(self, job_id: int, job_type: str, failure_rate: float):
        self.id = job_id
        self.log = {}
        self.job_type = job_type
        self.failure_rate = failure_rate

    def __str__(self) -> str:
        return f'{self.id}({self.job_type})'


class Queue(simpy.Store):
    def __init__(self, env, capacity=1, name='default', discipline='FIFO'):
        super().__init__(env, capacity)
        self.name = name
        self.log = {}
        self.state = {}
        self.discipline = discipline  # is default of store object


def arrival_process(env, queue, system_logs, arrival_log, job_limit=100, para=1.0):
    rng = np.random.default_rng(seed=42)
    job_id = 0
    while job_id < job_limit:
        job_id += 1
        rand_type = rng.choice(["A", "B", "C"], p=[0.1, 0.1, 0.8])
        t = rng.exponential(scale=para)
        yield env.timeout(t)

        job = Job(job_id, rand_type, failure_rate=0.0)

        # print(f'[Time: {"%.3f" % env.now}] - Job {job.id} has been assigned to be next processed in {queue.name}.')
        system_logs.append({f'Job {job} has been assigned to be be next processed in {queue.name}.': env.now})
        job.log.update({f'Assinged to {queue.name}': env.now})

        yield queue.put(job)
        arrival_log.update({env.now: str(job)})

        # print(f'[Time: {"%.3f" % env.now}] - Job {job.id} has been put to queue of {queue.name} and is waiting for service.')
        system_logs.append({f'Job {job} has been put to queue of {queue.name} and is waiting for service.': env.now})
        job.log.update({f'Waiting in queue of {queue.name}': env.now})
        queue.log.update({f'Job {job} has accessed waiting queue in {queue.name}': env.now})


def service_process(env, queue_process, system_logs, completed_jobs, queue_next: dict = None, job_limit=0,
                    break_event=None, output_queue=None, out_log=None, distr='exponential', para=1.0):
    rng = np.random.default_rng(seed=42)
    while True:
        job = yield queue_process.get()

        # print(f'[Time: {"%.3f" % env.now}] - Job {job.id} has been removed from queue and is being serviced in {queue_process.name}.')
        system_logs.append({f'Job {job} has been removed from queue and is being serviced in {queue_process.name}.': env.now})
        job.log.update({f'Start servicing in QS {queue_process.name}': env.now})
        queue_process.log.update({f'Job {job} started serving in {queue_process.name} ': env.now})

        t = rng.normal(loc=para, scale=0.2)
        yield env.timeout(t)

        # print(f'[Time: {"%.3f" % env.now}] - Job {job.id} has finished being serviced in {queue_process.name}.')
        system_logs.append({f'Job {job} has finished being serviced in {queue_process.name}.': env.now})
        job.log.update({f'Finish servicing in {queue_process.name}': env.now})
        queue_process.log.update({f'Job {job} finished serving in {queue_process.name} ': env.now})

        qm = rng.uniform()
        if job.failure_rate > qm:
            system_logs.append({f'Job {job} does not meet quality after {queue_process.name}.': env.now})
            job.log.update({f'Failed quality check after {queue_process.name}': env.now})
            queue_process.log.update({f'Job {job} failed quality check after {queue_process.name} ': env.now})

            yield queue_process.put(job)

            system_logs.append({f'Job {job} has left {queue_process.name} and has been reinserted to waiting queue in '
                                f'{queue_process.name}.': env.now})
            job.log.update({f'Transferred to {queue_process.name}': env.now})
            queue_process.log.update({f'Job {job} has been reinsert into {queue_process.name} ': env.now})

        if queue_next is None:
            # print(f'[Time: {"%.3f" % env.now}] - Job {job.id} has been terminated from {queue_process.name}.')
            system_logs.append({f'Job {job} has been terminated from {queue_process.name}.': env.now})
            job.log.update({f'Leaving {queue_process.name}': env.now})
            out_log.update({env.now: str(job)})
            queue_process.log.update({f'Job {job} left {queue_process.name} ': env.now})

            completed_jobs.append((env.now, job))
            output_queue.put(job)

            if len(completed_jobs) == job_limit:
                break_event.succeed()
        else:
            # print(f'[Time: {"%.3f" % env.now}] - Job {job.id} is waiting after service in {queue_process.name} to access {queue_next[job.job_type].name}.')
            system_logs.append({f'Job {job} is waiting after service in {queue_process.name} to access '
                                f'{queue_next[job.job_type].name}.': env.now})
            job.log.update({f'Completed at {queue_process.name}': env.now})
            job.log.update({f'Waiting for access to {queue_next[job.job_type].name}': env.now})

            yield queue_next[job.job_type].put(job)

            # print(f'[Time: {"%.3f" % env.now}] - Job {job.id} has left {queue_process.name} and been put to waiting queue in {queue_next[job.job_type].name}.')
            system_logs.append({f'Job {job} has left {queue_process.name} and been put to waiting queue in '
                                f'{queue_next[job.job_type].name}.': env.now})
            job.log.update({f'Transferred to {queue_next[job.job_type].name}': env.now})
            queue_process.log.update({f'Job {job} left {queue_process.name} ': env.now})


def logging_process(env, queue):
    while True:
        queue.state.update({round(env.now, 1): [str(i) for i in queue.items]})
        yield env.timeout(0.1)


def run_mm1_experiment(job_limit=None, steps=None):
    if job_limit is None and steps is None:
        raise Exception('Give a job limit or a number of steps to simulate!')

    completed_jobs = []
    system_logs = []
    arrival_log = {}
    out_log = {}

    env = simpy.Environment()
    break_event = env.event()

    # Entities
    queueing_system_1 = Queue(env, capacity=np.inf, name='Queueing System 1')
    queueing_system_2 = Queue(env, capacity=np.inf, name='Queueing System 2')
    queueing_system_3 = Queue(env, capacity=np.inf, name='Queueing System 3')
    queueing_system_4 = Queue(env, capacity=np.inf, name='Queueing System 4')
    queueing_system_5 = Queue(env, capacity=np.inf, name='Queueing System 5')
    queueing_system_6 = Queue(env, capacity=np.inf, name='Queueing System 6')

    out = Queue(env, capacity=np.inf, name='Out')

    # Arrival (system input) process
    env.process(arrival_process(env, queueing_system_1, system_logs, arrival_log, job_limit=job_limit))

    # tool processes (and logging)
    env.process(service_process(env, queueing_system_1, system_logs, completed_jobs,
                                queue_next={'A': queueing_system_2, 'B': queueing_system_2,
                                            'C': queueing_system_5}))
    env.process(logging_process(env, queueing_system_1))

    env.process(service_process(env, queueing_system_2, system_logs, completed_jobs,
                                queue_next=dict.fromkeys(['A', 'B'], queueing_system_3)))
    env.process(logging_process(env, queueing_system_2))

    env.process(service_process(env, queueing_system_3, system_logs, completed_jobs,
                                queue_next=dict.fromkeys(['A', 'B'], queueing_system_4)))
    env.process(logging_process(env, queueing_system_3))

    env.process(service_process(env, queueing_system_4, system_logs, completed_jobs, job_limit=job_limit,
                                break_event=break_event, output_queue=out, out_log=out_log))  # system output
    env.process(logging_process(env, queueing_system_4))

    env.process(
        service_process(env, queueing_system_5, system_logs, completed_jobs, queue_next={'C': queueing_system_6}))
    env.process(logging_process(env, queueing_system_5))

    env.process(
        service_process(env, queueing_system_6, system_logs, completed_jobs, queue_next={'C': queueing_system_4}))
    env.process(logging_process(env, queueing_system_6))

    env.process(logging_process(env, out))

    # Run
    if steps:
        env.run(until=steps)
    else:
        env.run(until=break_event)

    results = (queueing_system_1.state, queueing_system_2.state, queueing_system_3.state, queueing_system_4.state,
               queueing_system_5.state, queueing_system_6.state, out.state)
    return results, completed_jobs, system_logs, arrival_log, out_log


def parse_log(log, precision=1):
    new = {}
    for key, job in log.items():
        new.update({round(key, precision): job})
    return new


def fill_log(log: dict, default_value='', start=0, stop=None, precision=1):
    multiplier = precision * 10
    if stop is None:
        stop = int(list(log.keys())[-1] * multiplier)

    for step in range(start, stop):
        step = round(step / multiplier, precision)
        value = log.get(step, default_value)
        log.update({step: value})
    return dict(sorted(log.items()))


def plot(exp_res):
    fig, axs = plt.subplots(figsize=(16, 9), ncols=4, nrows=2)
    for res, ax in zip(exp_res, axs.ravel()):
        # df = pd.DataFrame(res.items(), columns=['TimeStamp','Objects in Queue'])
        ax.plot(res.keys(), [len(r) for r in res.values()])
    plt.show()


def main():
    exp_res, completed_jobs, system_logs, in_log, out_log = run_mm1_experiment(job_limit=100)

    out_log = parse_log(out_log)
    out_log = fill_log(out_log)

    in_log = parse_log(in_log)
    in_log = fill_log(in_log, stop=len(out_log))

    df = pd.DataFrame.from_dict({
        'step': out_log.keys(),
        'in': in_log.values(),
        'out': out_log.values(),
    })

    plot(exp_res)


if __name__ == '__main__':
    main()
