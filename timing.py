import atexit
from datetime import timedelta
from time import time, strftime, localtime


TIME = {
    'start': 0.0,
    'end': 0.0,
}


def seconds_to_str(elapsed: float = None):
    if elapsed is None:
        return strftime('%Y-%m-%d %H:%M:%S', localtime())
    else:
        return str(timedelta(seconds=elapsed))


def log(s, elapsed: str = None):
    line = '=' * 40
    print(line)
    print(seconds_to_str(), '-', s)
    if elapsed:
        print('Elapsed time:', elapsed)
    print(line)
    print()


def end_log():
    TIME['end'] = time()
    elapsed = TIME['end'] - TIME['start']
    log('End Program', seconds_to_str(elapsed))


def log_runtime(func):
    def inner(*args, **kwargs):
        TIME['start'] = time()
        atexit.register(end_log)
        log('Start Program')
        func(*args, **kwargs)

    return inner
