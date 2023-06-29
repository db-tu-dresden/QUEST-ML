import atexit
from time import time, strftime, localtime
from datetime import timedelta


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
    end = time()
    elapsed = end - start
    log('End Program', seconds_to_str(elapsed))


start = time()
atexit.register(end_log)
log('Start Program')
