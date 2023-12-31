import os
import socket
import sys
from contextlib import closing

import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp

from ml.config import Config


def find_free_port(addr: str):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((addr, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def setup(rank: int, config: Config):
    torch.backends.cuda.matmul.allow_tf32 = config['allow_tf32']

    os.environ['MASTER_ADDR'] = config['master_addr']
    os.environ['MASTER_PORT'] = str(config['master_port'])
    os.environ['WORLD_SIZE'] = str(config['world_size'])
    os.environ['RANK'] = str(rank)
    config['rank'] = rank

    dist.init_process_group('nccl', rank=rank, world_size=config['world_size'])
    config['device'] = f'cuda:{rank}'
    torch.cuda.set_device(config['device'])

    if config['log_to_file']:
        set_stdout(config['output_file'])
    dist.barrier()


def cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def run(fn, config: Config, *args):
    print(f'Spawning {config["world_size"]} process{"es" if config["world_size"] == 1 else ""}...')
    mp.spawn(fn, args=(config, *args),
             nprocs=config['world_size'], join=True)


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def sync():
    if is_dist_avail_and_initialized():
        dist.barrier()


def set_stdout(stdout_base: str):
    if is_dist_avail_and_initialized():
        index = stdout_base.rindex('.')
        stdout = stdout_base[:index] + f'-{dist.get_rank()}' + stdout_base[index:]
        sys.stdout = open(stdout, 'a')
        sys.stderr = open(stdout, 'a')
