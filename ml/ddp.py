import os
import socket
import sys
from contextlib import closing

import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp

import const


def find_free_port(addr):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((addr, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def setup(rank, world_size, port):
    torch.backends.cuda.matmul.allow_tf32 = const.ALLOW_TF32

    os.environ['MASTER_ADDR'] = const.MASTER_ADDR
    os.environ['MASTER_PORT'] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    const.DEVICE = f'cuda:{rank}'
    torch.cuda.set_device(const.DEVICE)
    set_stdout()
    dist.barrier()


def cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def run(fn, world_size, arch_mode, module, experiment_name, port):
    mp.spawn(fn, args=(world_size, arch_mode, module, experiment_name, port,),
             nprocs=world_size, join=True)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def sync():
    if is_dist_avail_and_initialized():
        dist.barrier()


def set_stdout():
    if is_dist_avail_and_initialized():
        stdout_base = const.ROOT_DIR + '/../slurm-' + const.SLURM_JOB_ID
        stdout = f'{stdout_base}-{dist.get_rank()}.out'
        sys.stdout = open(stdout, 'a')
        sys.stderr = open(stdout, 'a')
